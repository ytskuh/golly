// This file is part of Golly.
// See docs/License.html for the copyright notice.

/*
 *   Generations of QuickLife universes on a CUDA GPU.
 *
 *   The universe is a set of tiles of TS x TS cells, found by position in
 *   a hash table on the GPU.  Cells are bits, 32 to a word, row by row; the
 *   most significant bit of a word is its leftmost cell.  The coordinates
 *   are those of the QuickLife tree for even generations (the tree stores
 *   odd generations shifted by one cell in x and y, see qlifealgo.h).
 *   Missing tiles are empty.
 *
 *   Each tile holds two sets of its cells, and each set two generations:
 *   the last one computed of each parity.  The generations run in epochs of
 *   at most HALO (32) generations.  In an epoch, each tile to compute reads
 *   its cells and the HALO cells around them from one set of each tile
 *   (neighbors included), computes the generations in shared memory, and
 *   writes its cells to its other set; so a tile never writes cells that
 *   another one reads in the same epoch.  The cells within HALO of the
 *   region's edge go wrong as the generations proceed (cells beyond the
 *   region count as empty), but a cell changes only cells next to it in a
 *   generation, so the tile's own cells are exact after HALO generations.
 *
 *   Change tracking.  A tile is divided into TB x TB blocks of 32x32 cells.  A
 *   block "changed" in generation g if its cells differ from generation
 *   g-2.  In generation g+1 a block is computed only if it or one of its 8
 *   neighbors changed in generation g; otherwise its cells of generation
 *   g+1 equal those of g-1, which its buffer for that parity still holds.
 *   Each tile keeps the change bits of its blocks in its last generation
 *   (`mask', TB*TB bits).  Changes spread one cell per generation, so in an
 *   epoch a tile can change only if it or one of the blocks of its
 *   neighbors along its edges changed at the end of the last epoch.  So a
 *   tile, when it has computed an epoch, lists itself for the next one if
 *   it changed, and the neighbors next to its changed edge blocks (creating
 *   them if missing).  Every SWEEP epochs, the tiles that are empty, did
 *   not change and are not listed are freed.
 *
 *   Each epoch is one kernel, qgstep, with one thread block per listed
 *   tile.  The last block to finish checks that the next epoch has room for
 *   the tiles it may create; if not, the epochs stop there and the host
 *   enlarges the tables and resumes.
 */
#include "qlifecuda.h"
#include "util.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <unordered_set>
#include <climits>
#include <future>
#include <mutex>

typedef unsigned long long u64 ;

#ifndef QLG_LOGTS
#define QLG_LOGTS 7
#endif
static const int LOGTS = QLG_LOGTS ;     // tile side in cells: TS = 2^LOGTS (128 or 256)
static const int TS = 1 << LOGTS ;
static const int LOGTW = LOGTS - 5 ;
static const int TW = TS / 32 ;          // words per tile row
static const int TB = TW ;               // blocks of 32x32 cells per tile side
static const int TWORDS = TS * TW ;      // words of one generation of a tile
static const int HALO = 32 ;             // most generations per epoch
static const int RS = TS + 2 * HALO ;    // side of the region an epoch computes
static const int RW = RS / 32 ;          // words per region row
static const int NB = RS / 32 ;          // blocks per region side
// shared memory row: a zero word, then RW words; and a zero row above and
// below the region (so reading the 3x3 words around any word needs no test)
static const int RP = RW + 1 ;
static const int SBW = (RS + 2) * RP + 1 ;
static const int NWARP = NB > 8 ? 16 : 8 ;   // warps per block of qgstep, a power of 2 > NB
static const int BS = 32 * NWARP ;
static const int NLD = (2 * RS * RW + BS - 1) / BS ;   // words of the region each thread loads
// bricks a tile may write in qgsave: its own, and those of missing tiles
// on its left, above it and above left
static const int SAVEOWN = TW * TS / 8 ;
static const int SAVEN = SAVEOWN + TS / 8 + TW + 1 ;
static const int EPOCHS = 64 ;
static const int SWEEP = 16 ;            // epochs between frees of tiles
// blocks of qgstep: GROW times the tiles listed at the last synchronization
// (blocks without a tile cost time, and a block with two tiles takes twice
// as long), at most as many as fit on the GPU at once
#ifndef QLG_GROW
#define QLG_GROW 1.25
#endif
static const double GROW = QLG_GROW ;           // epochs queued between synchronizations
static const int HLOAD = 4 ;             // hash table entries per entry used, at least

static const u64 KEMPTY = ~0ULL, KTOMB = ~0ULL - 1 ;
static const int KOFF = 0x40000000 ;

#define CK(x) do { cudaError_t ce = (x) ; if (ce != cudaSuccess) qgfail(ce, __LINE__) ; } while (0)
static void qgfail(cudaError_t e, int line) {
   static char msg[200] ;
   snprintf(msg, sizeof(msg), "QuickLife CUDA: %s (qlifecuda.cu:%d)", cudaGetErrorString(e), line) ;
   lifefatal(msg) ;
}

// counters shared by the kernels and the host
struct qlctl {
   int count[2] ;      // lengths of the tile lists of even and odd epochs
   int top ;           // slots ever used
   int nfree ;         // free slots on the stack
   int npop ;          // slots taken from the stack in this epoch
   int nfreed ;        // slots freed in this epoch (added to the stack at its end)
   int hused ;         // hash entries used, including removed ones
   int stopat ;        // if not 0, the first epoch not to run (no room)
   int done ;          // blocks of qgstep finished
   int sdone ;         // blocks of qgsweep finished
   int epochs ;        // epochs computed
   int nout ;          // bricks written by qgsave
} ;

// hash table entry: a tile key and its slot
struct __align__(16) qlhent {
   u64 key ;
   int val, pad ;
} ;

// per slot
struct __align__(16) qlmeta {
   u64 mask[2] ;       // by epoch parity: change bits of the blocks after that epoch
   int mstamp[2] ;     // ... and the epoch they belong to
   int st ;            // epoch << 1 | set: the set written by the last epoch that computed the tile
   int empty ;         // that epoch left the tile empty
} ;

// the GPU tables, passed to the kernels by value
struct qlg {
   unsigned *data ;       // [slot][set][parity][row][word]
   u64 *pos ;             // key of the tile in each slot, KEMPTY if free
   qlmeta *meta ;
   int *want ;            // last epoch that listed the tile
   int *list[2] ;         // tiles to compute in even and odd epochs
   int *freestack, *freed ;
   qlhent *hash ;
   qlctl *ctl ;
   int cap, hmask, hbits ;
} ;

struct qlgpu {
   qlg g ;
   qlctl *hctl ;          // pinned copy of *g.ctl
   cudaStream_t stream ;
   int epoch ;            // number of the next epoch
   int nsm ;              // multiprocessors
   int perstep ;          // blocks of qgstep that fit on a multiprocessor
   int lastcount ;        // tiles listed for the next epoch, at the last synchronization
} ;

__host__ __device__ inline u64 qgkey(int tx, int ty) {
   return ((u64)(unsigned)(ty + KOFF) << 32) | (unsigned)(tx + KOFF) ;
}
__device__ inline int qgkx(u64 k) { return (int)(unsigned)(k & 0xffffffffu) - KOFF ; }
__device__ inline int qgky(u64 k) { return (int)(unsigned)(k >> 32) - KOFF ; }
__device__ inline unsigned qghash(u64 k, int bits) {
   return (unsigned)((k * 0x9E3779B97F4A7C15ULL) >> (64 - bits)) ;
}
__device__ inline size_t qgoff(int s, int set, int par) {
   return ((size_t)s * 4 + set * 2 + par) * TWORDS ;
}
// change bits of a tile after epoch e
__device__ inline u64 qgmask(const qlmeta &m, int e) {
   return m.mstamp[e & 1] == e ? m.mask[e & 1] : 0 ;
}

// neighbors in the order NW, N, NE, W, (self), E, SW, S, SE: index (dy+1)*3 + dx+1
// edge blocks of a tile next to the neighbor in each direction
constexpr u64 qgrow(int i) { return ((1ULL << TB) - 1) << (TB * i) ; }
constexpr u64 qgcol(int j) { return TB == 8 ? 0x0101010101010101ULL << j : 0x1111ULL << j ; }
__constant__ u64 qgedge[9] = { 1ULL, qgrow(0), 1ULL << (TB - 1),
                               qgcol(0), 0, qgcol(TB - 1),
                               1ULL << (TB * (TB - 1)), qgrow(TB - 1), 1ULL << (TB * TB - 1) } ;

// the slot of a tile, or -1.  Other threads may be adding tiles: a tile
// being added may be missed, which is right as long as new tiles are empty
__device__ int qgfind(const qlg &g, u64 key) {
   unsigned h = qghash(key, g.hbits) ;
   for (;;) {
      qlhent en = g.hash[h] ;
      if (en.key == key)
         return en.val ;
      if (en.key == KEMPTY)
         return -1 ;
      h = (h + 1) & g.hmask ;
   }
}

// a slot for a new tile; the free stack does not change while slots are taken
__device__ int qgalloc(const qlg &g) {
   int i = atomicAdd(&g.ctl->npop, 1) ;
   int nf = g.ctl->nfree ;
   if (i < nf)
      return g.freestack[nf - 1 - i] ;
   return atomicAdd(&g.ctl->top, 1) ;
}

// the slot of the tile with this key; if missing, it is created empty, as
// if last written in epoch e-1 (so readers in epochs e and e+1 read its
// empty set)
__device__ int qginsert(const qlg &g, u64 key, int e) {
   unsigned h = qghash(key, g.hbits) ;
   for (;;) {
      u64 k = ((volatile u64 *)&g.hash[h].key)[0] ;
      if (k == KEMPTY) {
         k = atomicCAS(&g.hash[h].key, KEMPTY, key) ;
         if (k == KEMPTY) {
            int s = qgalloc(g) ;
            g.pos[s] = key ;
            // a free slot's current set is empty (a new slot's both sets)
            g.meta[s].st = ((e - 1) << 1) | (g.meta[s].st & 1) ;
            g.meta[s].empty = 1 ;
            // readers in this kernel need only the slot number
            ((volatile int *)&g.hash[h].val)[0] = s ;
            atomicAdd(&g.ctl->hused, 1) ;
            return s ;
         }
      }
      if (k == key) {
         int v ;
         while ((v = ((volatile int *)&g.hash[h].val)[0]) < 0)
            ;
         return v ;
      }
      h = (h + 1) & g.hmask ;
   }
}

__device__ inline void qgwant(const qlg &g, int s, int e) {
   if (atomicExch(&g.want[s], e) != e)
      g.list[e & 1][atomicAdd(&g.ctl->count[e & 1], 1)] = s ;
}

__device__ inline int qgstopped(const qlg &g, int e) {
   int s = g.ctl->stopat ;
   return s && e >= s ;
}

/*
 *   Free the tiles that are empty, did not change in epoch e-1 and are not
 *   listed for epoch e (between the epochs e-1 and e).  The last block to
 *   finish adds the freed slots to the stack.
 */
__global__ void qgsweep(qlg g, int e) {
   if (qgstopped(g, e))
      return ;
   int top = g.ctl->top ;
   for (int x = blockIdx.x * blockDim.x + threadIdx.x ; x < top ; x += gridDim.x * blockDim.x) {
      u64 key = g.pos[x] ;
      if (key == KEMPTY)
         continue ;
      qlmeta m = g.meta[x] ;
      if (m.empty && !qgmask(m, e - 1) && g.want[x] != e) {
         unsigned h = qghash(key, g.hbits) ;
         while (g.hash[h].key != key)
            h = (h + 1) & g.hmask ;
         g.hash[h].key = KTOMB ;
         g.pos[x] = KEMPTY ;
         g.freed[atomicAdd(&g.ctl->nfreed, 1)] = x ;
      }
   }
   __syncthreads() ;
   if (threadIdx.x == 0) {
      __threadfence() ;
      if (atomicAdd(&g.ctl->sdone, 1) == (int)gridDim.x - 1) {
         volatile qlctl *c = g.ctl ;
         const volatile int *fr = g.freed ;
         c->sdone = 0 ;
         int nf = c->nfree, nfd = c->nfreed ;
         for (int i = 0 ; i < nfd ; i++)
            g.freestack[nf++] = fr[i] ;
         c->nfree = nf ;
         c->nfreed = 0 ;
      }
   }
}

// the new cells from the 2-bit sums (x0 + 2 x1) of each cell and its left
// and right neighbors in the rows above (u), at (a) and below (d) them
__device__ __forceinline__ unsigned qgrule(unsigned u0, unsigned u1, unsigned a0, unsigned a1,
                                           unsigned d0, unsigned d1, unsigned w,
                                           unsigned born, unsigned stays, int life) {
   // T = s0 + 2 v0 + 4 v1 + 8 v2: the live cells of the 3x3 block
   unsigned s0 = u0 ^ a0 ^ d0, c0 = (u0 & a0) | (d0 & (u0 ^ a0)) ;
   unsigned p = u1 ^ a1 ^ d1, q = (u1 & a1) | (d1 & (u1 ^ a1)) ;
   unsigned v0 = p ^ c0, k = p & c0 ;
   unsigned v1 = q ^ k, v2 = q & k ;
   if (life)
      return ~v2 & ((s0 & v0 & ~v1) | (w & ~s0 & ~v0 & v1)) ;
   unsigned b = 0, s = 0 ;
#pragma unroll
   for (int t = 0 ; t < 10 ; t++) {
      unsigned eq = (t & 1 ? s0 : ~s0) & (t & 2 ? v0 : ~v0) & (t & 4 ? v1 : ~v1) & (t & 8 ? v2 : ~v2) ;
      if ((born >> t) & 1)
         b |= eq ;
      if ((stays >> t) & 1)
         s |= eq ;
   }
   return (b & ~w) | (s & w) ;
}

__device__ __forceinline__ void qgsum3(unsigned l, unsigned w, unsigned r, unsigned &x0, unsigned &x1) {
   unsigned a = __funnelshift_r(w, l, 1) ;   // left neighbors
   unsigned b = __funnelshift_l(r, w, 1) ;   // right neighbors
   x0 = a ^ w ^ b ;
   x1 = (a & w) | (b & (a ^ w)) ;
}

/*
 *   Epoch e: compute K generations of the listed tiles, one thread block
 *   per tile; p0 is the parity of the current generation.  Warp w computes
 *   the blocks w, w + NWARP, ... of the region (numbered by rows), a row of
 *   cells per lane; NWARP > NB, so a warp has at most one block per row.
 *   The changed blocks of a generation are bits of rb (bit j of rb[][i] for
 *   block (i, j)), kept for three generations so that one barrier per
 *   generation suffices: generation gi reads rb[(gi-1)%3], sets bits of
 *   rb[gi%3] and clears rb[(gi+1)%3].
 */
__global__ void __launch_bounds__(BS, 1536 / BS) qgstep(qlg g, int e, int K, int p0,
                                                unsigned born0, unsigned born1,
                                                unsigned stays0, unsigned stays1, int life) {
   // cells of the region by generation parity: word c of row r at (r+1)*RP+1+c
   __shared__ unsigned sb[2][SBW] ;
   __shared__ unsigned rb[3][NB] ;
   __shared__ int nbs[9], nset[9], outset ;
   __shared__ u64 nmask[9], tmask ;
   if (qgstopped(g, e))
      return ;
   int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5 ;
   int n = g.ctl->count[e & 1] ;
   // blocks with tiles to compute (block 0 also when there are none)
   int nact = n < (int)gridDim.x ? (n > 0 ? n : 1) : gridDim.x ;
   if ((int)blockIdx.x >= nact)
      return ;
   const int *ln = g.list[e & 1] ;
   // the zero words and rows, never written
   for (int i = tid ; i < 2 * SBW ; i += BS)
      sb[0][i] = 0 ;
   for (int it = blockIdx.x ; it < n ; it += gridDim.x) {
      int x = ln[it] ;
      if (tid < 9) {
         int s = x ;
         if (tid != 4) {
            u64 key = g.pos[x] ;
            s = qgfind(g, qgkey(qgkx(key) + tid % 3 - 1, qgky(key) + tid / 3 - 1)) ;
         }
         nbs[tid] = s ;
         nmask[tid] = 0 ;
         nset[tid] = 0 ;
         if (s >= 0) {
            // the set holding the cells at the start of the epoch (a
            // neighbor computed in this epoch may have marked its new set)
            qlmeta m = g.meta[s] ;
            int set = (m.st & 1) ^ ((m.st >> 1) == e) ;
            nset[tid] = set ;
            nmask[tid] = qgmask(m, e - 1) ;
            if (tid == 4) {
               outset = set ^ 1 ;
               g.meta[x].st = (e << 1) | (set ^ 1) ;
            }
         }
      }
      __syncthreads() ;
      // the region: the tile and HALO cells around it, generations t-1 and t
      unsigned v[NLD] ;
#pragma unroll
      for (int i = 0 ; i < NLD ; i++) {
         int w = tid + i * BS ;
         v[i] = 0 ;
         if (w < 2 * RS * RW) {
            int q = w >= RS * RW ;
            int rem = w - q * RS * RW ;
            int r = rem / RW, c = rem - r * RW ;
            int k = (r < HALO ? 0 : r < HALO + TS ? 3 : 6) + (c == 0 ? 0 : c == RW - 1 ? 2 : 1) ;
            int s = nbs[k] ;
            if (s >= 0)
               v[i] = g.data[qgoff(s, nset[k], q) + ((r - HALO) & (TS - 1)) * TW + ((c - 1) & (TW - 1))] ;
         }
      }
#pragma unroll
      for (int i = 0 ; i < NLD ; i++) {
         int w = tid + i * BS ;
         if (w < 2 * RS * RW) {
            int q = w >= RS * RW ;
            int rem = w - q * RS * RW ;
            int r = rem / RW, c = rem - r * RW ;
            sb[q][(r + 1) * RP + 1 + c] = v[i] ;
         }
      }
      if (tid < NB) {
         unsigned bits = 0 ;
         for (int bj = 0 ; bj < NB ; bj++) {
            int k = (tid == 0 ? 0 : tid == NB - 1 ? 6 : 3) + (bj == 0 ? 0 : bj == NB - 1 ? 2 : 1) ;
            bits |= (unsigned)((nmask[k] >> (((tid - 1) & (TB - 1)) * TB + ((bj - 1) & (TB - 1)))) & 1) << bj ;
         }
         rb[0][tid] = bits ;
         rb[1][tid] = 0 ;
      }
      __syncthreads() ;
      for (int gi = 1 ; gi <= K ; gi++) {
         int pc = (p0 + gi) & 1, pp = pc ^ 1 ;
         const unsigned *rp = rb[(gi + 2) % 3] ;
         unsigned *rc = rb[gi % 3] ;
         if (warp == 0 && lane < NB)
            rb[(gi + 1) % 3][lane] = 0 ;
         const unsigned *src = sb[pp] ;
         unsigned *dst = sb[pc] ;
         unsigned born = pp ? born1 : born0, stays = pp ? stays1 : stays0 ;
         // a block is computed if it or a block next to it changed in the
         // last generation; lane i finds whether this warp's block in row i
         // is (the warp of block (i, j) is (i*NB + j) % NWARP, at most one
         // per row)
         unsigned act = 0 ;
         if (lane < NB) {
            unsigned m = rp[lane] | (lane > 0 ? rp[lane - 1] : 0) | (lane < NB - 1 ? rp[lane + 1] : 0) ;
            m |= (m << 1) | (m >> 1) ;
            int j = (warp + NWARP * NB - NB * lane) & (NWARP - 1) ;
            act = j < NB && ((m >> j) & 1) ;
         }
         unsigned rows = __ballot_sync(~0u, act) ;
         while (rows) {
            int bi = __ffs(rows) - 1 ;
            rows &= rows - 1 ;
            int c = (warp + NWARP * NB - NB * bi) & (NWARP - 1) ;
            int o = (bi * 32 + lane + 1) * RP + 1 + c ;
            unsigned u0, u1, a0, a1, d0, d1 ;
            unsigned w = src[o] ;
            qgsum3(src[o - RP - 1], src[o - RP], src[o - RP + 1], u0, u1) ;
            qgsum3(src[o - 1], w, src[o + 1], a0, a1) ;
            qgsum3(src[o + RP - 1], src[o + RP], src[o + RP + 1], d0, d1) ;
            unsigned nw = qgrule(u0, u1, a0, a1, d0, d1, w, born, stays, life) ;
            unsigned old = dst[o] ;
            dst[o] = nw ;
            if (__any_sync(~0u, nw != old) && lane == 0)
               atomicOr(&rc[bi], 1u << c) ;
         }
         __syncthreads() ;
      }
      // write both generations of the tile's own cells to its other set
      unsigned acc = 0 ;
      for (int w = tid ; w < 2 * TWORDS ; w += BS) {
         int q = w >= TWORDS ;
         int rem = w - q * TWORDS ;
         unsigned c = sb[q][(rem / TW + HALO + 1) * RP + 2 + rem % TW] ;
         g.data[qgoff(x, outset, q) + rem] = c ;
         acc |= c ;
      }
      int nonempty = __syncthreads_or(acc) ;
      if (tid == 0) {
         u64 m = 0 ;
         for (int i = 0 ; i < TB ; i++)
            m |= (u64)((rb[K % 3][i + 1] >> 1) & ((1u << TB) - 1)) << (TB * i) ;
         qlmeta *mt = &g.meta[x] ;
         mt->mask[e & 1] = m ;
         mt->mstamp[e & 1] = e ;
         mt->empty = !nonempty ;
         tmask = m ;
      }
      __syncthreads() ;
      // list this tile for the next epoch if it changed, and the neighbors
      // next to its changed edge blocks
      if (tid < 9) {
         u64 m = tmask ;
         if (tid == 4) {
            if (m)
               qgwant(g, x, e + 1) ;
         } else if (m & qgedge[tid]) {
            u64 key = g.pos[x] ;
            qgwant(g, qginsert(g, qgkey(qgkx(key) + tid % 3 - 1, qgky(key) + tid / 3 - 1), e), e + 1) ;
         }
      }
   }
   // the last block to finish ends the epoch
   __syncthreads() ;
   if (tid == 0) {
      __threadfence() ;
      if (atomicAdd(&g.ctl->done, 1) == nact - 1) {
         volatile qlctl *c = g.ctl ;
         c->done = 0 ;
         int nf = c->nfree, np = c->npop ;
         nf -= np < nf ? np : nf ;
         c->nfree = nf ;
         c->npop = 0 ;
         c->count[e & 1] = 0 ;   // the list of this epoch is used up
         c->epochs++ ;
         // epoch e+1 creates at most 8 tiles per tile it computes
         int need = 8 * c->count[(e + 1) & 1] ;
         if (nf + g.cap - c->top < need || HLOAD * (c->hused + need) > g.hmask + 1)
            c->stopat = e + 1 ;
      }
   }
}

__device__ void qgor(const qlg &g, int k, int y, unsigned v, int e) {
   int s = qginsert(g, qgkey(k >> LOGTW, y >> LOGTS), e) ;
   size_t o = qgoff(s, 0, 0) + (y & (TS - 1)) * TW + (k & (TW - 1)) ;
   atomicOr(&g.data[o], v) ;
   atomicOr(&g.data[o + TWORDS], v) ;
}

// load bricks of a generation of parity odd into set 0, as both generations
__global__ void qgload(qlg g, const qlgbrick *b, int n, int odd, int e) {
   for (int i = blockIdx.x * blockDim.x + threadIdx.x ; i < n ; i += gridDim.x * blockDim.x) {
      qlgbrick k = b[i] ;
      for (int r = 0 ; r < 8 ; r++) {
         unsigned row = 0 ;
         for (int j = 0 ; j < 8 ; j++)
            row |= ((k.d[j] >> (28 - 4 * r)) & 0xf) << (28 - 4 * j) ;
         if (row == 0)
            continue ;
         int y = k.y + r + odd ;
         int w = k.x >> 5 ;
         if (odd) {
            if (row >> 1)
               qgor(g, w, y, row >> 1, e) ;
            if (row & 1)
               qgor(g, w + 1, y, row << 31, e) ;
         } else
            qgor(g, w, y, row, e) ;
      }
   }
}

// after loading the tiles in slots 0 to n-1: they count as changed
// everywhere, and they and their neighbors are listed for epoch e
__global__ void qgstart(qlg g, int n, int e) {
   for (int i = blockIdx.x * blockDim.x + threadIdx.x ; i < 9 * n ; i += gridDim.x * blockDim.x) {
      int s = i / 9, d = i % 9 ;
      if (d == 4) {
         g.meta[s].mask[(e - 1) & 1] = ~0ULL ;
         g.meta[s].mstamp[(e - 1) & 1] = e - 1 ;
         qgwant(g, s, e) ;
      } else {
         u64 key = g.pos[s] ;
         qgwant(g, qginsert(g, qgkey(qgkx(key) + d % 3 - 1, qgky(key) + d / 3 - 1), e), e) ;
      }
   }
}

// new slots: no change bits, both sets empty
__global__ void qginitmeta(qlg g, int from, int to) {
   for (int s = from + blockIdx.x * blockDim.x + threadIdx.x ; s < to ; s += gridDim.x * blockDim.x) {
      qlmeta m ;
      m.mask[0] = m.mask[1] = 0 ;
      m.mstamp[0] = m.mstamp[1] = -1 ;
      m.st = 0 ;
      m.empty = 1 ;
      g.meta[s] = m ;
   }
}

// rebuild the hash table from the slots
__global__ void qgrehash(qlg g) {
   int top = g.ctl->top ;
   for (int s = blockIdx.x * blockDim.x + threadIdx.x ; s < top ; s += gridDim.x * blockDim.x) {
      u64 key = g.pos[s] ;
      if (key == KEMPTY)
         continue ;
      unsigned h = qghash(key, g.hbits) ;
      while (atomicCAS(&g.hash[h].key, KEMPTY, key) != KEMPTY)
         h = (h + 1) & g.hmask ;
      g.hash[h].val = s ;
      atomicAdd(&g.ctl->hused, 1) ;
   }
}

__device__ inline unsigned qgword(const qlg &g, const int *nbs, const int *nset,
                                  int tx, int ty, int k, int y, int par) {
   int i = ((y >> LOGTS) - ty + 1) * 3 + (k >> LOGTW) - tx + 1 ;
   int s = nbs[i] ;
   if (s < 0)
      return 0 ;
   return g.data[qgoff(s, nset[i], par) + (y & (TS - 1)) * TW + (k & (TW - 1))] ;
}

/*
 *   The non-empty bricks of a generation of parity odd; one thread block
 *   per slot.  Odd bricks lie across tiles: a brick at tree coordinates
 *   (x, y) holds the cells (x+1, y+1) to (x+32, y+8) here.  The tile
 *   holding (x, y) writes it, or if that tile is missing the first
 *   existing one of the tiles to its right, below, and below right that
 *   hold some of its cells.
 */
__global__ void qgsave(qlg g, int odd, qlgbrick *out) {
   __shared__ int nbs[9], nset[9] ;
   int x = blockIdx.x ;
   u64 key = g.pos[x] ;
   if (key == KEMPTY)
      return ;
   int tx = qgkx(key), ty = qgky(key) ;
   if (threadIdx.x < 9) {
      int s = threadIdx.x == 4 ? x : qgfind(g, qgkey(tx + threadIdx.x % 3 - 1, ty + threadIdx.x / 3 - 1)) ;
      nbs[threadIdx.x] = s ;
      nset[threadIdx.x] = s >= 0 ? g.meta[s].st & 1 : 0 ;
   }
   __syncthreads() ;
   for (int c = threadIdx.x ; c < SAVEN ; c += blockDim.x) {
      int x0, y0 ;
      if (c < SAVEOWN) {
         x0 = tx * TS + 32 * (c & (TW - 1)) ;
         y0 = ty * TS + 8 * (c >> LOGTW) ;
      } else {
         if (!odd)
            continue ;
         if (c < SAVEOWN + TS / 8) {   // the tile on the left is missing
            if (nbs[3] >= 0)
               continue ;
            x0 = tx * TS - 32 ;
            y0 = ty * TS + 8 * (c - SAVEOWN) ;
         } else if (c < SAVEN - 1) {   // the tile above is missing
            int a = c - SAVEOWN - TS / 8 ;
            if (nbs[1] >= 0 || (a == TW - 1 && nbs[2] >= 0))
               continue ;
            x0 = tx * TS + 32 * a ;
            y0 = ty * TS - 8 ;
         } else {                // the tiles above left, above and left are missing
            if (nbs[0] >= 0 || nbs[1] >= 0 || nbs[3] >= 0)
               continue ;
            x0 = tx * TS - 32 ;
            y0 = ty * TS - 8 ;
         }
      }
      unsigned rows[8], any = 0 ;
      for (int r = 0 ; r < 8 ; r++) {
         int y = y0 + r + odd, k = x0 >> 5 ;
         unsigned v = qgword(g, nbs, nset, tx, ty, k, y, odd) ;
         if (odd)
            v = (v << 1) | (qgword(g, nbs, nset, tx, ty, k + 1, y, odd) >> 31) ;
         rows[r] = v ;
         any |= v ;
      }
      if (!any)
         continue ;
      qlgbrick o ;
      o.x = x0 ;
      o.y = y0 ;
      for (int j = 0 ; j < 8 ; j++) {
         unsigned d = 0 ;
         for (int r = 0 ; r < 8 ; r++)
            d |= ((rows[r] >> (28 - 4 * j)) & 0xf) << (28 - 4 * r) ;
         o.d[j] = d ;
      }
      out[atomicAdd(&g.ctl->nout, 1)] = o ;
   }
}


/*
 *   Drawing, population and bounding box.  A cell at Golly's (x, y) is at
 *   (x, -y) here.  One thread block per slot.
 */
__device__ inline const unsigned *qgcur(const qlg &g, int x, int par) {
   return g.data + qgoff(x, g.meta[x].st & 1, par) ;
}

__global__ void qgdraw(qlg g, int par, long long cx, long long cy, int ls, int w, int h,
                       int ow, unsigned *out) {
   int top = g.ctl->top ;
   for (int x = blockIdx.x ; x < top ; x += gridDim.x) {
      u64 key = g.pos[x] ;
      if (key == KEMPTY)
         continue ;
      long long X0 = (long long)qgkx(key) * TS, Y0 = (long long)qgky(key) * TS ;
      // elements the tile covers
      if (((X0 + TS - 1 - cx) >> ls) < 0 || ((X0 - cx) >> ls) >= w ||
          ((-Y0 - cy) >> ls) < 0 || ((-(Y0 + TS - 1) - cy) >> ls) >= h)
         continue ;
      const unsigned *d = qgcur(g, x, par) ;
      for (int i = threadIdx.x ; i < TWORDS ; i += blockDim.x) {
         unsigned v = d[i] ;
         if (v == 0)
            continue ;
         long long X = X0 + 32 * (i & (TW - 1)), Y = Y0 + (i >> LOGTW) ;
         long long py = (-Y - cy) >> ls ;
         if (py < 0 || py >= h)
            continue ;
         unsigned *row = out + py * ow ;
         if (ls == 0) {
            // cell X + j is element X - cx + j, both most significant bit first
            long long px = X - cx ;
            if (px <= -32 || px >= w)
               continue ;
            if (px < 0) {
               atomicOr(&row[0], v << (-px)) ;
            } else {
               int o = px & 31, k = (int)(px >> 5) ;
               atomicOr(&row[k], v >> o) ;
               if (o && k + 1 < ow)
                  atomicOr(&row[k + 1], v << (32 - o)) ;
            }
         } else {
            long long s = 1LL << ls ;
            for (long long p = (X - cx) >> ls ; p <= (X + 31 - cx) >> ls ; p++) {
               if (p < 0 || p >= w)
                  continue ;
               // the cells of element p in this word: bits of offsets a to b-1
               long long c0 = cx + p * s - X, c1 = c0 + s ;
               int a = c0 < 0 ? 0 : (int)c0, b = c1 > 32 ? 32 : (int)c1 ;
               unsigned m = b - a == 32 ? ~0u : ((1u << (b - a)) - 1) << (32 - b) ;
               if (v & m)
                  atomicOr(&row[p >> 5], 1u << (31 - (p & 31))) ;
            }
         }
      }
   }
}

__global__ void qgpop(qlg g, int par, u64 *pop) {
   int top = g.ctl->top ;
   u64 n = 0 ;
   for (int x = blockIdx.x ; x < top ; x += gridDim.x) {
      if (g.pos[x] == KEMPTY)
         continue ;
      const unsigned *d = qgcur(g, x, par) ;
      for (int i = threadIdx.x ; i < TWORDS ; i += blockDim.x)
         n += __popc(d[i]) ;
   }
   atomicAdd(pop, n) ;
}

// e[0..3]: smallest x, smallest -x, smallest y, smallest -y of live cells (Golly's coordinates)
__global__ void qgedges(qlg g, int par, long long *e) {
   int top = g.ctl->top ;
   long long m[4] = { LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX } ;
   for (int x = blockIdx.x ; x < top ; x += gridDim.x) {
      u64 key = g.pos[x] ;
      if (key == KEMPTY)
         continue ;
      long long X0 = (long long)qgkx(key) * TS, Y0 = (long long)qgky(key) * TS ;
      const unsigned *d = qgcur(g, x, par) ;
      for (int i = threadIdx.x ; i < TWORDS ; i += blockDim.x) {
         unsigned v = d[i] ;
         if (v == 0)
            continue ;
         long long X = X0 + 32 * (i & (TW - 1)), y = -(Y0 + (i >> LOGTW)) ;
         m[0] = min(m[0], X + __clz(v)) ;
         m[1] = min(m[1], -(X + 31 - (__ffs(v) - 1))) ;
         m[2] = min(m[2], y) ;
         m[3] = min(m[3], -y) ;
      }
   }
   for (int k = 0 ; k < 4 ; k++)
      atomicMin(&e[k], m[k]) ;
}

/* ---------------------------------------------------------------- host */

template<class T> static void qgmalloc(T *&p, size_t n) {
   CK(cudaMalloc((void **)&p, n * sizeof(T))) ;
}

// allocate the tables for cap slots, copying the slots of the old tables (if any)
static void qgslots(qlgpu *q, int cap) {
   qlg o = q->g, &g = q->g ;
   int ocap = o.data ? o.cap : 0 ;
   g.cap = cap ;
   cudaStream_t s = q->stream ;
   qgmalloc(g.data, (size_t)cap * 4 * TWORDS) ;
   qgmalloc(g.pos, cap) ;
   qgmalloc(g.meta, cap) ;
   qgmalloc(g.want, cap) ;
   qgmalloc(g.list[0], cap) ;
   qgmalloc(g.list[1], cap) ;
   qgmalloc(g.freestack, cap) ;
   qgmalloc(g.freed, cap) ;
   size_t nn = cap - ocap ;
   CK(cudaMemsetAsync(g.data + (size_t)ocap * 4 * TWORDS, 0, nn * 4 * TWORDS * sizeof(unsigned), s)) ;
   CK(cudaMemsetAsync(g.pos + ocap, 0xff, nn * sizeof(u64), s)) ;
   CK(cudaMemsetAsync(g.want + ocap, 0xff, nn * sizeof(int), s)) ;
   qginitmeta<<<q->nsm * 4, 256, 0, s>>>(g, ocap, cap) ;
   CK(cudaGetLastError()) ;
   if (ocap) {
      CK(cudaMemcpyAsync(g.data, o.data, (size_t)ocap * 4 * TWORDS * sizeof(unsigned), cudaMemcpyDeviceToDevice, s)) ;
      CK(cudaMemcpyAsync(g.pos, o.pos, ocap * sizeof(u64), cudaMemcpyDeviceToDevice, s)) ;
      CK(cudaMemcpyAsync(g.meta, o.meta, ocap * sizeof(qlmeta), cudaMemcpyDeviceToDevice, s)) ;
      CK(cudaMemcpyAsync(g.want, o.want, ocap * sizeof(int), cudaMemcpyDeviceToDevice, s)) ;
      CK(cudaMemcpyAsync(g.list[0], o.list[0], ocap * sizeof(int), cudaMemcpyDeviceToDevice, s)) ;
      CK(cudaMemcpyAsync(g.list[1], o.list[1], ocap * sizeof(int), cudaMemcpyDeviceToDevice, s)) ;
      CK(cudaMemcpyAsync(g.freestack, o.freestack, ocap * sizeof(int), cudaMemcpyDeviceToDevice, s)) ;
      CK(cudaStreamSynchronize(s)) ;
      cudaFree(o.data) ; cudaFree(o.pos) ; cudaFree(o.meta) ; cudaFree(o.want) ;
      cudaFree(o.list[0]) ; cudaFree(o.list[1]) ; cudaFree(o.freestack) ; cudaFree(o.freed) ;
   }
}

// rebuild the hash table with 2^hbits entries (dropping removed ones)
static void qghashtab(qlgpu *q, int hbits) {
   qlg &g = q->g ;
   cudaStream_t s = q->stream ;
   if (g.hash == 0 || hbits != g.hbits) {
      if (g.hash) {
         CK(cudaStreamSynchronize(s)) ;
         cudaFree(g.hash) ;
      }
      g.hbits = hbits ;
      g.hmask = (1 << hbits) - 1 ;
      qgmalloc(g.hash, (size_t)1 << hbits) ;
   }
   CK(cudaMemsetAsync(g.hash, 0xff, ((size_t)1 << hbits) * sizeof(qlhent), s)) ;
   CK(cudaMemsetAsync(&g.ctl->hused, 0, sizeof(int), s)) ;
   qgrehash<<<q->nsm * 4, 256, 0, s>>>(g) ;
   CK(cudaGetLastError()) ;
}

static int qgbits(size_t n) {
   int b = 10 ;
   while (((size_t)1 << b) < n)
      b++ ;
   return b ;
}

static void qgfreeall(qlgpu *q) {
   qlg &g = q->g ;
   if (g.data) {
      cudaFree(g.data) ; cudaFree(g.pos) ; cudaFree(g.meta) ; cudaFree(g.want) ;
      cudaFree(g.list[0]) ; cudaFree(g.list[1]) ; cudaFree(g.freestack) ; cudaFree(g.freed) ;
      g.data = 0 ;
   }
   if (g.hash) {
      cudaFree(g.hash) ;
      g.hash = 0 ;
   }
}

// starting the CUDA driver and context takes 0.1 s or more; qlgpu_warmup
// does it in another thread
static std::once_flag qgwarmonce ;
static std::future<void> qgwarmed ;
void qlgpu_warmup() {
   std::call_once(qgwarmonce, [] {
      qgwarmed = std::async(std::launch::async, [] {
         int n = 0 ;
         if (cudaGetDeviceCount(&n) == cudaSuccess && n > 0)
            cudaFree(0) ;
      }) ;
   }) ;
}

qlgpu *qlgpu_new() {
   if (qgwarmed.valid())
      qgwarmed.wait() ;
   int n = 0 ;
   if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
      return 0 ;
   qlgpu *q = new qlgpu ;
   memset(q, 0, sizeof(*q)) ;
   CK(cudaStreamCreateWithFlags(&q->stream, cudaStreamNonBlocking)) ;
   CK(cudaDeviceGetAttribute(&q->nsm, cudaDevAttrMultiProcessorCount, 0)) ;
   CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&q->perstep, qgstep, BS, 0)) ;
   CK(cudaMalloc((void **)&q->g.ctl, sizeof(qlctl))) ;
   CK(cudaMallocHost((void **)&q->hctl, sizeof(qlctl))) ;
   return q ;
}

void qlgpu_delete(qlgpu *q) {
   qgfreeall(q) ;
   cudaFree(q->g.ctl) ;
   cudaFreeHost(q->hctl) ;
   cudaStreamDestroy(q->stream) ;
   delete q ;
}

void qlgpu_load(qlgpu *q, const std::vector<qlgbrick> &b, int odd) {
   // the tiles the bricks touch
   std::unordered_set<u64> keys ;
   for (size_t i = 0 ; i < b.size() ; i++)
      for (int dy = 0 ; dy <= odd ; dy++)
         for (int dx = 0 ; dx <= odd ; dx++)
            keys.insert(qgkey((b[i].x + 31 * dx + odd) >> LOGTS, (b[i].y + 7 * dy + odd) >> LOGTS)) ;
   // room for the tiles and those the first epoch may create
   int cap = 1024 ;
   while ((size_t)cap < 9 * keys.size() + 1024)
      cap *= 2 ;
   qgfreeall(q) ;
   qlg &g = q->g ;
   CK(cudaMemsetAsync(g.ctl, 0, sizeof(qlctl), q->stream)) ;
   qgslots(q, cap) ;
   qghashtab(q, qgbits(HLOAD * (size_t)cap)) ;
   q->epoch = 1 ;
   if (!b.empty()) {
      qlgbrick *db ;
      CK(cudaMalloc((void **)&db, b.size() * sizeof(qlgbrick))) ;
      CK(cudaMemcpyAsync(db, b.data(), b.size() * sizeof(qlgbrick), cudaMemcpyHostToDevice, q->stream)) ;
      qgload<<<q->nsm * 4, 256, 0, q->stream>>>(g, db, (int)b.size(), odd, q->epoch) ;
      CK(cudaGetLastError()) ;
      CK(cudaStreamSynchronize(q->stream)) ;
      cudaFree(db) ;
   }
   CK(cudaMemcpyAsync(q->hctl, g.ctl, sizeof(qlctl), cudaMemcpyDeviceToHost, q->stream)) ;
   CK(cudaStreamSynchronize(q->stream)) ;
   qgstart<<<q->nsm * 4, 256, 0, q->stream>>>(g, q->hctl->top, q->epoch) ;
   q->lastcount = 9 * q->hctl->top ;
   CK(cudaGetLastError()) ;
}

void qlgpu_run(qlgpu *q, int gens, int odd, const unsigned int born[2], const unsigned int stays[2]) {
   qlg &g = q->g ;
   int life = 1 ;
   for (int p = 0 ; p < 2 ; p++)
      if (born[p] != (1u << 3) || stays[p] != ((1u << 3) | (1u << 4)))
         life = 0 ;
   while (gens > 0) {
      int ks[EPOCHS], n = 0, p = odd, left = gens ;
      int e0 = q->epoch - 1 ;   // epochs computed
      int nstep = (int)(GROW * q->lastcount) + 8 ;
      if (nstep > q->nsm * q->perstep)
         nstep = q->nsm * q->perstep ;
      for (n = 0 ; n < EPOCHS && left > 0 ; n++) {
         int k = left < HALO ? left : HALO ;
         int e = q->epoch + n ;
         if (e % SWEEP == 0)
            qgsweep<<<q->nsm, 256, 0, q->stream>>>(g, e) ;
         qgstep<<<nstep, BS, 0, q->stream>>>(g, e, k, p, born[0], born[1], stays[0], stays[1], life) ;
         ks[n] = k ;
         left -= k ;
         p ^= k & 1 ;
      }
      CK(cudaGetLastError()) ;
      CK(cudaMemcpyAsync(q->hctl, g.ctl, sizeof(qlctl), cudaMemcpyDeviceToHost, q->stream)) ;
      CK(cudaStreamSynchronize(q->stream)) ;
      int done = q->hctl->epochs - e0 ;
      for (int i = 0 ; i < done ; i++) {
         gens -= ks[i] ;
         odd ^= ks[i] & 1 ;
      }
      q->epoch += done ;
      q->lastcount = q->hctl->count[q->epoch & 1] ;
      if (q->hctl->stopat) {
         // room for the tiles the next epoch may create: enlarge the slots
         // if needed, and rebuild the hash table without removed entries
         int need = 8 * q->hctl->count[q->epoch & 1] ;
         int live = q->hctl->top - q->hctl->nfree ;
         int cap = g.cap ;
         while (cap - live < need + 1024)
            cap *= 2 ;
         if (cap != g.cap)
            qgslots(q, cap) ;
         int hbits = g.hbits ;
         while (((size_t)1 << hbits) < HLOAD * ((size_t)live + need))
            hbits++ ;
         qghashtab(q, hbits) ;
         CK(cudaMemsetAsync(&g.ctl->stopat, 0, sizeof(int), q->stream)) ;
      }
   }
}

void qlgpu_save(qlgpu *q, std::vector<qlgbrick> &b, int odd) {
   qlg &g = q->g ;
   CK(cudaMemcpyAsync(q->hctl, g.ctl, sizeof(qlctl), cudaMemcpyDeviceToHost, q->stream)) ;
   CK(cudaStreamSynchronize(q->stream)) ;
   int top = q->hctl->top ;
   b.clear() ;
   if (top == 0)
      return ;
   qlgbrick *db ;
   CK(cudaMalloc((void **)&db, (size_t)top * SAVEN * sizeof(qlgbrick))) ;
   CK(cudaMemsetAsync(&g.ctl->nout, 0, sizeof(int), q->stream)) ;
   qgsave<<<top, 128, 0, q->stream>>>(g, odd, db) ;
   CK(cudaGetLastError()) ;
   CK(cudaMemcpyAsync(q->hctl, g.ctl, sizeof(qlctl), cudaMemcpyDeviceToHost, q->stream)) ;
   CK(cudaStreamSynchronize(q->stream)) ;
   b.resize(q->hctl->nout) ;
   CK(cudaMemcpy(b.data(), db, b.size() * sizeof(qlgbrick), cudaMemcpyDeviceToHost)) ;
   cudaFree(db) ;
}

void qlgpu_draw(qlgpu *q, int odd, long long cx, long long cy, int ls, int w, int h,
                std::vector<unsigned int> &bits) {
   int ow = (w + 31) / 32 ;
   size_t n = (size_t)ow * h ;
   unsigned *d ;
   CK(cudaMalloc((void **)&d, n * sizeof(unsigned))) ;
   CK(cudaMemsetAsync(d, 0, n * sizeof(unsigned), q->stream)) ;
   // y = cy + py * 2^ls is at -y here
   qgdraw<<<q->nsm * 8, 256, 0, q->stream>>>(q->g, odd, cx, cy, ls, w, h, ow, d) ;
   CK(cudaGetLastError()) ;
   bits.resize(n) ;
   CK(cudaMemcpyAsync(bits.data(), d, n * sizeof(unsigned), cudaMemcpyDeviceToHost, q->stream)) ;
   CK(cudaStreamSynchronize(q->stream)) ;
   cudaFree(d) ;
}

long long qlgpu_population(qlgpu *q, int odd) {
   u64 *d, n = 0 ;
   CK(cudaMalloc((void **)&d, sizeof(u64))) ;
   CK(cudaMemsetAsync(d, 0, sizeof(u64), q->stream)) ;
   qgpop<<<q->nsm * 8, 256, 0, q->stream>>>(q->g, odd, d) ;
   CK(cudaGetLastError()) ;
   CK(cudaMemcpyAsync(&n, d, sizeof(u64), cudaMemcpyDeviceToHost, q->stream)) ;
   CK(cudaStreamSynchronize(q->stream)) ;
   cudaFree(d) ;
   return (long long)n ;
}

int qlgpu_edges(qlgpu *q, int odd, long long &left, long long &top,
                long long &right, long long &bottom) {
   long long *d, e[4] = { LLONG_MAX, LLONG_MAX, LLONG_MAX, LLONG_MAX } ;
   CK(cudaMalloc((void **)&d, sizeof(e))) ;
   CK(cudaMemcpyAsync(d, e, sizeof(e), cudaMemcpyHostToDevice, q->stream)) ;
   qgedges<<<q->nsm * 8, 256, 0, q->stream>>>(q->g, odd, d) ;
   CK(cudaGetLastError()) ;
   CK(cudaMemcpyAsync(e, d, sizeof(e), cudaMemcpyDeviceToHost, q->stream)) ;
   CK(cudaStreamSynchronize(q->stream)) ;
   cudaFree(d) ;
   if (e[0] == LLONG_MAX)
      return 0 ;
   left = e[0] ;
   right = -e[1] ;
   top = e[2] ;
   bottom = -e[3] ;
   return 1 ;
}
