// Prototype: run an outer-totalistic Life-like rule on the GPU, to measure
// how a GPU compares with QuickLife.  Not part of Golly.
//
//   gpulife PATTERN.rle GENS
//
// Prints a line in the format of `bgolly --summary`:
//   summary load_s=... run_s=... steps=1 gen=GENS pop=... bbox=left,top,right,bottom
// run_s covers the generations, including growing the grid, and the final
// population and bounding box.
//
// The universe is a dense grid of bits, 64 cells per 64-bit word, row by
// row; bit 63 of a word is its westmost cell.  One GPU thread computes one
// word per generation, from the 3x3 words around it (a bit-sliced adder
// counts the neighbors of 64 cells at once).  Cells outside the grid are
// dead.  Every CHECK generations the bounding box of the live cells is
// found; if it is within MARGIN cells of an edge, the grid is enlarged,
// so that cells outside it would have stayed dead (a pattern grows by at
// most one cell per generation).
//
// The grid is divided into blocks of BW words by BH rows, one GPU thread
// block each.  A block whose cells did not change in the last generation,
// with none of its 8 neighbor blocks changed either, would compute the
// same cells again; it is skipped, and the other buffer already holds
// those cells (they are the same in the last two generations).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
   fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)) ; exit(1) ; } } while (0)

static const int CHECK = 64 ;      // generations between bounding box checks
static const int MARGIN = 192 ;    // empty cells kept around the pattern (> CHECK)

typedef unsigned long long u64 ;

// neighbor count bit planes and rule, for 64 cells
__device__ __forceinline__ u64 step64(u64 ul, u64 u, u64 ur, u64 l, u64 m, u64 r,
                                      u64 dl, u64 d, u64 dr, unsigned bmask, unsigned smask) {
   // the 8 neighbors of each bit: shifted copies (west neighbor = higher bit)
   u64 n0 = (u << 1) | (ur >> 63), n1 = u, n2 = (u >> 1) | (ul << 63) ;
   u64 n3 = (m << 1) | (r >> 63), n4 = (m >> 1) | (l << 63) ;
   u64 n5 = (d << 1) | (dr >> 63), n6 = d, n7 = (d >> 1) | (dl << 63) ;
   // full adders
   u64 a0 = n0 ^ n1 ^ n2, a1 = (n0 & n1) | (n2 & (n0 ^ n1)) ;
   u64 b0 = n5 ^ n6 ^ n7, b1 = (n5 & n6) | (n7 & (n5 ^ n6)) ;
   u64 c0 = n3 ^ n4, c1 = n3 & n4 ;
   u64 s0 = a0 ^ b0 ^ c0, k1 = (a0 & b0) | (c0 & (a0 ^ b0)) ;
   u64 t0 = a1 ^ b1 ^ c1, t1 = (a1 & b1) | (c1 & (a1 ^ b1)) ;
   u64 s1 = t0 ^ k1, q = t0 & k1 ;
   u64 s2 = t1 ^ q, s3 = t1 & q ;
   u64 res = 0 ;
   for (int k=0; k<9; k++) {
      unsigned b = (bmask >> k) & 1, s = (smask >> k) & 1 ;
      if (!(b | s))
         continue ;
      u64 eq = ((k & 1) ? s0 : ~s0) & ((k & 2) ? s1 : ~s1) &
               ((k & 4) ? s2 : ~s2) & ((k & 8) ? s3 : ~s3) ;
      res |= eq & ((b ? ~m : 0) | (s ? m : 0)) ;
   }
   return res ;
}

static const int BW = 32, BH = 8 ;

// chg[block]: the block changed in the generation that made in
__global__ void gen(const u64 *in, u64 *out, int w, int h, unsigned bmask, unsigned smask,
                    const unsigned char *chg, unsigned char *nchg) {
   int bx = blockIdx.x, by = blockIdx.y, nbx = gridDim.x, nby = gridDim.y ;
   int any = 0 ;
   for (int j=-1; j<=1; j++)
      for (int i=-1; i<=1; i++) {
         int xx = bx + i, yy = by + j ;
         if (xx >= 0 && xx < nbx && yy >= 0 && yy < nby && chg[yy * nbx + xx])
            any = 1 ;
      }
   if (!any) {
      if (threadIdx.x == 0 && threadIdx.y == 0)
         nchg[by * nbx + bx] = 0 ;
      return ;
   }
   int x = bx * BW + threadIdx.x ;
   int y = by * BH + threadIdx.y ;
   int changed = 0 ;
   if (x < w && y < h) {
      auto at = [&](int xx, int yy) -> u64 {
         return (xx < 0 || xx >= w || yy < 0 || yy >= h) ? 0 : in[(size_t)yy * w + xx] ;
      } ;
      u64 m = at(x, y) ;
      u64 v = step64(at(x-1, y-1), at(x, y-1), at(x+1, y-1),
                     at(x-1, y), m, at(x+1, y),
                     at(x-1, y+1), at(x, y+1), at(x+1, y+1), bmask, smask) ;
      out[(size_t)y * w + x] = v ;
      changed = v != m ;
   }
   changed = __syncthreads_or(changed) ;
   if (threadIdx.x == 0 && threadIdx.y == 0)
      nchg[by * nbx + bx] = (unsigned char)(changed != 0) ;
}

// per row: OR of the row's words into rowor[y]; per word column: OR into color[x]
__global__ void extent(const u64 *g, int w, int h, unsigned *rowany, u64 *color) {
   int x = blockIdx.x * blockDim.x + threadIdx.x ;
   int y = blockIdx.y * blockDim.y + threadIdx.y ;
   if (x >= w || y >= h)
      return ;
   u64 v = g[(size_t)y * w + x] ;
   if (v) {
      rowany[y] = 1 ;
      atomicOr(&color[x], v) ;
   }
}

__global__ void popcount(const u64 *g, size_t n, unsigned long long *pop) {
   unsigned long long c = 0 ;
   for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x ; i < n ; i += (size_t)gridDim.x * blockDim.x)
      c += __popcll(g[i]) ;
   atomicAdd(pop, c) ;
}

struct grid {
   int w, h ;            // words per row, rows
   long long x0, y0 ;    // universe coordinates of the westmost cell of row 0
   u64 *a, *b ;
   unsigned char *ca, *cb ;   // changed flags of the blocks (for a, for the next)
   int nbx, nby ;
} ;

// a new grid: every block marked changed, so all are computed once
static void galloc(grid &g) {
   size_t n = (size_t)g.w * g.h ;
   CK(cudaMalloc(&g.a, n * 8)) ;
   CK(cudaMalloc(&g.b, n * 8)) ;
   CK(cudaMemset(g.a, 0, n * 8)) ;
   CK(cudaMemset(g.b, 0, n * 8)) ;
   g.nbx = (g.w + BW - 1) / BW ;
   g.nby = (g.h + BH - 1) / BH ;
   CK(cudaMalloc(&g.ca, (size_t)g.nbx * g.nby)) ;
   CK(cudaMalloc(&g.cb, (size_t)g.nbx * g.nby)) ;
   CK(cudaMemset(g.ca, 1, (size_t)g.nbx * g.nby)) ;
   CK(cudaMemset(g.cb, 1, (size_t)g.nbx * g.nby)) ;
}
static void gfree(grid &g) {
   cudaFree(g.a) ;
   cudaFree(g.b) ;
   cudaFree(g.ca) ;
   cudaFree(g.cb) ;
}

// bounding box of the live cells in grid coordinates; false if empty
static bool bbox(const grid &g, long long &l, long long &t, long long &r, long long &bt) {
   unsigned *rowany ;
   u64 *color ;
   CK(cudaMalloc(&rowany, g.h * sizeof(unsigned))) ;
   CK(cudaMalloc(&color, g.w * sizeof(u64))) ;
   CK(cudaMemset(rowany, 0, g.h * sizeof(unsigned))) ;
   CK(cudaMemset(color, 0, g.w * sizeof(u64))) ;
   dim3 bs(32, 8), gs((g.w + 31) / 32, (g.h + 7) / 8) ;
   extent<<<gs, bs>>>(g.a, g.w, g.h, rowany, color) ;
   std::vector<unsigned> ra(g.h) ;
   std::vector<u64> co(g.w) ;
   CK(cudaMemcpy(ra.data(), rowany, g.h * sizeof(unsigned), cudaMemcpyDeviceToHost)) ;
   CK(cudaMemcpy(co.data(), color, g.w * sizeof(u64), cudaMemcpyDeviceToHost)) ;
   cudaFree(rowany) ;
   cudaFree(color) ;
   int y1 = -1, y2 = -1, x1 = -1, x2 = -1 ;
   for (int y=0; y<g.h; y++)
      if (ra[y]) {
         if (y1 < 0)
            y1 = y ;
         y2 = y ;
      }
   for (int x=0; x<g.w; x++)
      if (co[x]) {
         if (x1 < 0)
            x1 = x ;
         x2 = x ;
      }
   if (y1 < 0)
      return false ;
   l = (long long)x1 * 64 + __builtin_clzll(co[x1]) ;
   r = (long long)x2 * 64 + 63 - __builtin_ctzll(co[x2]) ;
   t = y1 ;
   bt = y2 ;
   return true ;
}

// copy g into a larger grid so that the live cells are at least
// 2 * MARGIN cells from each edge
static void grow(grid &g, long long l, long long t, long long r, long long bt) {
   long long nl = std::min(l - 2 * MARGIN, 0LL), nt = std::min(t - 2 * MARGIN, 0LL) ;
   long long nr = std::max(r + 2 * MARGIN, (long long)g.w * 64 - 1) ;
   long long nb = std::max(bt + 2 * MARGIN, (long long)g.h - 1) ;
   int dx = (int)((-nl + 63) / 64) ;          // words added on the west
   int dy = (int)(-nt) ;                       // rows added on the north
   grid n ;
   n.w = (int)((nr + 64 + 63) / 64) + dx ;
   n.h = (int)(nb + 1) + dy ;
   n.x0 = g.x0 - (long long)dx * 64 ;
   n.y0 = g.y0 - dy ;
   galloc(n) ;
   CK(cudaMemcpy2D(n.a + (size_t)dy * n.w + dx, n.w * 8, g.a, g.w * 8, g.w * 8, g.h,
                   cudaMemcpyDeviceToDevice)) ;
   gfree(g) ;
   g = n ;
}

// minimal RLE reader: header "x = .., y = .., rule = B../S.."
static bool readrle(const char *path, std::vector<std::pair<long long, long long> > &cells,
                    unsigned &bmask, unsigned &smask) {
   std::ifstream f(path) ;
   if (!f)
      return false ;
   std::string line, body ;
   bmask = 1 << 3 ;
   smask = (1 << 2) | (1 << 3) ;
   bool header = false ;
   while (std::getline(f, line)) {
      line.erase(std::remove(line.begin(), line.end(), '\r'), line.end()) ;
      if (line.empty() || line[0] == '#')
         continue ;
      if (!header && line[0] == 'x') {
         header = true ;
         size_t p = line.find("rule") ;
         if (p != std::string::npos) {
            std::string rule = line.substr(line.find('=', p) + 1) ;
            rule.erase(std::remove(rule.begin(), rule.end(), ' '), rule.end()) ;
            if (rule.find(':') != std::string::npos)
               return false ;   // bounded grids are not handled
            size_t sl = rule.find('/') ;
            std::string a = rule.substr(0, sl), b = sl == std::string::npos ? "" : rule.substr(sl + 1) ;
            if (!a.empty() && (a[0] == 'B' || a[0] == 'b')) {
               bmask = smask = 0 ;
               for (char c : a.substr(1)) {
                  if (c < '0' || c > '8')
                     return false ;
                  bmask |= 1 << (c - '0') ;
               }
               for (char c : b.substr(1)) {
                  if (c < '0' || c > '8')
                     return false ;
                  smask |= 1 << (c - '0') ;
               }
            } else
               return false ;
         }
         continue ;
      }
      body += line ;
   }
   long long x = 0, y = 0, n = 0 ;
   for (char c : body) {
      if (c >= '0' && c <= '9') {
         n = n * 10 + (c - '0') ;
         continue ;
      }
      long long k = n ? n : 1 ;
      n = 0 ;
      if (c == 'b' || c == '.')
         x += k ;
      else if (c == 'o' || c == 'A') {
         for (long long i=0; i<k; i++)
            cells.push_back(std::make_pair(x + i, y)) ;
         x += k ;
      } else if (c == '$') {
         y += k ;
         x = 0 ;
      } else if (c == '!')
         break ;
      else if (c > ' ')
         return false ;   // multistate or unknown
   }
   return true ;
}

int main(int argc, char **argv) {
   if (argc < 3) {
      fprintf(stderr, "usage: gpulife PATTERN.rle GENS\n") ;
      return 2 ;
   }
   auto tl = std::chrono::steady_clock::now() ;
   std::vector<std::pair<long long, long long> > cells ;
   unsigned bmask, smask ;
   if (!readrle(argv[1], cells, bmask, smask)) {
      fprintf(stderr, "cannot read %s (only two-state B/S rules on an unbounded grid)\n", argv[1]) ;
      return 1 ;
   }
   if (bmask & 1) {
      fprintf(stderr, "B0 rules are not handled\n") ;
      return 1 ;
   }
   long long gens = atoll(argv[2]) ;
   long long maxx = 0, maxy = 0 ;
   for (auto &c : cells) {
      maxx = std::max(maxx, c.first) ;
      maxy = std::max(maxy, c.second) ;
   }
   grid g ;
   int dx = (2 * MARGIN + 63) / 64 ;
   g.w = (int)((maxx + 64) / 64) + 2 * dx ;
   g.h = (int)maxy + 1 + 4 * MARGIN ;
   g.x0 = -(long long)dx * 64 ;
   g.y0 = -2 * MARGIN ;
   galloc(g) ;
   {
      std::vector<u64> host((size_t)g.w * g.h, 0) ;
      for (auto &c : cells) {
         long long gx = c.first - g.x0, gy = c.second - g.y0 ;
         host[(size_t)gy * g.w + gx / 64] |= 1ULL << (63 - gx % 64) ;
      }
      CK(cudaMemcpy(g.a, host.data(), host.size() * 8, cudaMemcpyHostToDevice)) ;
   }
   CK(cudaDeviceSynchronize()) ;
   double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count() ;
   auto t0 = std::chrono::steady_clock::now() ;
   for (long long done=0; done<gens; ) {
      long long l, t, r, bt ;
      if (bbox(g, l, t, r, bt) &&
          (l < MARGIN || t < MARGIN || r > (long long)g.w * 64 - 1 - MARGIN || bt > g.h - 1 - MARGIN))
         grow(g, l, t, r, bt) ;
      long long n = std::min((long long)CHECK, gens - done) ;
      dim3 bs(BW, BH), gs(g.nbx, g.nby) ;
      for (long long i=0; i<n; i++) {
         gen<<<gs, bs>>>(g.a, g.b, g.w, g.h, bmask, smask, g.ca, g.cb) ;
         std::swap(g.a, g.b) ;
         std::swap(g.ca, g.cb) ;
      }
      done += n ;
   }
   unsigned long long *dpop, pop = 0 ;
   CK(cudaMalloc(&dpop, 8)) ;
   CK(cudaMemset(dpop, 0, 8)) ;
   popcount<<<1024, 256>>>(g.a, (size_t)g.w * g.h, dpop) ;
   CK(cudaMemcpy(&pop, dpop, 8, cudaMemcpyDeviceToHost)) ;
   long long l = 0, t = 0, r = 0, bt = 0 ;
   bool any = bbox(g, l, t, r, bt) ;
   CK(cudaDeviceSynchronize()) ;
   double run_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() ;
   if (any)
      printf("summary load_s=%g run_s=%g steps=1 gen=%lld pop=%llu bbox=%lld,%lld,%lld,%lld\n",
             load_s, run_s, gens, pop, l + g.x0, t + g.y0, r + g.x0, bt + g.y0) ;
   else
      printf("summary load_s=%g run_s=%g steps=1 gen=%lld pop=0 bbox=empty\n", load_s, run_s, gens) ;
   return 0 ;
}
