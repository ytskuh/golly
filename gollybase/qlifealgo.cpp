// This file is part of Golly.
// See docs/License.html for the copyright notice.

/*
 *   Inspired by Alan Hensel's Life applet and also by xlife.  Tries to
 *   improve the cache, TLB, and branching behavior for modern CPUs.
 */
#include "qlifealgo.h"
#include "liferules.h"
#include "util.h"
#include "qlifecuda.h"
#include <stdlib.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <string>
#ifdef __linux__
#include <sched.h>
#include <pthread.h>
#include <fstream>
#endif
#include <string.h>
#include <limits.h>
#include <iostream>
using namespace std ;
/*
 *   The ai array is used to figure out the index number of the bit set in
 *   the set [1, 2, 4, 8, 16, 32, 64, 128].  Also, for the value 0, it
 *   returns the result 4, to eliminate a conditional in some obscure piece
 *   of code.
 */
static unsigned char ai[129] ;
static void qlkernels(qlkernel &k01, qlkernel &k10) ;   // the vector kernels
/*
 *   This define is the size of memory to ask for at one time.  8K is a good
 *   size; we drop 16 bits because malloc overhead is probably near this.
 *
 *   Values much smaller than this will be impacted by malloc overhead (both
 *   speed and space); values much larger than this will occupy excessive
 *   memory for small universes.
 */
#define MEMCHUNK (8192-16)
/*
 *   When we need a bunch more structures of a particular size, we call this.
 *   This code allocates the memory, adds it to our universe memory allocated
 *   list, tries to maximize the cache alignment of the structures, and then
 *   links all the substructures together into a linked list which is then
 *   returned.
 */
/*
 *   This preprocessor directive is used to work around a bug in
 *   register allocation when using function inlining (-O3 or
 *   better) and gcc 3.4.2, which is very common having shipped with
 *   Fedora Core 3.
 */
#ifdef __GNUC__
__attribute__((noinline))
#endif
linkedmem *qlifealgo::filllist(int size) {
   usedmemory += MEMCHUNK ;
   if (maxmemory != 0 && usedmemory > maxmemory)
      lifefatal("exceeded user-specified memory limit") ;
   linkedmem *p, *safep, *r = (linkedmem *)calloc(MEMCHUNK, 1) ;
   int i = size & - size ;
   if (r == 0)
      lifefatal("No memory.") ;
   r->next = memused ;
   memused = r ;
   safep = p = (linkedmem *)((((g_uintptr_t)(r+1))+i-1)&-i) ;
   while (((g_uintptr_t)p) + 2 * size <= MEMCHUNK+(g_uintptr_t)r) {
      p->next = (linkedmem *)(size + (g_uintptr_t)p) ;
      p = (linkedmem *)(size + (g_uintptr_t)p) ;
   }
   return safep ;
}
/*
 *   While generations are computed by several threads (see "Parallel
 *   generations"), allocation takes a lock.
 */
struct qlallock {
   qlifealgo *a ;
   qlallock(qlifealgo *q) : a(q) {
      while (a->alloclock.test_and_set(std::memory_order_acquire))
         ;
   }
   ~qlallock() {
      a->alloclock.clear(std::memory_order_release) ;
   }
} ;
/*
 *   ... and each worker thread takes structures from the free lists
 *   QLGRAB at a time into its own lists (qlcache), so it takes the lock
 *   rarely.  A thread's lists belong to one universe (owner); the worker
 *   threads end with their universe, and parstop() resets the calling
 *   thread's lists.
 */
static const int QLGRAB = 32 ;
struct qlcache {
   qlifealgo *owner ;
   linkedmem *l[3] ;    // bricks, tiles, supertiles
} ;
static thread_local qlcache qlcaches = { 0, { 0, 0, 0 } } ;
linkedmem *qlifealgo::grab(linkedmem *&list, int size, int which) {
   if (!inpar) {
      if (list == 0)
         list = filllist(size) ;
      linkedmem *r = list ;
      list = list->next ;
      return r ;
   }
   qlcache &c = qlcaches ;
   if (c.owner != this) {
      c.owner = this ;
      c.l[0] = c.l[1] = c.l[2] = 0 ;
   }
   if (c.l[which] == 0) {
      qlallock lk(this) ;
      if (list == 0)
         list = filllist(size) ;
      linkedmem *first = list, *last = list ;
      for (int k=1; k<QLGRAB && last->next; k++)
         last = last->next ;
      list = last->next ;
      last->next = 0 ;
      c.l[which] = first ;
   }
   linkedmem *r = c.l[which] ;
   c.l[which] = r->next ;
   return r ;
}
/*
 *   Returns a structure to its free list; while generations are computed
 *   by several threads, to the calling thread's own list.
 */
void qlifealgo::release(linkedmem *&list, int which, void *p) {
   linkedmem *m = (linkedmem *)p ;
   if (!inpar) {
      m->next = list ;
      list = m ;
      return ;
   }
   qlcache &c = qlcaches ;
   if (c.owner != this) {
      c.owner = this ;
      c.l[0] = c.l[1] = c.l[2] = 0 ;
   }
   m->next = c.l[which] ;
   c.l[which] = m ;
}
#ifdef STATS
static int bricks, tiles, supertiles, rcc, dq, ds, rccs, dqs, dss ;
#define STAT(a) a
#else
#define STAT(a)
#endif
/*
 *   If we need a new empty brick, we call this.  This structure is guaranteed
 *   to be all zeros.
 */
brick *qlifealgo::newbrick() {
   brick *r ;
   r = (brick *)grab(bricklist, sizeof(brick), 0) ;
   memset(r, 0, sizeof(brick)) ;
   STAT(bricks++) ;
   return r ;
}
/*
 *   If we need a new tile, we call this.  The structure is also initialized
 *   appropriately, with all the pointers pointing to the empty brick.
 */
tile *qlifealgo::newtile() {
   tile *r ;
   r = (tile *)grab(tilelist, sizeof(tile), 1) ;
   r->b[0] = r->b[1] = r->b[2] = r->b[3] = emptybrick ;
   r->flags = -1 ;
   r->localdeltaforward = 0 ;
   STAT(tiles++) ;
   return r ;
}
/*
 *   Finally, a new supertile is provided by this routine.  It initializes
 *   all the subtiles to point to the next level down's empty tile.
 */
supertile *qlifealgo::newsupertile(int lev) {
   supertile *r ;
   r = (supertile *)grab(supertilelist, sizeof(supertile), 2) ;
   r->d[0] = r->d[1] = r->d[2] = r->d[3] = r->d[4] = r->d[5] =
                                 r->d[6] = r->d[7] = nullroots[lev-1] ;
   STAT(supertiles++) ;
   return r ;
}
/*
 *   This short little subroutine plays a very important role.  It takes bits
 *   set up according to the c01 or c10 fields of supertiles, and translates
 *   these bits into the impact on the next level up.  Essentially, it swaps
 *   the parallel bits (bits 9 through 16), any of them set, with the edge
 *   bit (bit 8), in a way that requires no conditional branches.  On a
 *   slower older processor without large branch penalties, it might be
 *   faster to use a conditional variation that executes fewer instructions,
 *   but actually this code is not totally performance critical.
 */
static int upchanging(int x) {
   int a = (x & 0x1feff) + 0x1feff ;
   return ((a >> 8) & 1) | ((a >> 16) & 2) | ((x << 1) & 0x200) |
          ((x >> 7) & 0x400) ;
}
/*
 *   If it is determined that the universe is not large enough, this
 *   subroutine adds another level to it, expanding it by a factor of 8
 *   in one dimension, depending on whether the level is even or odd.
 *
 *   It also allocates a new emptytile at the appropriate level.
 *
 *   The old root is always placed at position 4.  This allows expansion
 *   in both positive and negative directions.
 *
 *   The sequence of sizes is as follows:
 *
 *   0 31
 *   -128 127
 *   -1,152 895
 *   -9,344 7,039
 *   -74,880 56,191
 *   -599,168 449,407
 *   -4,793,472 3,595,135
 *   -38,347,904 28,760,959
 *   -306,783,360 230,087,551
 *   INT_MIN 1,840,700,287
 *   INT_MIN INT_MAX
 *
 *   Remember these are only relevant for set() calls and have nothing
 *   to do with rendering or generation.
 */
void qlifealgo::uproot() {
   if (min < -100000000)
      min = INT_MIN ;
   else
      min = 8 * min - 128 ;
   if (max > 500000000)
      max = INT_MAX ;
   else
      max = 8 * max - 121 ;
   bmin <<= 3 ;
   bmin -= 128 ;
   bmax <<= 3 ;
   bmax -= 121 ;
   minlow32 = 8 * minlow32 - 4 ;
   if (rootlev >= 38)
     lifefatal("internal:  push too deep for qlifealgo") ;
   for (int i=0; i<2; i++) {
     supertile *oroot = root ;
     rootlev++ ;
     root = newsupertile(rootlev) ;
     if (rootlev > 1)
       root->flags = 0xf0000000 |
         (upchanging(oroot->flags) << (3 + (generation.odd()))) ;
     root->d[4] = oroot ;
     if (oroot != nullroot) {
       nullroots[rootlev] = nullroot = newsupertile(rootlev) ;
     } else {
       nullroots[rootlev] = nullroot = root ;
     }
   }
   // Need to clear this because we don't have valid population values
   // in the new root.
   popValid = 0 ;
}
/*
 *   This subroutine allocates a new empty universe.  The universe starts
 *   out as a 256x256 universe.
 */
static int bc[256] ; // popcount
qlifealgo::qlifealgo() {
   nthreads = numthreads ;
   int test = (INT_MAX != 0x7fffffff) ;
   if (test)
      lifefatal("bad platform for this program") ;
   memused = 0 ;
   maxmemory = 0 ;
   poller->bailIfCalculating() ;
   generation = 0 ;
   increment = 1 ;
   tilelist = 0 ;
   supertilelist = 0 ;
   bricklist = 0 ;
   rootlev = 0 ;
   cleandowncounter = 63 ;
   usedmemory = 0 ;
   deltaforward = 0 ;
   ai[0] = 4 ; ai[1] = 0 ; ai[2] = 1 ; ai[4] = 2 ; ai[8] = 3 ;
   ai[16] = 4 ; ai[32] = 5 ; ai[64] = 6 ; ai[128] = 7 ;
   minlow32 = min = 0 ;
   max = 31 ;
   bmin = 0 ;
   bmax = 31 ;
   emptybrick = newbrick() ;
   nullroots[0] = nullroot = root = (supertile *)(emptytile = newtile()) ;
   uproot() ;
   popValid = 0 ;
   llxb = 0 ;
   llyb = 0 ;
   llbits = 0 ;
   llsize = 0 ;
   usegpu = 0 ;
   gpu = 0 ;
   gpufailed = gpuvalid = gpuahead = 0 ;
   par = 0 ;
   inpar = 0 ;
   parlev = 2 ;

   parstamp = 0 ;
   work = 0 ;
   tclock = 0 ;
   lastmode = 0 ;
   lastlev = runlev = trylev = 2 ;
   trylevel = 0 ;
   trymode = -1 ;
   tryleft = 0 ;
   trywork = trysecs = trytiles = trysum = trysumtiles = 0 ;
   nmodes = 1 ;
   for (int m=0; m<QLMODES; m++) {
      modethreads[m] = 1 ;
      parlevm[m] = 2 ;
      tmode[m] = 0 ;
      wlost[m] = 0 ;
      tnext[m] = 0 ;
      tlast[m] = -1 ;
      for (int d=0; d<2; d++) {
         lnext[m][d] = lwork[m][d] = 0 ;
         llast[m][d] = -1 ;
      }
   }
   alloclock.clear() ;
   vecok[0] = vecok[1] = 0 ;
   qlkernels(kern01, kern10) ;
   if (bc[255] == 0)
     for (int i=1; i<256; i++)
       bc[i] = bc[i & (i-1)] + 1 ;
}
/*
 *   This subroutine frees a universe.
 */
qlifealgo::~qlifealgo() {
   parstop() ;
#ifdef ENABLE_CUDA
   if (gpu)
      qlgpu_delete(gpu) ;
#endif
   while (memused) {
      linkedmem *nu = memused->next ;
      free(memused) ;
      memused = nu ;
   }
}
/*
 *   Set the max memory
 */
void qlifealgo::setMaxMemory(int newmemlimit) {
   // AKT: allow setting maxmemory to 0
   if (newmemlimit == 0) {
      maxmemory = 0 ;
      return;
   }
   if (newmemlimit < 10)
      newmemlimit = 10 ;
#ifndef GOLLY64BIT
   else if (newmemlimit > 4000)
      newmemlimit = 4000 ;
#endif
   g_uintptr_t newlimit = ((g_uintptr_t)newmemlimit) << 20 ;
   if (usedmemory > newlimit) {
      lifewarning("Sorry, more memory currently used than allowed.") ;
      return ;
   }
   maxmemory = newlimit ;
}
/*
 *   Finally, our first generation subroutine!  This one handles supertiles
 *   for even to odd generation (0->1).  What is passed in is the universe
 *   itself, the tile (this) to focus on, its three neighbor tiles (the
 *   one `parallel' to it [by the way the subtiles are stacked], the one
 *   past it, and the corner one.
 *
 *   The way we walk down the tree in this subroutine is one of the major
 *   keys to cache and TLB performance.  We walk the universe in a
 *   spatially local way, always doing an entire 32x32 tile before
 *   moving on, always doing a 256x32 supertile before moving on, always
 *   doing a 256x256 supertile, etc.  That is, if we need to access a
 *   particular brick four times in recomputing four bricks, chances are
 *   good that those four times will occur closely in time since we do
 *   spatially adjacent bricks near each other.
 *
 *   Another trick is to do the universe from bottom to top in phase 0->1
 *   and from top to bottom in phase 1->0; this also helps cut down on
 *   those cache misses.
 */
/*
 *   The generation code has a serial (PAR == 0) and a parallel (PAR == 1)
 *   version; see "Parallel generations" below for the differences.
 */
#define QLEXEDGE 1
#define QLEXPAR 2
#define QLEXCOR 4
#define QLNB9(x, exact) ((PAR && !(exact)) ? (((x) >> 9) | (((x) >> 8) & 1)) : ((x) >> 9))
#define QLCHANGING(zis, par, edge, cor, ex) \
   (((zis)->flags | (par)->flags >> 19 | \
     (((edge)->flags >> 18 | (cor)->flags >> 27) & 1) | \
     (PAR ? (((ex) & QLEXPAR) ? 0 : (par)->flags >> 9) | \
            ((((ex) & QLEXEDGE) ? 0 : (edge)->flags >> 8) & 1) | \
            ((((ex) & QLEXCOR) ? 0 : (cor)->flags >> 17) & 1) : 0)) & 0xff)
template<class T> static inline T *qlload(int par, T *const &a) {
   return par ? __atomic_load_n(&a, __ATOMIC_ACQUIRE) : a ;
}
// is this subtile (at level lev) a frozen part of a shadow (see parshadows)?
static inline int qlfrozen(supertile *p, int lev) {
   return lev == 0 ? ((tile *)p)->frozen : p->amark == -1 ;
}
template<class T> static inline void qlstore(int par, T *&a, T *v) {
   if (par)
      __atomic_store_n(&a, v, __ATOMIC_RELEASE) ;
   else
      a = v ;
}
/*
 *   Vector brick kernel.  For outer-totalistic Moore rules (see qlvecrule),
 *   p01 and p10 can compute the new values of all eight slices of a brick
 *   at once, one slice per 32-bit lane, with bit-sliced arithmetic instead
 *   of eight table lookups per slice.
 *
 *   Phase 0->1: new cell (i,k) of slice j is the rule applied to the old
 *   cells in columns i..i+2 and rows k..k+2 of slice j, where columns 4
 *   and 5 come from slice j+1 (the right brick's slice 0 for j = 7) and
 *   rows 8 and 9 from the brick below.  Phase 1->0 mirrors this with
 *   columns i-2..i and rows k-2..k (left and upper neighbors).
 *
 *   x1, x2: the old cells shifted by one and two columns; s, c: sum and
 *   carry of the three cells of each row; s1, s2, c1, c2: the same for the
 *   next two rows; t0..t3: the 4-bit count of the 3x3 block; ctr: its
 *   center cell.
 *
 *   The kernel also does the work of the per-slice loop of p01/p10 for
 *   the slices in recomp: it stores their new values and returns the
 *   change bits that loop would leave in maskprev, in their final
 *   positions (see qlchg01, qlchg10).
 */
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__aarch64__))
#define QLVEC 1
// the vector helpers are always inlined, so their vector arguments never
// pass through a call
#pragma GCC diagnostic ignored "-Wpsabi"
typedef unsigned int qlv __attribute__((vector_size(32))) ;
#define QLINLINE static inline __attribute__((always_inline))
QLINLINE qlv qlmaj(qlv a, qlv b, qlv c) {
   return (a & b) | (c & (a ^ b)) ;
}
QLINLINE qlv qlapply(qlv s, qlv s1, qlv s2, qlv c, qlv c1, qlv c2, qlv ctr,
                     const qlvrule *ru) {
   qlv sl = s ^ s1 ^ s2, sh = qlmaj(s, s1, s2) ;
   qlv cl = c ^ c1 ^ c2, ch = qlmaj(c, c1, c2) ;
   qlv t0 = sl, t1 = sh ^ cl, u = sh & cl, t2 = u ^ ch, t3 = u & ch ;
   if (ru->life)
      return ~t3 & ((~t2 & t1 & t0) | (ctr & t2 & ~t1 & ~t0)) ;
   qlv lo[4] = { ~(t0 | t1), t0 & ~t1, t1 & ~t0, t0 & t1 } ;
   qlv hi[3] = { ~(t2 | t3), t2 & ~t3, t3 & ~t2 } ;
   qlv r = t0 ^ t0 ;
   for (int v=0; v<10; v++) {
      unsigned int mb = ru->mb[v], ms = ru->ms[v] ;
      if (mb | ms)
         r |= hi[v >> 2] & lo[v & 3] & (mb ^ (ctr & (ms ^ mb))) ;
   }
   return r ;
}
QLINLINE unsigned int qlor8(qlv v) {
   v |= __builtin_shufflevector(v, v, 4, 5, 6, 7, 0, 1, 2, 3) ;
   v |= __builtin_shufflevector(v, v, 2, 3, 0, 1, 6, 7, 4, 5) ;
   v |= __builtin_shufflevector(v, v, 1, 0, 3, 2, 5, 4, 7, 6) ;
   return v[0] ;
}
/*
 *   The change bits of p01's per-slice loop, given the changes d of the
 *   eight slices (0 for those not recomputed): slice j (j = -1 for the
 *   slice left of the brick) changed in m(j) = d[j+1] | (d[j] & 0x33333333)
 *   (itself, or the two right columns that the slice to its left reads),
 *   which gives bit j+1 of "any" (any change) and of "low" (in the lower
 *   two rows).  Returns any | low << 16.
 */
QLINLINE int qlchg01(qlv d) {
   const qlv w = { 2, 4, 8, 16, 32, 64, 128, 256 } ;
   qlv z = d ^ d ;
   qlv m = __builtin_shufflevector(d, z, 1, 2, 3, 4, 5, 6, 7, 8) | (d & 0x33333333) ;
   unsigned int any = qlor8((qlv)(m != 0) & w) | (d[0] != 0) ;
   unsigned int low = qlor8((qlv)((m & 0xff) != 0) & w) | ((d[0] & 0xff) != 0) ;
   return (int)(any | low << 16) ;
}
/*
 *   The same for p10: slice j (j = 8 for the slice right of the brick)
 *   changed in m(j) = d[j-1] | (d[j] & 0xcccccccc), which gives bit 8-j of
 *   "any" and of "high" (in the upper two rows).  Returns any | high << 16.
 */
QLINLINE int qlchg10(qlv d) {
   const qlv w = { 256, 128, 64, 32, 16, 8, 4, 2 } ;
   qlv z = d ^ d ;
   qlv m = __builtin_shufflevector(z, d, 7, 8, 9, 10, 11, 12, 13, 14) | (d & 0xcccccccc) ;
   unsigned int any = qlor8((qlv)(m != 0) & w) | (d[7] != 0) ;
   unsigned int high = qlor8((qlv)((m >> 24) != 0) & w) | ((d[7] >> 24) != 0) ;
   return (int)(any | high << 16) ;
}
// the slices of recomp (bit k for slice k, or for slice 7-k if rev) as
// lane masks
QLINLINE qlv qllanes(int recomp, int rev) {
   const qlv j = { 0, 1, 2, 3, 4, 5, 6, 7 }, jr = { 7, 6, 5, 4, 3, 2, 1, 0 } ;
   qlv r = j ^ j ;
   r += (unsigned int)recomp ;
   return -((r >> (rev ? jr : j)) & 1) ;
}
// a: the brick's old slices; r: the slice right of a[7]; u, ur: the same
// for the brick below
QLINLINE qlv qlk01(const unsigned int *a, unsigned int r,
                   const unsigned int *u, unsigned int ur,
                   const qlvrule *ru) {
   qlv A, R, U, UR ;
   memcpy(&A, a, sizeof(A)) ;
   memcpy(&R, a + 1, sizeof(R)) ;
   R[7] = r ;
   memcpy(&U, u, sizeof(U)) ;
   memcpy(&UR, u + 1, sizeof(UR)) ;
   UR[7] = ur ;
   qlv x1 = ((A << 1) & 0xeeeeeeee) | ((R >> 3) & 0x11111111) ;
   qlv x2 = ((A << 2) & 0xcccccccc) | ((R >> 2) & 0x33333333) ;
   qlv ux1 = ((U << 1) & 0xeeeeeeee) | ((UR >> 3) & 0x11111111) ;
   qlv ux2 = ((U << 2) & 0xcccccccc) | ((UR >> 2) & 0x33333333) ;
   qlv s = A ^ x1 ^ x2, c = qlmaj(A, x1, x2) ;
   qlv us = U ^ ux1 ^ ux2, uc = qlmaj(U, ux1, ux2) ;
   return qlapply(s, (s << 4) | (us >> 28), (s << 8) | (us >> 24),
                  c, (c << 4) | (uc >> 28), (c << 8) | (uc >> 24),
                  (x1 << 4) | (ux1 >> 28), ru) ;
}
// a: the brick's old slices (odd copy); l: the slice left of a[0]; u, ul:
// the same for the brick above
QLINLINE qlv qlk10(const unsigned int *a, unsigned int l,
                   const unsigned int *u, unsigned int ul,
                   const qlvrule *ru) {
   qlv A, L, U, UL ;
   memcpy(&A, a, sizeof(A)) ;
   memcpy(&L, a - 1, sizeof(L)) ;
   L[0] = l ;
   memcpy(&U, u, sizeof(U)) ;
   memcpy(&UL, u - 1, sizeof(UL)) ;
   UL[0] = ul ;
   qlv x1 = ((A >> 1) & 0x77777777) | ((L << 3) & 0x88888888) ;
   qlv x2 = ((A >> 2) & 0x33333333) | ((L << 2) & 0xcccccccc) ;
   qlv ux1 = ((U >> 1) & 0x77777777) | ((UL << 3) & 0x88888888) ;
   qlv ux2 = ((U >> 2) & 0x33333333) | ((UL << 2) & 0xcccccccc) ;
   qlv s = A ^ x1 ^ x2, c = qlmaj(A, x1, x2) ;
   qlv us = U ^ ux1 ^ ux2, uc = qlmaj(U, ux1, ux2) ;
   return qlapply(s, (s >> 4) | (us << 28), (s >> 8) | (us << 24),
                  c, (c >> 4) | (uc << 28), (c >> 8) | (uc << 24),
                  (x1 >> 4) | (ux1 << 28), ru) ;
}
// brick b: the new slices from the old ones (b[0..7] for p01, b[8..15]
// for p10) into the other half, for the slices in recomp; df: changes
// to assume (deltaforward)
QLINLINE int qlstep(unsigned int *b, qlv n, int odd, int recomp,
                    unsigned int df) {
   qlv o, sel = qllanes(recomp, !odd) ;
   unsigned int *dst = odd ? b : b + 8 ;
   memcpy(&o, dst, sizeof(o)) ;
   qlv d = ((o ^ n) | df) & sel ;
   o = (n & sel) | (o & ~sel) ;
   memcpy(dst, &o, sizeof(o)) ;
   return odd ? qlchg10(d) : qlchg01(d) ;
}
// one copy of the kernels per instruction set; qlkernels() picks one
#define QLKERNELS(SUFFIX, TARGET) \
TARGET static int qlk01##SUFFIX(unsigned int *b, unsigned int r, \
      const unsigned int *u, unsigned int ur, int recomp, unsigned int df, \
      const qlvrule *ru) { \
   return qlstep(b, qlk01(b, r, u, ur, ru), 0, recomp, df) ; \
} \
TARGET static int qlk10##SUFFIX(unsigned int *b, unsigned int l, \
      const unsigned int *u, unsigned int ul, int recomp, unsigned int df, \
      const qlvrule *ru) { \
   return qlstep(b, qlk10(b + 8, l, u, ul, ru), 1, recomp, df) ; \
}
QLKERNELS(base, )
#ifdef __x86_64__
QLKERNELS(avx2, __attribute__((target("avx2"))))
QLKERNELS(avx512, __attribute__((target("avx512f,avx512vl"))))
#endif
static void qlkernels(qlkernel &k01, qlkernel &k10) {
   k01 = qlk01base ;
   k10 = qlk10base ;
#ifdef __x86_64__
   __builtin_cpu_init() ;
   if (__builtin_cpu_supports("avx512vl")) {
      k01 = qlk01avx512 ;
      k10 = qlk10avx512 ;
   } else if (__builtin_cpu_supports("avx2")) {
      k01 = qlk01avx2 ;
      k10 = qlk10avx2 ;
   }
#endif
}
#else
static void qlkernels(qlkernel &k01, qlkernel &k10) {
   k01 = k10 = 0 ;
}
#endif
/*
 *   Is the rule table tab an outer-totalistic Moore rule?  If so, fill in
 *   ru.  An entry of the table maps a 4x4 block (bit 15 is the upper left
 *   cell, then across, then down) to the new states of its four center
 *   cells (bits 5, 4, 1, 0).
 */
static int qlvecrule(const char *tab, qlvrule &ru) {
   static const int cr[4] = { 1, 1, 2, 2 }, cc[4] = { 1, 2, 1, 2 },
                    pos[4] = { 5, 4, 1, 0 } ;
   int val[2][9] ;
   for (int a=0; a<2; a++)
      for (int n=0; n<9; n++)
         val[a][n] = -1 ;
   for (int idx=0; idx<65536; idx++) {
      if (tab[idx] & ~0x33)
         return 0 ;
      for (int k=0; k<4; k++) {
         int n = 0 ;
         for (int dr=-1; dr<=1; dr++)
            for (int dc=-1; dc<=1; dc++)
               if (dr || dc)
                  n += (idx >> (15 - ((cr[k] + dr) * 4 + cc[k] + dc))) & 1 ;
         int alive = (idx >> (15 - (cr[k] * 4 + cc[k]))) & 1 ;
         int out = (tab[idx] >> pos[k]) & 1 ;
         if (val[alive][n] < 0)
            val[alive][n] = out ;
         else if (val[alive][n] != out)
            return 0 ;
      }
   }
   int life = 1 ;
   for (int t=0; t<10; t++) {
      int born = t <= 8 && val[0][t] == 1 ;
      int stays = t >= 1 && val[1][t-1] == 1 ;
      ru.mb[t] = born ? ~0u : 0 ;
      ru.ms[t] = stays ? ~0u : 0 ;
      if (born != (t == 3) || stays != (t == 3 || t == 4))
         life = 0 ;
   }
   ru.life = life ;
   return 1 ;
}
// bricks with fewer slices to recompute use the lookup table
static const int QLVECMIN = 2 ;
// the rule of the generation this thread computes (see setruletable):
// threads of parallel generations may be at different generations
static thread_local const char *ruletable ;
static thread_local const qlvrule *curvrule ;
static thread_local long long qltiles ;   // tiles computed by this thread (see parpick)
template<int PAR>
int qlifealgo::doquad01(supertile *zis, supertile *edge,
                        supertile *par, supertile *cor, int lev, int ex) {
/*
 *   First we figure out which subtiles we need to recalculate.  There will
 *   always be at least one if we got into this subroutine (except for the
 *   case of a static universe and at the root level).  To do this, we
 *   use the edge bits from the parallel supertile, and one bit each from
 *   the other two neighbor tiles, blending them into a single 8-bit
 *   recalculate int.
 *
 *   Note that the parallel and corner have already been recomputed so
 *   their changing bits are shifted up 10 positions in c.
 */
   if (!PAR)
      poller->poll() ;
   int changing = QLCHANGING(zis, par, edge, cor, ex) ;
   int x, b, nchanging = (zis->flags & 0x3ff00) << 10 ;
   supertile *p, *pf, *pu, *pfu ;
   // which of pf and pfu are exact (see "Parallel generations")
   int pfex = QLEXPAR, pfuex = QLEXCOR ;
   STAT(ds++) ;
/*
 *   Only if the first subtile needs to be recomputed do we actually need to
 *   `visit' the edge and corner neighbors.  We always keep track of the
 *   subtiles one level down.
 */
   if (changing & 1) {
      x = 7 ;
      b = 1 ;
      pf = qlload(PAR, edge->d[0]) ;
      pfu = qlload(PAR, cor->d[0]) ;
      pfex = (ex & QLEXEDGE) ? QLEXPAR : 0 ;
      pfuex = (ex & QLEXCOR) ? QLEXCOR : 0 ;
   } else {
/*
 *   Otherwise, we compute which tile we need to examine first with the help
 *   of the ai array.
 */
      b = (changing & - changing) ;
      x = 7 - ai[b] ;
      pf = zis->d[x + 1] ;
      pfu = qlload(PAR, par->d[x + 1]) ;
      pfuex = (ex & QLEXPAR) ? QLEXCOR : 0 ;
   }
   for (;;) {
      p = zis->d[x] ;
      pu = qlload(PAR, par->d[x]) ;
/*
 *   Do we need to recompute this subtile?
 */
      if (changing & b && PAR && qlfrozen(p, lev-1)) {
         changing -= b ;   // frozen: not computed, no change
      } else if (changing & b) {
/*
 *   If so, is it the canonical empty supertile for this level?  If so,
 *   allocate a new empty supertile and set the void bits appropriately.
 */
         if (zis->d[x] == nullroots[lev-1]) {
            p = (lev == 1 ? (supertile *)newtile() : newsupertile(lev-1)) ;
            qlstore(PAR, zis->d[x], p) ;
         }
/*
 *   If it's level 1, call the tile handler, else call the next level down of
 *   the supertile handler.  The return value is the changing indicators that
 *   should be propogated up.
 */
         if (lev == 1) {
            nchanging |= p01<PAR>((tile *)p, (tile *)pf, (tile *)pu, (tile *)pfu,
                                  pfex != 0) << x ;
         } else {
            nchanging |= doquad01<PAR>(p, pu, pf, pfu, lev-1,
                          ((ex & QLEXPAR) ? QLEXEDGE : 0) | pfex | pfuex) << x ;
         }
         changing -= b ;
      } else if (changing == 0)
         break ;
      b <<= 1 ;
      x-- ;
      pfu = pu ;
      pf = p ;
      pfuex = (ex & QLEXPAR) ? QLEXCOR : 0 ;
      pfex = QLEXPAR ;
   }
   zis->flags = nchanging | 0xf0000000 ;
   return upchanging(nchanging) ;
}
/*
 *   This is for odd to even generations, and the documentation is pretty
 *   much the same as for the previous subroutine.
 */
template<int PAR>
int qlifealgo::doquad10(supertile *zis, supertile *edge,
                        supertile *par, supertile *cor, int lev, int ex) {
   if (!PAR)
      poller->poll() ;
   int changing = QLCHANGING(zis, par, edge, cor, ex) ;
   int x, b, nchanging = (zis->flags & 0x3ff00) << 10 ;
   supertile *p, *pf, *pu, *pfu ;
   int pfex = QLEXPAR, pfuex = QLEXCOR ;
   STAT(ds++) ;
   if (changing & 1) {
      x = 0 ;
      b = 1 ;
      pf = qlload(PAR, edge->d[7]) ;
      pfu = qlload(PAR, cor->d[7]) ;
      pfex = (ex & QLEXEDGE) ? QLEXPAR : 0 ;
      pfuex = (ex & QLEXCOR) ? QLEXCOR : 0 ;
   } else {
      b = (changing & - changing) ;
      x = ai[b] ;
      pf = zis->d[x - 1] ;
      pfu = qlload(PAR, par->d[x - 1]) ;
      pfuex = (ex & QLEXPAR) ? QLEXCOR : 0 ;
   }
   for (;;) {
      p = zis->d[x] ;
      pu = qlload(PAR, par->d[x]) ;
      if (changing & b && PAR && qlfrozen(p, lev-1)) {
         changing -= b ;   // frozen: not computed, no change
      } else if (changing & b) {
         if (zis->d[x] == nullroots[lev-1]) {
            p = (lev == 1 ? (supertile *)newtile() : newsupertile(lev-1)) ;
            qlstore(PAR, zis->d[x], p) ;
         }
         if (lev == 1) {
            nchanging |= p10<PAR>((tile *)pfu, (tile *)pu, (tile *)pf, (tile *)p,
                                  pfex != 0) << (7-x) ;
         } else {
            nchanging |= doquad10<PAR>(p, pu, pf, pfu, lev-1,
                          ((ex & QLEXPAR) ? QLEXEDGE : 0) | pfex | pfuex) << (7-x) ;
         }
         changing -= b ;
      } else if (changing == 0)
         break ;
      b <<= 1 ;
      x++ ;
      pfu = pu ;
      pf = p ;
      pfuex = (ex & QLEXPAR) ? QLEXCOR : 0 ;
      pfex = QLEXPAR ;
   }
   zis->flags = nchanging | 0xf0000000 ;
   return upchanging(nchanging) ;
}
/*
 *   This is our monster subroutine that, with its mirror below, accounts for
 *   about 90% of the runtime.  It handles recomputation for a 32x32 tile.
 *   Passed in are the neighbor tiles:  pr (to the right), pd (down), and
 *   prd (down and to the right).
 */
template<int PAR>
int qlifealgo::p01(tile *p, tile *pr, tile *pd, tile *prd, int prex) {
   brick *db = qlload(PAR, pd->b[0]), *rdb = qlload(PAR, prd->b[0]) ;
/*
 *   Do we need to recompute the fourth brick?  This happens here because its
 *   the only place we need to pull in changing from the down and corner
 *   neighbor.
 */
   int i, recomp = (p->c[4] | pd->c[0] | QLNB9(pr->c[4], prex) | (prd->c[0] >> 8)) & 0xff ;
   STAT(dq++) ;
   qltiles++ ;
   p->c[5] = 0 ;
   p->flags |= 0xfff00000 ;
/*
 *   For each brick . . .
 */
   for (i=3; i>=0; i--) {
      brick *b = p->b[i], *rb = qlload(PAR, pr->b[i]) ;
/*
 *   Do we need to recompute?
 */
      if (recomp) {
         unsigned int traildata, trailunderdata ;
         int j, cdelta = 0, maska, maskb, maskprev = 0 ;
/*
 *   If so, set the dirty bit.  Also, if this brick is the canonical empty
 *   brick, get a new one.
 */
         p->flags |= 1 << i ;
         if (b == emptybrick) {
            b = newbrick() ;
            qlstore(PAR, p->b[i], b) ;
         }
         if (curvrule && bc[recomp] >= QLVECMIN) {
            int m = kern01(b->d, rb->d[0], db->d, rdb->d[0], recomp,
                           deltaforward | p->localdeltaforward, curvrule) ;
            p->c[i+2] |= m >> 16 ;
            p->c[i+1] = (short)(((p->c[i+1] & 0x100) << 1) | (m & 0x1ff)) ;
            goto next01 ;
         }
/*
 *   If we need to recompute the end slice, now is a good time to get the
 *   right neighbor's data.
 */
         if (recomp & 1) {
            j = 7 ;
            traildata = rb->d[0] ;
            trailunderdata = rdb->d[0] ;
         } else {
/*
 *   Otherwise we use the ai[] array to figure out where to begin in this
 *   brick.
 */
            j = ai[recomp & - recomp] ;
            recomp >>= j ;
            j = 7 - j ;
            traildata = b->d[j+1] ;
            trailunderdata = db->d[j+1] ;
         }
         trailunderdata = (traildata << 8) + (trailunderdata >> 24) ;
         for (;;) {
/*
 *   At all times here, we have traildata (the data from the slice to the
 *   right) and trailunderdata (24 bits of traildata and eight bits from
 *   the slice under and to the right).
 *
 *   Do we need  to recompute this slice?
 */
            if (recomp & 1) {
/*
 *   Our main recompute chunk recomputes a single slice.
 */
               unsigned int zisdata = b->d[j] ;
               unsigned int underdata = (zisdata << 8) + (db->d[j] >> 24) ;
               unsigned int otherdata = ((zisdata << 2) & 0xcccccccc) +
                                        ((traildata >> 2) & 0x33333333) ;
               unsigned int otherunderdata = ((underdata << 2) & 0xcccccccc) +
                                    ((trailunderdata >> 2) & 0x33333333) ;
               int newv = (ruletable[zisdata >> 16] << 26) +
                          (ruletable[underdata >> 16] << 18) +
                          (ruletable[zisdata & 0xffff] << 10) +
                          (ruletable[underdata & 0xffff] << 2) +
                          (ruletable[otherdata >> 16] << 24) +
                          (ruletable[otherunderdata >> 16] << 16) +
                          (ruletable[otherdata & 0xffff] << 8) +
                           ruletable[otherunderdata & 0xffff] ;
/*
 *   Has anything changed?
 *   Keep track of what has changed in the entire cell, the rightmost
 *   two columns, the lowest two rows, and the lowest rightmost 2x2 cell, into
 *   the maskprev int.  Do all of this without conditionals.
 */
               int delta = (b->d[j + 8] ^ newv) | deltaforward | p->localdeltaforward ;
               STAT(rcc++) ;
               b->d[j + 8] = newv ;
               maska = cdelta | (delta & 0x33333333) ;
               maskb = maska | -maska ;
               maskprev = (maskprev << 1) |
                          ((maskb >> 9) & 0x400000) | (maskb & 0x80) ;
               cdelta = delta ;
               traildata = zisdata ;
               trailunderdata = underdata ;
            } else {
/*
 *   No need to recompute?  Well, maintain our necessary invariants and bail
 *   if we're done.
 */
               maskb = cdelta | -cdelta ;
               maskprev = (maskprev << 1) |
                          ((maskb >> 9) & 0x400000) | (maskb & 0x80) ;
               if (recomp == 0)
                  break ; ;
               cdelta = 0 ;
               traildata = b->d[j] ;
               trailunderdata = (traildata << 8) + (db->d[j] >> 24) ;
            }
            recomp >>= 1 ;
            j-- ;
         }
/*
 *   Finally done with that brick!  Update our changing for the next
 *   call to p10, and or-in any changes to the lower two rows that we saw
 *   into the next brick down's changing variable.
 */
         p->c[i+2] |= (maskprev >> (6 - j)) & 0x1ff ;
         p->c[i+1] =
           (short)(((p->c[i+1] & 0x100) << 1) | (maskprev >> (21 - j))) ;
      } else
         p->c[i+1] = 0 ;
next01:
/*
 *   Calculate recomp for the next row down.
 */
      recomp = (p->c[i] | QLNB9(pr->c[i], prex)) & 0xff ;
      db = b ;
      rdb = rb ;
   }
/*
 *   Propogate the changing information for this tile to the supertile on
 *   the next level up.
 */
   recomp = p->c[5] ;
   i = recomp | p->c[0] | p->c[1] | p->c[2] | p->c[3] | p->c[4] ;
   p->localdeltaforward = 0 ;
   if (recomp)
      return 0x201 | ((recomp & 0x100) << 2) | ((i & 0x100) >> 7) ;
   else
      return i ? ((i & 0x100) >> 7) | 1 : 0 ;
}
/*
 *   This subroutine is the mirror of the one above, used for odd to even
 *   generations.
 */
template<int PAR>
int qlifealgo::p10(tile *plu, tile *pu, tile *pl, tile *p, int plex) {
   brick *ub = qlload(PAR, pu->b[3]), *lub = qlload(PAR, plu->b[3]) ;
   int i, recomp = (p->c[1] | pu->c[5] | QLNB9(pl->c[1], plex) | (plu->c[5] >> 8)) & 0xff ;
   STAT(dq++) ;
   qltiles++ ;
   p->c[0] = 0 ;
   p->flags |= 0x000fff00 ;
   for (i=0; i<=3; i++) {
      brick *b = p->b[i], *lb = qlload(PAR, pl->b[i]) ;
      if (recomp) {
         int maska, maskprev = 0, j, cdelta = 0 ;
         unsigned int traildata, trailoverdata ;
         p->flags |= 1 << i ;
         if (b == emptybrick) {
            b = newbrick() ;
            qlstore(PAR, p->b[i], b) ;
         }
         if (curvrule && bc[recomp] >= QLVECMIN) {
            int m = kern10(b->d, lb->d[15], ub->d + 8, lub->d[15], recomp,
                           deltaforward | p->localdeltaforward, curvrule) ;
            p->c[i+1] = (short)(((p->c[i+1] & 0x100) << 1) | (m & 0x1ff)) ;
            p->c[i] |= m >> 16 ;
            goto next10 ;
         }
         if (recomp & 1) {
            j = 0 ;
            traildata = lb->d[15] ;
            trailoverdata = lub->d[15] ;
         } else {
            j = ai[recomp & - recomp] ;
            traildata = b->d[j+7] ;
            trailoverdata = ub->d[j+7] ;
            recomp >>= j ;
         }
         trailoverdata = (traildata >> 8) + (trailoverdata << 24) ;
         for (;;) {
            if (recomp & 1) {
               unsigned int zisdata = b->d[j + 8] ;
               unsigned int overdata = (zisdata >> 8) + (ub->d[j + 8] << 24) ;
               unsigned int otherdata = ((zisdata >> 2) & 0x33333333) +
                                        ((traildata << 2) & 0xcccccccc) ;
               unsigned int otheroverdata = ((overdata >> 2) & 0x33333333) +
                                    ((trailoverdata << 2) & 0xcccccccc) ;
               int newv = (ruletable[otheroverdata >> 16] << 26) +
                          (ruletable[otherdata >> 16] << 18) +
                          (ruletable[otheroverdata & 0xffff] << 10) +
                          (ruletable[otherdata & 0xffff] << 2) +
                          (ruletable[overdata >> 16] << 24) +
                          (ruletable[zisdata >> 16] << 16) +
                          (ruletable[overdata & 0xffff] << 8) +
                           ruletable[zisdata & 0xffff] ;
               int delta = (b->d[j] ^ newv) | deltaforward | p->localdeltaforward ;
               STAT(rcc++) ;
               maska = cdelta | (delta & 0xcccccccc) ;
               maskprev = (maskprev << 1) |
                          (((maska | - maska) >> 9) & 0x400000) |
                          ((((maska >> 24) | 0x100) - 1) & 0x100) ;
               b->d[j] = newv ;
               cdelta = delta ;
               traildata = zisdata ;
               trailoverdata = overdata ;
            } else {
               maskprev = (maskprev << 1) |
                          (((cdelta | - cdelta) >> 9) & 0x400000) |
                          ((((cdelta >> 24) | 0x100) - 1) & 0x100) ;
               if (recomp == 0)
                  break ;
               cdelta = 0 ;
               traildata = b->d[j + 8] ;
               trailoverdata = (traildata >> 8) + (ub->d[j + 8] << 24) ;
            }
            recomp >>= 1 ;
            j++ ;
         }
         p->c[i+1] =
           (short)(((p->c[i+1] & 0x100) << 1) | (maskprev >> (14 + j))) ;
         p->c[i] |= (maskprev >> j) & 0x1ff ;
      } else
         p->c[i+1] = 0 ;
next10:
      recomp = (p->c[i+2] | QLNB9(pl->c[i+2], plex)) & 0xff ;
      ub = b ;
      lub = lb ;
   }
   recomp = p->c[0] ;
   i = recomp | p->c[1] | p->c[2] | p->c[3] | p->c[4] | p->c[5] ;
   p->localdeltaforward = 0 ;
   if (recomp)
      return 0x201 | ((recomp & 0x100) << 2) | ((i & 0x100) >> 7) ;
   else
      return i ? ((i & 0x100) >> 7) | 1 : 0 ;
}
/*
 *   Parallel generations (docs/design/2026-10-07-quicklife-parallel.md).
 *
 *   With nthreads > 1, step() can run the generations in epochs of up to
 *   63 generations (parepoch).  At the start of an epoch the calling
 *   thread picks the regions: the supertiles at level parlev that changed
 *   lately (active) and their eight neighbors (the ring, created where the
 *   tree has none).  Change moves at most one cell per generation, and an
 *   epoch is shorter than a region's side, so nothing outside the regions
 *   changes during the epoch.  Each generation, the threads run
 *   doquad01/10<1> on the regions that changed in the previous generation
 *   or whose neighbors it reads did (parregion), and then wait at a
 *   barrier (parbarrier).  At the end of the epoch the calling thread sets
 *   the flags of the tree above the regions as doquad would have
 *   (parcombine).
 *
 *   Within one phase, a tile writes only its own cells (the half of each
 *   brick that holds the new generation) and its own change bits, and
 *   reads the other half of its neighbors' bricks, which nothing writes
 *   during the phase.  The serial code reads a neighbor's change bits of
 *   the previous phase after that neighbor has been processed in this
 *   phase, when those bits have been shifted up (by one bit in tiles, by
 *   ten in supertiles).  In parallel the neighbor may not have been
 *   processed yet, so the parallel code reads both positions (QLNB9,
 *   QLCHANGING).  That can only cause extra recomputation, which gives
 *   the same cells.  New bricks and subtiles are published with release
 *   stores (qlstore), and allocation takes a lock (qlallock).  Memory is
 *   freed (mdelete) only between epochs.
 */
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define ql_relax() _mm_pause()
#elif defined(__aarch64__)
#define ql_relax() __asm__ __volatile__("yield")
#else
#define ql_relax() do {} while (0)
#endif
int qlifealgo::numthreads = 0 ;
int qlifealgo::parthreads = (int)std::thread::hardware_concurrency() ;
static const int QLSPIN = 4096 ;      // waits (see qlwait) before an idle worker sleeps
static const int QLCHUNK = 4 ;        // regions a worker takes at a time
static const int QLMINGENS = 4 ;      // shorter runs of generations are serial
static const int QLMAXDEPTH = 19 ;    // levels above the regions (3 bits each in a key, 2 levels of margin)
static const int QLYIELD = 1024 ;     // spins before a waiting thread yields
/*
 *   Waiting for other threads: pause, and after a while let other threads
 *   run (there may be more threads than CPUs, or other programs).
 */
struct qlwait {
   int n ;
   qlwait() : n(0) {}
   void operator()() {
      if (++n < QLYIELD)
         ql_relax() ;
      else
         std::this_thread::yield() ;
   }
} ;
/*
 *   A region.  nb[k]: the edge, par and cor arguments of doquad01 (k = 0)
 *   and doquad10 (k = 1); nbi[k]: their region numbers, or -1.  If the
 *   region changed in phase q, the thread that ran it sets chgq and retq
 *   to q and ret to doquad's return value, in slot q & 1 (so a region
 *   that does not change writes nothing that its neighbors read).
 */
struct alignas(64) qlregion {
   supertile *st ;
   supertile *nb[2][3] ;
   int nbi[2][3] ;
   int chgq[2], ret[2], retq[2] ;
   unsigned char active ;
   unsigned long long key ;    // position in the tree: the child numbers from the root down
   double t ;                  // time in doquad in the phases q % QLSAMPLE == 0
   long long tiles ;           // tiles computed
   // shadows (see parepoch): the region copied, its number, and the sides
   // copied (bit (ey+1)*3 + ex+1 for the side or corner in direction (ex,ey))
   supertile *src ;
   int srcidx, band ;
} ;
static const int QLSAMPLE = 8 ;
static const double QLIMBALANCE = 1.2 ;   // see parepoch
/*
 *   Each worker first takes regions from its own range, a contiguous part
 *   of the region list (regions are listed in tree order, so a range is a
 *   compact part of the universe), then from the back of the ranges of
 *   the others in its L3 group.  The ranges are split by the regions' cost (time per
 *   generation) in the previous epoch, kept in their supertiles.
 */
struct alignas(64) qlrange {
   std::atomic<unsigned long long> span ;   // regions left: front | back << 32
   int lo, hi ;
   void reset() { span.store((unsigned)lo | (unsigned long long)hi << 32, std::memory_order_relaxed) ; }
   // the owner takes up to n regions from the front, the others one from
   // the back, so that the regions taken by others are the same from one
   // generation to the next as long as the imbalance is; returns the
   // first region taken and sets e past the last, or returns -1
   int take(int own, int n, int &e) {
      unsigned long long s = span.load(std::memory_order_relaxed) ;
      for (;;) {
         int f = (int)(s & 0xffffffff), b = (int)(s >> 32) ;
         if (f >= b)
            return -1 ;
         unsigned long long ns ;
         if (own) {
            e = std::min(f + n, b) ;
            ns = (unsigned)e | (unsigned long long)b << 32 ;
         } else {
            e = b-- ;
            f = b ;
            ns = (s & 0xffffffff) | (unsigned long long)b << 32 ;
         }
         if (span.compare_exchange_weak(s, ns, std::memory_order_relaxed))
            return f ;
      }
   }
} ;
/*
 *   An L3 group: its barrier, and its shadows (see parepoch).  At a
 *   barrier the workers of a group count in their group; at a global one
 *   the last of them also counts in gcount and waits for gdone; then it
 *   releases its group.  Barriers are numbered, so nothing has to be
 *   reset between epochs.
 */
struct alignas(64) qlgroupbar {
   std::atomic<int> count ;
   std::atomic<long long> done ;   // last barrier this group passed
   int size ;
   int first ;                     // its first worker
   int shlo, shhi ;                // its shadows
   std::atomic<int> shnext ;
} ;
struct qlpar {
   int n ;
   std::vector<std::thread> threads ;
   std::mutex mx ;
   std::condition_variable cv[2] ;   // workers of the first L3 group, the others
   std::atomic<int> sleeping[2] ;
   std::atomic<long long> epoch ;    // number of epochs * 65536 + nact of the last
   std::atomic<int> quit ;
   int nact ;                        // workers taking part in this epoch
   int ngroup ;                      // workers in the L3 group of worker 0
   int nphases, odd0 ;               // generations in this epoch, parity of the first
   int clean ;                       // whether the epoch ends with parclean
   long long bar0 ;                  // barriers before this epoch
   std::vector<qlregion> regions ;   // the regions (nown), then the shadows
   int nown ;
   std::vector<int> near ;           // the regions around each region (9 each, 4 itself), or -1
   std::vector<int> rgroup ;         // the group of each region
   std::vector<int> onbi ;           // the regions' own nbi (6 each)
   std::vector<int> shmap ;          // a group's shadow of each region, or -1
   std::vector<double> weight ;      // of each region, for the ranges
   // the ranges of each mode start at these keys (see parepoch); valid
   // for blev, broot
   std::vector<unsigned long long> bounds[QLMODES] ;
   int blev[QLMODES], broot[QLMODES] ;
   qlrange *ranges ;
   std::vector<std::vector<int> > cpus ;   // CPUs of each worker (empty: no affinity)
   std::vector<std::vector<int> > order ;  // ranges each worker takes from, in turn
   std::vector<int> group ;          // L3 group of each worker
   qlgroupbar *gbar ;
   int ngroups ;                     // groups taking part in this epoch
   int bandwidth ;                   // of the shadows, in cells
   alignas(64) std::atomic<int> gcount ;
   alignas(64) std::atomic<long long> gdone ;
   qlpar(int nthreads) : n(nthreads), epoch(0), quit(0), nact(nthreads),
                         ngroup(nthreads), nphases(0), odd0(0), clean(0), bar0(0), nown(0),
                         ngroups(1), gcount(0), gdone(0) {
      sleeping[0] = sleeping[1] = 0 ;
      for (int m=0; m<QLMODES; m++)
         blev[m] = broot[m] = -1 ;
      ranges = new qlrange[n] ;
      gbar = new qlgroupbar[n] ;
      for (int i=0; i<n; i++) {
         gbar[i].count = 0 ;
         gbar[i].done = 0 ;
         gbar[i].size = 0 ;
         gbar[i].first = 0 ;
         gbar[i].shlo = gbar[i].shhi = 0 ;
         gbar[i].shnext = 0 ;
      }
   }
   ~qlpar() {
      delete [] ranges ;
      delete [] gbar ;
   }
} ;
/*
 *   On Linux, put the workers on CPUs that share an L3 cache with the
 *   calling thread first, then on the next group of CPUs, and so on, and
 *   keep each worker within its group.
 */
static void qlplace(qlpar *par) {
#ifdef __linux__
   cpu_set_t allowed ;
   if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
      return ;
   std::vector<std::string> keys ;
   std::vector<std::vector<int> > doms ;
   for (int c=0; c<CPU_SETSIZE; c++) {
      if (!CPU_ISSET(c, &allowed))
         continue ;
      std::ifstream f(("/sys/devices/system/cpu/cpu" + std::to_string(c) +
                       "/cache/index3/shared_cpu_list").c_str()) ;
      std::string key ;
      if (!(f >> key))
         key = "" ;
      size_t d = 0 ;
      while (d < keys.size() && keys[d] != key)
         d++ ;
      if (d == keys.size()) {
         keys.push_back(key) ;
         doms.push_back(std::vector<int>()) ;
      }
      doms[d].push_back(c) ;
   }
   if (doms.size() < 2)
      return ;
   int c0 = sched_getcpu() ;
   for (size_t d=0; d<doms.size(); d++)
      for (size_t k=0; k<doms[d].size(); k++)
         if (doms[d][k] == c0)
            std::swap(doms[0], doms[d]) ;
   size_t ncpus = 0 ;
   for (size_t d=0; d<doms.size(); d++)
      ncpus += doms[d].size() ;
   for (int i=0; i<par->n; i++) {
      size_t k = i % ncpus, d = 0 ;
      while (k >= doms[d].size())
         k -= doms[d++].size() ;
      par->cpus.push_back(doms[d]) ;
   }
#endif
}
/*
 *   A worker runs its own range first, then those of the workers in its
 *   L3 group, then the others.
 */
static void qlorder(qlpar *par) {
   for (int i=0; i<par->n; i++) {
      std::vector<int> o ;
      for (int pass=0; pass<2; pass++)
         for (int j=0; j<par->n; j++) {
            int k = (i + j) % par->n ;
            int same = par->cpus.empty() || par->cpus[k] == par->cpus[i] ;
            if (same == (pass == 0))
               o.push_back(k) ;
         }
      par->order.push_back(o) ;
   }
}
/*
 *   Workers 1 .. n-1; the calling thread is worker 0.  Between epochs a
 *   worker spins for a while, then sleeps; a worker left out of an epoch
 *   sleeps right away (spinning on every core can lower the clock of the
 *   busy ones).
 */
static void qlworker(qlifealgo *a, qlpar *par, int id) {
#ifdef __linux__
   if (!par->cpus.empty()) {
      cpu_set_t set ;
      CPU_ZERO(&set) ;
      for (size_t i=0; i<par->cpus[id].size(); i++)
         CPU_SET(par->cpus[id][i], &set) ;
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set) ;
   }
#endif
   int c = id < par->ngroup ? 0 : 1 ;
   long long seen = 0 ;
   int spinlimit = QLSPIN ;
   for (;;) {
      int spins = 0 ;
      qlwait w ;
      while (par->epoch.load() == seen) {
         if (par->quit.load(std::memory_order_relaxed))
            return ;
         if (++spins < spinlimit) {
            w() ;
         } else {
            std::unique_lock<std::mutex> lk(par->mx) ;
            par->sleeping[c]++ ;
            par->cv[c].wait(lk, [&] { return par->epoch.load() != seen || par->quit.load() ; }) ;
            par->sleeping[c]-- ;
         }
      }
      seen = par->epoch.load() ;
      int takepart = id < (int)(seen & 0xffff) ;
      spinlimit = takepart ? QLSPIN : 0 ;
      if (takepart)
         a->parrun(id) ;
   }
}
/*
 *   Waits until the workers of id's group (or all nact workers, if
 *   global) reached barrier b.  The last one of the group first resets its
 *   group's cursors for the next pass.
 */
void qlifealgo::parbarrier(int id, long long b, int global) {
   qlpar *P = par ;
   qlgroupbar &g = P->gbar[P->group[id]] ;
   if (g.count.fetch_add(1, std::memory_order_acq_rel) == g.size - 1) {
      g.count.store(0, std::memory_order_relaxed) ;
      for (int w=g.first; w<g.first+g.size; w++)
         P->ranges[w].reset() ;
      g.shnext.store(g.shlo, std::memory_order_relaxed) ;
      if (global) {
         if (P->gcount.fetch_add(1, std::memory_order_acq_rel) == P->ngroups - 1) {
            P->gcount.store(0, std::memory_order_relaxed) ;
            P->gdone.store(b, std::memory_order_release) ;
         } else {
            qlwait w ;
            while (P->gdone.load(std::memory_order_acquire) < b)
               w() ;
         }
      }
      g.done.store(b, std::memory_order_release) ;
   } else {
      qlwait w ;
      while (g.done.load(std::memory_order_acquire) < b)
         w() ;
   }
}
/*
 *   Phase q of the epoch for region k.
 */
void qlifealgo::parregion(int k, int q, int odd) {
   qlregion *rs = par->regions.data() ;
   qlregion &r = rs[k] ;
   int pq = q - 1, ps = pq & 1 ;   // the previous phase and its slot
   const int *ni = r.nbi[odd] ;
   int act = q == 0 || r.chgq[ps] == pq ||
             (ni[0] >= 0 && rs[ni[0]].chgq[ps] == pq) ||
             (ni[1] >= 0 && rs[ni[1]].chgq[ps] == pq) ||
             (ni[2] >= 0 && rs[ni[2]].chgq[ps] == pq) ;
   int ret = 0 ;
   if (act) {
      supertile *nb[3] ;
      for (int i=0; i<3; i++)
         nb[i] = ni[i] >= 0 ? rs[ni[i]].st : r.nb[odd][i] ;
      int timed = q % QLSAMPLE == 0 ;
      std::chrono::steady_clock::time_point t0 ;
      if (timed)
         t0 = std::chrono::steady_clock::now() ;
      long long n0 = qltiles ;
      if (odd)
         ret = doquad10<1>(r.st, nb[0], nb[1], nb[2], parlev, 0) ;
      else
         ret = doquad01<1>(r.st, nb[0], nb[1], nb[2], parlev, 0) ;
      if (timed)
         r.t += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() ;
      r.tiles += qltiles - n0 ;
   }
   if (ret) {
      r.chgq[q & 1] = q ;
      r.ret[q & 1] = ret ;
      r.retq[q & 1] = q ;
   }
}
/*
 *   After the last phase of an epoch, if the tree is due to be cleaned:
 *   mdelete below region k (odd: the parity of the next generation).  The
 *   region itself is freed, if empty, by the calling thread's mdelete
 *   after the epoch, so it keeps its dirty bit in that case.
 */
void qlifealgo::parclean(int k, int odd) {
   supertile *p = par->regions[k].st ;
   if (!(p->flags & 0x10000000))
      return ;
   int keep = 0 ;
   for (int i=0; i<8; i++)
      if (p->d[i] != nullroots[parlev-1])
         if ((p->d[i] = mdelete(p->d[i], parlev-1, odd)) != nullroots[parlev-1])
            keep++ ;
   if (keep || (p->flags & 0x3ffff))
      p->flags &= 0xefffffff ;
}
/*
 *   Worker id's part of an epoch: the copies of the shadows (then a
 *   global barrier, since other groups compute the regions copied), the
 *   phases (each followed by a barrier of the group), and a last pass that
 *   cleans the tree if due and frees the shadows (then a global barrier).
 */
void qlifealgo::parrun(int id) {
   qlpar *P = par ;
   const std::vector<int> &o = P->order[id] ;
   // the calling thread may set up the next epoch while this worker is
   // still leaving the last barrier, so it keeps its own copies
   int nphases = P->nphases, odd0 = P->odd0, nact = P->nact, clean = P->clean ;
   long long b = P->bar0 ;
   qlgroupbar &g = P->gbar[P->group[id]] ;
   for (;;) {
      int k = g.shnext.fetch_add(1, std::memory_order_relaxed) ;
      if (k >= g.shhi)
         break ;
      qlregion &r = P->regions[k] ;
      r.st = shcopy(r.src, parlev, 0, 0, r.band) ;
   }
   parbarrier(id, ++b, 1) ;
   for (int q=0; q<=nphases; q++) {
      int odd = (odd0 + q) & 1 ;
      if (q < nphases)
         setruletable(odd) ;
      if (q < nphases || clean)
         for (size_t j=0; j<o.size(); j++) {
            if (o[j] >= nact || P->group[o[j]] != P->group[id])
               continue ;
            qlrange &rg = P->ranges[o[j]] ;
            for (;;) {
               int e, k = rg.take(j == 0, QLCHUNK, e) ;
               if (k < 0)
                  break ;
               for (; k<e; k++)
                  if (q < nphases)
                     parregion(k, q, odd) ;
                  else
                     parclean(k, odd) ;
            }
         }
      for (;;) {
         int k = g.shnext.fetch_add(1, std::memory_order_relaxed) ;
         if (k >= g.shhi)
            break ;
         if (q < nphases)
            parregion(k, q, odd) ;
         else
            shfree(P->regions[k].st, parlev) ;
      }
      if (id == 0)
         poller->poll() ;
      parbarrier(id, ++b, q == nphases) ;
   }
}
/*
 *   Shadows: a copy of region s (at level lev, whose part at offset
 *   (x0,y0) cells this is).  The cells within bandwidth cells of the sides
 *   and corners in band are computed; the tiles in a ring of one tile
 *   around those are copied but frozen: never computed, and without
 *   change bits, so that they neither change nor make their neighbors
 *   recompute; beyond that the shadow is frozen and empty (frozennull).
 */
static int qlwidth(int lev) {
   return 32 << (3 * ((lev + 1) / 2)) ;
}
static int qlheight(int lev) {
   return 32 << (3 * (lev / 2)) ;
}
static int qlinband(int band, int bw, int lev, int rlev, int x0, int y0) {
   int w = qlwidth(lev), h = qlheight(lev), rw = qlwidth(rlev), rh = qlheight(rlev) ;
   for (int ey=-1; ey<=1; ey++)
      for (int ex=-1; ex<=1; ex++) {
         if (!(band & (1 << ((ey+1)*3 + ex+1))))
            continue ;
         int inx = ex < 0 ? x0 < bw : (ex > 0 ? x0 + w > rw - bw : 1) ;
         int iny = ey < 0 ? y0 < bw : (ey > 0 ? y0 + h > rh - bw : 1) ;
         if (inx && iny)
            return 1 ;
      }
   return 0 ;
}
supertile *qlifealgo::shcopy(supertile *s, int lev, int x0, int y0, int band) {
   int bw = par->bandwidth ;
   int inband = qlinband(band, bw, lev, parlev, x0, y0) ;
   if (!inband && !qlinband(band, bw + 32, lev, parlev, x0, y0))
      return frozennull[lev] ;
   if (lev == 0) {
      tile *t = (tile *)s ;
      if (t == emptytile)
         return inband ? s : frozennull[0] ;
      tile *n = (tile *)grab(tilelist, sizeof(tile), 1) ;
      memcpy(n, t, sizeof(tile)) ;
      for (int i=0; i<4; i++)
         if (t->b[i] != emptybrick) {
            n->b[i] = (brick *)grab(bricklist, sizeof(brick), 0) ;
            memcpy(n->b[i], t->b[i], sizeof(brick)) ;
         }
      if (!inband) {
         n->frozen = 1 ;
         for (int i=0; i<6; i++)
            n->c[i] = 0 ;
      }
      return (supertile *)n ;
   }
   if (s == nullroots[lev] && lev < parlev)
      return s ;
   supertile *n = (supertile *)grab(supertilelist, sizeof(supertile), 2) ;
   memcpy(n, s, sizeof(supertile)) ;
   n->amark = n->reg = 0 ;
   n->cost = 0 ;
   int cw = (lev & 1) ? qlwidth(lev-1) : 0, ch = (lev & 1) ? 0 : qlheight(lev-1) ;
   for (int x=0; x<8; x++)
      n->d[x] = shcopy(s->d[x], lev-1, x0 + x * cw, y0 + x * ch, band) ;
   return n ;
}
// frees a shadow (cleared, as the free lists keep them)
void qlifealgo::shfree(supertile *s, int lev) {
   if (s == frozennull[lev])
      return ;
   if (lev == 0) {
      tile *t = (tile *)s ;
      if (t == emptytile)
         return ;
      for (int i=0; i<4; i++)
         if (t->b[i] != emptybrick)
            release(bricklist, 0, t->b[i]) ;
      memset(t, 0, sizeof(tile)) ;
      release(tilelist, 1, t) ;
      return ;
   }
   if (s == nullroots[lev])
      return ;
   for (int x=0; x<8; x++)
      shfree(s->d[x], lev-1) ;
   memset(s, 0, sizeof(supertile)) ;
   release(supertilelist, 2, s) ;
}
/*
 *   The nine supertiles around child x of a supertile at level lev, from
 *   the nine around that supertile (index (dy+1)*3 + dx+1; 4 is the
 *   supertile itself).  Odd levels stack their children along x, even
 *   levels along y; the empty supertiles' children are the empty ones a
 *   level down.
 */
static void qlkids(supertile *const nb[9], int lev, int x, supertile *out[9]) {
   for (int dy=-1; dy<=1; dy++)
      for (int dx=-1; dx<=1; dx++) {
         int along = (lev & 1) ? dx : dy, across = (lev & 1) ? dy : dx ;
         int xi = x + along, pa = xi < 0 ? -1 : (xi > 7 ? 1 : 0) ;
         int px = (lev & 1) ? pa : across, py = (lev & 1) ? across : pa ;
         out[(dy+1)*3 + dx+1] = nb[(py+1)*3 + px+1]->d[xi & 7] ;
      }
}
int qlifealgo::isregion(supertile *s) {
   return s->reg >= 0 && s->reg < (int)par->regions.size() &&
          par->regions[s->reg].st == s ;
}
/*
 *   Epoch setup, pass 1: mark (amark) the subtree of n down to level
 *   parlev where the flags show change; returns the number of active
 *   regions.
 */
int qlifealgo::parmark(supertile *n, int lev) {
   n->amark = parstamp ;
   if (lev == parlev)
      return 1 ;
   int k = 0 ;
   for (int x=0; x<8; x++) {
      supertile *c = n->d[x] ;
      if (c != nullroots[lev-1] && (c->flags & 0x0fffffff))
         k += parmark(c, lev-1) ;
   }
   return k ;
}
/*
 *   Pass 2: list as regions the supertiles at level parlev that are marked
 *   or next to a marked one, creating them (and the supertiles above them)
 *   where the tree has none.  nb holds the nine supertiles around n, which
 *   may be empty; returns n, created if needed.  Supertiles above parlev
 *   with regions below get reg = -1 - parstamp.
 */
supertile *qlifealgo::parcollect(supertile *const nb0[9], int lev,
                                  unsigned long long key) {
   supertile *nb[9] ;
   for (int i=0; i<9; i++)
      nb[i] = nb0[i] ;
   int found = 0 ;
   for (int x=0; x<8; x++) {
      supertile *k[9] ;
      qlkids(nb, lev, x, k) ;
      int near = 0 ;
      for (int i=0; i<9; i++)
         if (k[i]->amark == parstamp)
            near = 1 ;
      if (!near)
         continue ;
      supertile *c = k[4] ;
      if (lev - 1 == parlev) {
         if (c == nullroots[lev-1])
            c = newsupertile(lev-1) ;
         qlregion r ;
         memset(&r, 0, sizeof(r)) ;
         r.st = c ;
         r.active = c->amark == parstamp ;
         r.chgq[0] = r.chgq[1] = r.retq[0] = r.retq[1] = -1 ;
         r.key = key * 8 + x ;
         c->reg = (int)par->regions.size() ;
         par->regions.push_back(r) ;
      } else {
         c = parcollect(k, lev-1, key * 8 + x) ;
         if (c == nullroots[lev-1])
            continue ;
      }
      if (nb[4] == nullroots[lev])
         nb[4] = newsupertile(lev) ;
      nb[4]->d[x] = c ;
      found = 1 ;
   }
   if (found)
      nb[4]->reg = -1 - parstamp ;
   return nb[4] ;
}
/*
 *   Pass 3: the neighbors of the regions.
 */
void qlifealgo::parlink(supertile *const nb[9], int lev) {
   for (int x=0; x<8; x++) {
      supertile *c = nb[4]->d[x] ;
      if (lev - 1 == parlev ? !isregion(c) : c->reg != -1 - parstamp)
         continue ;
      supertile *k[9] ;
      qlkids(nb, lev, x, k) ;
      if (lev - 1 > parlev) {
         parlink(k, lev-1) ;
         continue ;
      }
      // as in doquad: at even levels par is along x and edge along y, at
      // odd levels the other way around
      static const int nbidx[2][2][3] = {
         { { 7, 5, 8 }, { 1, 3, 0 } },    // even: edge, par, cor for 0->1, 1->0
         { { 5, 7, 8 }, { 3, 1, 0 } } } ; // odd
      qlregion &r = par->regions[c->reg] ;
      for (int ph=0; ph<2; ph++)
         for (int i=0; i<3; i++) {
            supertile *s = k[nbidx[parlev & 1][ph][i]] ;
            r.nb[ph][i] = s ;
            r.nbi[ph][i] = isregion(s) ? s->reg : -1 ;
         }
      for (int i=0; i<9; i++)
         par->near[9 * c->reg + i] = isregion(k[i]) ? k[i]->reg : -1 ;
   }
}
/*
 *   Epoch end: the flags of the supertiles above the regions after phase
 *   q, as doquad01 (odd = 0) or doquad10 (odd = 1) would have set them;
 *   returns n's change bits for its parent.
 */
int qlifealgo::parcombine(supertile *n, int lev, int q, int odd) {
   if (lev == parlev) {
      if (!isregion(n))
         return 0 ;
      qlregion &r = par->regions[n->reg] ;
      return r.retq[q & 1] == q ? r.ret[q & 1] : 0 ;
   }
   if (n->reg != -1 - parstamp)
      return 0 ;
   int nchanging = (n->flags & 0x3ff00) << 10 ;
   for (int x=0; x<8; x++)
      if (n->d[x] != nullroots[lev-1])
         nchanging |= parcombine(n->d[x], lev-1, q, odd) << (odd ? 7 - x : x) ;
   n->flags = nchanging | 0xf0000000 ;
   return upchanging(nchanging) ;
}
void qlifealgo::parinit() {
   par = new qlpar(nthreads) ;
   qlplace(par) ;
   qlorder(par) ;
   if (!par->cpus.empty()) {
      par->ngroup = 0 ;
      while (par->ngroup < par->n && par->cpus[par->ngroup] == par->cpus[0])
         par->ngroup++ ;
   }
   for (int i=0; i<par->n; i++) {
      int g = 0 ;
      if (i > 0 && !par->cpus.empty())
         g = par->group[i-1] + (par->cpus[i] != par->cpus[i-1]) ;
      par->group.push_back(g) ;
   }
   for (int i=1; i<nthreads; i++)
      par->threads.push_back(std::thread(qlworker, this, par, i)) ;
   // the modes (see parpick): serial, then the first n/8, n/4, n/2 and n
   // workers (those with at least 2); the workers fill the L3 groups in
   // turn, so each mode uses as few groups as it can
   nmodes = 1 ;
   for (int d=8; d>=1; d/=2)
      if (nthreads / d >= 2 && nthreads / d > modethreads[nmodes-1])
         modethreads[nmodes++] = nthreads / d ;
}
void qlifealgo::parstop() {
   if (qlcaches.owner == this)   // the calling thread ran regions as worker 0
      qlcaches.owner = 0 ;
   if (par == 0)
      return ;
   {
      std::lock_guard<std::mutex> lk(par->mx) ;
      par->quit.store(1) ;
   }
   par->cv[0].notify_all() ;
   par->cv[1].notify_all() ;
   for (size_t i=0; i<par->threads.size(); i++)
      par->threads[i].join() ;
   delete par ;
   par = 0 ;
}
/*
 *   Can the next generations run as an epoch?
 */
int qlifealgo::parok() {
   return nthreads > 1 && (gridwd == 0 || gridht == 0) && deltaforward == 0 &&
          root != nullroot && rootlev - parlev <= QLMAXDEPTH ;
}
/*
 *   The most generations an epoch may have: a region's shorter side
 *   (32 * 8^(parlev/2) cells) less a margin.
 */
int qlifealgo::parmaxgens() {
   int side = 32 << (3 * (parlev / 2)) ;
   return std::min(side - 8, cleandowncounter) ;
}
/*
 *   Choosing the mode: serial (mode 0) or epochs with the first
 *   modethreads[m] workers (mode m, see parinit), and for each mode the
 *   region level (parlevm).  Use the mode with the least time per tile
 *   computed (so that times measured on a smaller or larger pattern
 *   compare), at its level.  Other modes, and the levels next to the best
 *   mode's, are tried now and then (one at a time, the one tried longest
 *   ago first); a mode or level that lost is tried again:
 *   - once the work (tiles computed per generation, which does not depend
 *     on the mode) has grown (for a mode with more threads than the best,
 *     or a higher level) or shrunk (fewer threads, a lower level) by
 *     QLREWORK since it lost, since more threads and larger regions gain
 *     as the work grows;
 *   - else after QLTRYGAP (nmodes + 1) times the time its try is expected
 *     to cost (its extra time per tile over the tiles of a try, and
 *     switching to it and back), so that all tries together cost about
 *     1 / QLTRYGAP of the time; this catches the other changes, such as a
 *     sparse pattern spreading out.
 *   A try is QLTRY runs of which only the last QLTRYTIMED are timed, since
 *   the first runs after a change move regions between threads, and more
 *   runs until the timed ones took QLTAU seconds; for the same reason the
 *   first run after a change does not count towards the time of its mode.
 *   The time of a mode is an average over about the last QLTAU seconds,
 *   since short runs (of a few hundred microseconds) vary too much.  Each
 *   mode keeps its own ranges, so that a try leaves the best mode's as
 *   they were.
 */
static const int QLTRY = 4, QLTRYTIMED = 2 ;
static const double QLREWORK = 1.3, QLTRYGAP = 20, QLTAU = 0.004 ;
int qlifealgo::parpick() {
   if (par == 0)
      parinit() ;
   if (trymode >= 0) {
      runlev = trylev ;
      return trymode ;
   }
   int best = 0 ;
   for (int m=1; m<nmodes; m++)
      if ((tmode[m] > 0 && tmode[m] < tmode[best]) || tmode[best] == 0)
         best = m ;
   runlev = parlevm[best] ;
   // of the modes due for a try, the one tried longest ago
   int m0 = -1 ;
   for (int m=0; m<nmodes; m++)
      if (m != best && (tmode[m] == 0 || tclock >= tnext[m] ||
                        (m > best && work >= QLREWORK * wlost[m]) ||
                        (m < best && work * QLREWORK <= wlost[m])) &&
          (m0 < 0 || tlast[m] < tlast[m0]))
         m0 = m ;
   // a mode tried with fewer threads than the best gets regions at least
   // as large as the best's, with more threads at most as large (else
   // its level may be one that has not suited it for a long time)
   int lev = parlevm[best], levtry = 0 ;
   if (m0 > 0)
      lev = m0 < best ? std::max(parlevm[m0], lev) : std::min(parlevm[m0], lev) ;
   // else the levels next to the best mode's, if due
   if (m0 < 0 && best > 0) {
      int d0 = -1 ;
      for (int d=0; d<2; d++) {
         int l = parlevm[best] + (d ? 1 : -1) ;
         if (l >= 2 && l <= rootlev - 2 &&
             (tclock >= lnext[best][d] || (d && work >= QLREWORK * lwork[best][d]) ||
              (!d && work * QLREWORK <= lwork[best][d])) &&
             (d0 < 0 || llast[best][d] < llast[best][d0]))
            d0 = d ;
      }
      if (d0 >= 0) {
         m0 = best ;
         lev = parlevm[best] + (d0 ? 1 : -1) ;
         llast[best][d0] = tclock ;
         levtry = 1 ;
      }
   }
   if (m0 < 0)
      return best ;
   trymode = m0 ;
   runlev = trylev = lev ;
   trylevel = levtry ;
   if (!levtry)
      tlast[m0] = tclock ;
   tryleft = QLTRY ;
   trysecs = trytiles = trysum = trysumtiles = 0 ;
   trywork = work ;
   return trymode ;
}
/*
 *   A run of mode took secs seconds and computed tiles tiles.
 */
void qlifealgo::partimed(int mode, double secs, double tiles) {
   int first = mode != lastmode || runlev != lastlev ;
   lastmode = mode ;
   lastlev = runlev ;
   tclock += secs ;
   tiles = std::max(tiles, 1.0) ;   // nothing changed
   if (mode != trymode) {
      if (!first || tmode[mode] == 0) {
         double a = std::min(secs / QLTAU, 1.0) ;
         tmode[mode] = tmode[mode] == 0 ? secs / tiles : (1 - a) * tmode[mode] + a * secs / tiles ;
      }
      return ;
   }
   trysecs += secs ;
   trytiles += tiles ;
   if (--tryleft < QLTRYTIMED) {
      trysum += secs ;
      trysumtiles += tiles ;
   }
   if (tryleft > 0 || trysum < QLTAU)
      return ;
   double t = trysum / trysumtiles ;
   double tswitch = std::max(trysecs - t * trytiles, 0.0) ;
   trymode = -1 ;
   if (trylevel) {
      // a level next to the mode's: the level that lost keeps the work and
      // when to try it again, as for modes
      int d = trylev > parlevm[mode] ;
      double cost = std::fabs(t - tmode[mode]) * trytiles + 2 * tswitch ;
      if (t < tmode[mode]) {
         parlevm[mode] = trylev ;
         tmode[mode] = t ;
         d = 1 - d ;   // the old level is now in the other direction
      }
      lwork[mode][d] = trywork ;
      lnext[mode][d] = tclock + QLTRYGAP * (nmodes + 1) * cost ;
      return ;
   }
   // each mode compared with the one tried: the loser keeps the work of
   // the comparison and when to try it again
   for (int m=0; m<nmodes; m++)
      if (m != mode && tmode[m] > 0) {
         int lost = t < tmode[m] ? m : mode ;
         double cost = std::fabs(t - tmode[m]) * trytiles + 2 * tswitch ;
         wlost[lost] = trywork ;
         tnext[lost] = tclock + QLTRYGAP * (nmodes + 1) * cost ;
      }
   tmode[mode] = t ;
   parlevm[mode] = trylev ;
}
void qlifealgo::setruletable(int odd) {
   int k = qliferules.alternate_rules && (odd & 1) ;
   ruletable = k ? qliferules.rule1 : qliferules.rule0 ;
   curvrule = (kern01 && vecok[k]) ? &vrule[k] : 0 ;
}
/*
 *   Shadows.  With workers in more than one L3 group, a group does not
 *   read the regions of other groups during an epoch, but copies of them
 *   (shadows), which it computes along with its own regions: of each
 *   region of another group next to one of its own, the cells within
 *   bandwidth (at least e + 8) cells of its own regions.  The cells of a
 *   shadow near the edge of the copy go wrong, since the shadow does not
 *   see what is beyond, but that error moves at most one cell per
 *   generation, so it does not reach the group's own regions within the
 *   epoch.  The groups then exchange cells once per epoch instead of
 *   every generation.  Sets up the shadows' records (after the regions)
 *   and the regions' links to them.
 */
void qlifealgo::parshadows(int e) {
   qlpar *P = par ;
   int nown = P->nown ;
   P->bandwidth = (e + 8 + 31) & ~31 ;
   while ((int)frozennull.size() < parlev) {
      int lev = (int)frozennull.size() ;
      supertile *f ;
      if (lev == 0) {
         tile *t = newtile() ;
         t->flags = 0 ;
         t->frozen = 1 ;
         f = (supertile *)t ;
      } else {
         f = newsupertile(lev) ;
         for (int x=0; x<8; x++)
            f->d[x] = frozennull[lev-1] ;
         f->flags = 0 ;
         f->amark = -1 ;
      }
      frozennull.push_back(f) ;
   }
   P->rgroup.resize(nown) ;
   for (int w=0; w<P->nact; w++)
      for (int k=P->ranges[w].lo; k<P->ranges[w].hi; k++)
         P->rgroup[k] = P->group[w] ;
   P->onbi.resize(6 * nown) ;
   for (int k=0; k<nown; k++)
      for (int i=0; i<6; i++)
         P->onbi[6 * k + i] = P->regions[k].nbi[i / 3][i % 3] ;
   P->shmap.assign(nown, -1) ;
   for (int g=0; g<P->ngroups; g++) {
      qlgroupbar &gb = P->gbar[g] ;
      gb.shlo = (int)P->regions.size() ;
      if (P->ngroups > 1) {
         int lo = P->ranges[gb.first].lo, hi = P->ranges[gb.first + gb.size - 1].hi ;
         for (int k=lo; k<hi; k++)
            for (int d=0; d<9; d++) {
               int f = P->near[9 * k + d] ;
               if (f < 0 || P->rgroup[f] == g)
                  continue ;
               if (P->shmap[f] < 0) {
                  qlregion r ;
                  memset(&r, 0, sizeof(r)) ;
                  r.chgq[0] = r.chgq[1] = r.retq[0] = r.retq[1] = -1 ;
                  r.src = P->regions[f].st ;
                  r.srcidx = f ;
                  P->shmap[f] = (int)P->regions.size() ;
                  P->regions.push_back(r) ;
               }
               // f lies in direction d of k, so the side of f facing k
               // is in the opposite direction
               P->regions[P->shmap[f]].band |= 1 << (8 - d) ;
            }
         // the shadows' links: to the group's regions and shadows, to
         // supertiles outside the regions (nothing changes there), else
         // to nothing
         for (int s=gb.shlo; s<(int)P->regions.size(); s++) {
            qlregion &r = P->regions[s] ;
            for (int i=0; i<6; i++) {
               int n = P->onbi[6 * r.srcidx + i] ;
               supertile *ptr = P->regions[r.srcidx].nb[i / 3][i % 3] ;
               if (n >= 0 && P->rgroup[n] != g) {
                  n = P->shmap[n] ;
                  if (n < 0)
                     ptr = nullroots[parlev] ;
               }
               r.nbi[i / 3][i % 3] = n ;
               r.nb[i / 3][i % 3] = ptr ;
            }
         }
         // the group's regions' links to other groups go to the shadows
         for (int k=lo; k<hi; k++)
            for (int i=0; i<6; i++) {
               int n = P->onbi[6 * k + i] ;
               if (n >= 0 && P->rgroup[n] != g)
                  P->regions[k].nbi[i / 3][i % 3] = P->shmap[n] ;
            }
         for (int s=gb.shlo; s<(int)P->regions.size(); s++)
            P->shmap[P->regions[s].srcidx] = -1 ;
      }
      gb.shhi = (int)P->regions.size() ;
      gb.shnext.store(gb.shlo, std::memory_order_relaxed) ;
   }
}
/*
 *   Run e generations as an epoch with the first modethreads[mode]
 *   workers.
 */
void qlifealgo::parepoch(int e, int mode) {
   if (par == 0)
      parinit() ;
   qlpar *P = par ;
   int nact = modethreads[mode] ;
   int nactive ;
   parlev = runlev ;
   for (;;) {
      while (uproot_needed())
         uproot() ;
      while (rootlev < parlev + 2)
         uproot() ;
      parstamp++ ;
      P->regions.clear() ;
      nactive = (root->flags & 0x0fffffff) ? parmark(root, rootlev) : 0 ;
      supertile *nb[9] ;
      for (int i=0; i<9; i++)
         nb[i] = nullroots[rootlev] ;
      nb[4] = root ;
      parcollect(nb, rootlev, 0) ;
      if (!uproot_needed())
         break ;
   }
   root->reg = -1 - parstamp ;
   P->nown = (int)P->regions.size() ;
   P->near.assign(9 * P->nown, -1) ;
   {
      supertile *nb[9] ;
      for (int i=0; i<9; i++)
         nb[i] = nullroots[rootlev] ;
      nb[4] = root ;
      parlink(nb, rootlev) ;
   }
   // the ranges: contiguous in key order, so compact parts of the
   // universe, and starting at the same keys as in the previous epoch, so
   // that regions stay with their worker (and their cells in its caches),
   // unless a range costs more than QLIMBALANCE times the mean.  A
   // region's cost is its time per generation in recent epochs; regions
   // without one count as the mean cost of those with one (active), or a
   // tenth of it (ring).
   int nr = (int)P->regions.size() ;
   double known = 0, total = 0 ;
   int nknown = 0 ;
   std::vector<double> &wt = P->weight ;
   wt.resize(nr) ;
   for (int k=0; k<nr; k++)
      if (P->regions[k].active && P->regions[k].st->cost > 0) {
         known += P->regions[k].st->cost ;
         nknown++ ;
      }
   double mean = nknown ? known / nknown : 1 ;
   for (int k=0; k<nr; k++) {
      float c = P->regions[k].st->cost ;
      wt[k] = c > 0 ? c : (P->regions[k].active ? mean : 0.1 * mean) ;
      total += wt[k] ;
   }
   std::vector<unsigned long long> &bounds = P->bounds[mode] ;
   int keep = P->blev[mode] == parlev && P->broot[mode] == rootlev ;
   if (keep) {
      std::vector<double> load(nact, 0.0) ;
      int w = 0 ;
      for (int k=0; k<nr; k++) {
         while (w < nact - 1 && P->regions[k].key >= bounds[w])
            w++ ;
         load[w] += wt[k] ;
      }
      for (int i=0; i<nact; i++)
         if (load[i] * nact > QLIMBALANCE * total)
            keep = 0 ;
   }
   if (!keep) {
      bounds.assign(nact - 1, ~0ull) ;
      double acc = 0 ;
      int w = 0 ;
      for (int k=0; k+1<nr; k++) {
         acc += wt[k] ;
         while (w < nact - 1 && acc * nact >= total * (w + 1))
            bounds[w++] = P->regions[k+1].key ;
      }
      P->blev[mode] = parlev ;
      P->broot[mode] = rootlev ;
   }
   {
      int w = 0 ;
      P->ranges[0].lo = 0 ;
      for (int k=0; k<nr; k++)
         while (w < nact - 1 && P->regions[k].key >= bounds[w]) {
            P->ranges[w].hi = k ;
            P->ranges[++w].lo = k ;
         }
      while (w < nact - 1) {
         P->ranges[w].hi = nr ;
         P->ranges[++w].lo = nr ;
      }
      P->ranges[nact-1].hi = nr ;
   }
   for (int i=0; i<nact; i++)
      P->ranges[i].reset() ;
   // the groups of the workers taking part
   P->ngroups = P->group[nact-1] + 1 ;
   for (int g=0; g<P->ngroups; g++)
      P->gbar[g].size = 0 ;
   for (int i=nact-1; i>=0; i--) {
      P->gbar[P->group[i]].size++ ;
      P->gbar[P->group[i]].first = i ;
   }
   P->nact = nact ;
   parshadows(e) ;
   P->nphases = e ;
   P->odd0 = generation.odd() ;
   P->clean = cleandowncounter == e ;
   inpar = 1 ;
   P->epoch.store(((P->epoch.load() >> 16) + 1) << 16 | nact) ;
   for (int c=0; c<2; c++)
      if ((c == 0 || nact > P->ngroup) && P->sleeping[c].load() > 0) {
         { std::lock_guard<std::mutex> lk(P->mx) ; }
         P->cv[c].notify_all() ;
      }
   parrun(0) ;
   inpar = 0 ;
   P->bar0 += e + 2 ;
   P->regions.resize(nr) ;   // the shadows are freed
   // each region's time per generation, for the next epoch's ranges
   int nsampled = (e + QLSAMPLE - 1) / QLSAMPLE ;
   long long tiles = 0 ;
   for (int k=0; k<nr; k++) {
      supertile *st = P->regions[k].st ;
      float c = (float)(P->regions[k].t / nsampled) ;
      st->cost = st->cost > 0 ? 0.5f * (st->cost + c) : c ;
      tiles += P->regions[k].tiles ;
   }
   work = (double)tiles / e ;
   if (e >= 2)
      parcombine(root, rootlev, e - 2, (P->odd0 + e - 2) & 1) ;
   parcombine(root, rootlev, e - 1, (P->odd0 + e - 1) & 1) ;
   // the level is chosen by time (see parpick), but with fewer active
   // regions than workers some would have nothing to do
   if (nactive < nact && parlev > 2 && parlev == parlevm[mode])
      parlevm[mode] = parlev - 1 ;
   generation += bigint(e) ;
   popValid = 0 ;
   cleandowncounter -= e ;
   if (cleandowncounter == 0) {
      cleandowncounter = 63 ;
      mdelete(root, rootlev, generation.odd()) ;
   }
}
/**
 *   Mark a node and its subnodes as changed.  We really
 *   only mark those nodes that have any cells set at all.
 *   And we remove all empty nodes, to prevent quadratic
 *   expansion if setrule() or something similar is called
 *   too frequently.
 */
supertile *qlifealgo::markglobalchange(supertile *p, int lev, int &bits) {
   int i ;
   bits = 0 ;
   if (lev == 0) {
      tile *pp = (tile *)p ;
      if (pp != emptytile) {
        int s = 0 ;
        for (int i=0; i<4; i++)
          for (int j=0; j<16; j++)
            s |= pp->b[i]->d[j] ;
        if (s) {
          pp->c[0] = pp->c[5] = 0x1ff ;
          pp->c[1] = pp->c[2] = pp->c[3] = pp->c[4] = 0x3ff ;
          bits = 0x603 ;
          return p ;
        }
        bits = 0 ;
        for (int i=0; i<4; i++)
           if (pp->b[i] != emptybrick) {
               STAT(bricks--) ;
               ((linkedmem *)(pp->b[i]))->next = bricklist ;
               bricklist = (linkedmem *)(pp->b[i]) ;
           }
        STAT(tiles--) ;
        memset(pp, 0, sizeof(tile)) ;
        ((linkedmem *)pp)->next = tilelist ;
        tilelist = (linkedmem *)pp ;
        return (supertile *)emptytile ;
      }
      return p ;
   } else {
      if (p != nullroots[lev]) {
         int nchanging = 0 ;
         int nbits ;
         if (generation.odd()) {
           for (i=0; i<8; i++) {
              p->d[i] = markglobalchange(p->d[i], lev-1, nbits) ;
              nchanging |= nbits << i ;
           }
         } else {
           for (i=0; i<8; i++) {
             p->d[i] = markglobalchange(p->d[i], lev-1, nbits) ;
             nchanging |= nbits << (7-i) ;
           }
         }
         if (nchanging != 0 || p == root) {
            p->flags |= nchanging | 0xf0000000 ;
            bits = upchanging(nchanging) ;
            return p ;
         } else {
            STAT(supertiles--) ;
            memset(p, 0, sizeof(supertile)) ;
            ((linkedmem *)p)->next = supertilelist ;
            supertilelist = (linkedmem *)p ;
            return nullroots[lev] ;
         }
      }
      return p ;
   }
}
void qlifealgo::markglobalchange() {
   int bits = 0 ;
   markglobalchange(root, rootlev, bits) ;
   deltaforward = 0xffffffff ;
}
/*
 *   This subroutine sets a bit at a particular location.
 *
 *   We walk down the tree to the particular bit, setting changing flags as
 *   we go.
 */
int qlifealgo::setcell(int x, int y, int newstate) {
   if (gpuahead)
      gpusync() ;
   gpuvalid = 0 ;
   if (newstate & ~1)
      return -1 ;
   y = - y ;
   supertile *b ;
   tile *p ;
   int lev ;
   int odd = generation.odd() ;
   if (odd) {
      x-- ;
      y-- ;
   }
   while (x < min || x > max || y < min || y > max)
      uproot() ;
   int xdel = (x >> 5) - minlow32 ;
   int ydel = (y >> 5) - minlow32 ;
   int xc = x - (minlow32 << 5) ;
   int yc = y - (minlow32 << 5) ;
   if (root == nullroot)
      root = newsupertile(rootlev) ;
   b = root ;
   lev = rootlev ;
   while (lev > 0) {
      int i, d = 1 ;
      if (lev & 1) {
         int s = (lev >> 1) + lev - 1 ;
         i = (xdel >> s) & 7 ;
         s = (1 << (s + 5)) - 2 ;
         if ((xc & s) == ((odd) ? s : 0))
            d += 2 ;
         if ((yc & s) == ((odd) ? s : 0))
            d += d << 9 ;
      } else {
         int s = (lev >> 1) + lev - 3 ;
         i = (ydel >> s) & 7 ;
         s = (1 << (s + 5)) - 2 ;
         if ((yc & s) == ((odd) ? s : 0))
            d += 2 ;
         s |= s << 3 ;
         if ((xc & s) == ((odd) ? s : 0))
            d += d << 9 ;
      }
      if (odd)
         b->flags |= (d << i) | 0xf0000000 ;
      else
         b->flags |= (d << (7 - i)) | 0xf0000000 ;
      if (b->d[i] == nullroots[lev-1])
         b->d[i] = (lev==1 ? (supertile *)newtile() :
                                                      newsupertile(lev-1)) ;
      lev -= 1 ;
      b = b->d[i] ;
   }
   x &= 31 ;
   y &= 31 ;
   p = (tile *)b ;
   if (p->b[(y >> 3) & 0x3] == emptybrick)
      p->b[(y >> 3) & 0x3] = newbrick() ;
   if (odd) {
      int mor = ((x & 2) ? 3 : 1) << ((x >> 2) & 0x7) ;
      p->c[((y >> 3) & 0x3) + 1] |= mor ;
      p->flags = -1 ;
      if ((y & 6) == 6)
         p->c[((y >> 3) & 0x3) + 2] |= mor ;
      if (newstate)
         p->b[(y >> 3) & 0x3]->d[8 + ((x >> 2) & 0x7)]
                                   |= (1 << (31 - (y & 7) * 4 - (x & 3))) ;
      else
         p->b[(y >> 3) & 0x3]->d[8 + ((x >> 2) & 0x7)]
                                  &= ~(1 << (31 - (y & 7) * 4 - (x & 3))) ;
      p->localdeltaforward |= (1 << (31 - (y & 7) * 4 - (x & 3))) ;
   } else {
      int mor = ((x & 2) ? 1 : 3) << (7 - ((x >> 2) & 0x7)) ;
      p->c[((y >> 3) & 0x3) + 1] |= mor ;
      p->flags = -1 ;
      if ((y & 6) == 0)
         p->c[((y >> 3) & 0x3)] |= mor ;
      if (newstate)
         p->b[(y >> 3) & 0x3]->d[(x >> 2) & 0x7]
                                   |= (1 << (31 - (y & 7) * 4 - (x & 3))) ;
      else
         p->b[(y >> 3) & 0x3]->d[(x >> 2) & 0x7]
                                  &= ~(1 << (31 - (y & 7) * 4 - (x & 3))) ;
      p->localdeltaforward |= (1 << (31 - (y & 7) * 4 - (x & 3))) ;
   }
   return 0 ;
}
/*
 *   This subroutine gets a bit at a particular location.
 */
int qlifealgo::getcell(int x, int y) {
   if (gpuahead)
      gpusync() ;
   y = - y ;
   supertile *b ;
   tile *p ;
   int lev ;
   int odd = generation.odd() ;
   if (odd) {
      x-- ;
      y-- ;
   }
   while (x < min || x > max || y < min || y > max)
      uproot() ;
   if (x < min || x > max || y < min || y > max)
      return 0 ;
   int xdel = (x >> 5) - minlow32 ;
   int ydel = (y >> 5) - minlow32 ;
   if (root == nullroot)
      return 0 ;
   b = root ;
   lev = rootlev ;
   while (lev > 0) {
      int i ;
      if (lev & 1) {
         int s = (lev >> 1) + lev - 1 ;
         i = (xdel >> s) & 7 ;
      } else {
         int s = (lev >> 1) + lev - 3 ;
         i = (ydel >> s) & 7 ;
      }
      if (b->d[i] == nullroots[lev-1])
         return 0 ;
      lev -= 1 ;
      b = b->d[i] ;
   }
   x &= 31 ;
   y &= 31 ;
   p = (tile *)b ;
   if (p->b[(y >> 3) & 0x3] == emptybrick)
      return 0 ;
   if (odd) {
      if (p->b[(y >> 3) & 0x3]->d[8 + ((x >> 2) & 0x7)] &
          (1 << (31 - (y & 7) * 4 - (x & 3))))
         return 1 ;
      else
         return 0 ;
   } else {
      if (p->b[(y >> 3) & 0x3]->d[(x >> 2) & 0x7] &
          (1 << (31 - (y & 7) * 4 - (x & 3))))
         return 1 ;
      else
         return 0 ;
   }
}
/**
 *   Similar but returns the distance to the next set cell horizontally.
 */
int qlifealgo::nextcell(int x, int y, int &v) {
   if (gpuahead)
      gpusync() ;
   v = 1 ;
   y = - y ;
   int odd = generation.odd() ;
   if (odd) {
      x-- ;
      y-- ;
   }
   while (x < min || x > max || y < min || y > max)
      uproot() ;
   if (x > max || x < min || y < min || y > max)
      return -1 ;
   return nextcell(x, y, root, rootlev) ;
}
int qlifealgo::nextcell(int x, int y, supertile *n, int lev) {
   if (lev > 0) {
      if (n == nullroots[lev])
         return -1 ;
      int xdel = (x >> 5) - minlow32 ;
      int ydel = (y >> 5) - minlow32 ;
      int i ;
      if (lev & 1) {
         int s = (lev >> 1) + lev - 1 ;
         i = (xdel >> s) & 7 ;
         int r = 0 ;
         int off = (x & 31) + ((xdel & ((1 << s) - 1)) << 5) ;
         while (i < 8) {
            int t = nextcell(x, y, n->d[i], lev-1) ;
            if (t < 0) {
              r += (32 << s) - off ;
              x += (32 << s) - off ;
              off = 0 ;
            } else {
              return r + t ;
            }
            i++ ;
         }
         return -1 ;
      } else {
         int s = (lev >> 1) + lev - 3 ;
         i = (ydel >> s) & 7 ;
         return nextcell(x, y, n->d[i], lev-1) ;
      }
   }
   x &= 31 ;
   y &= 31 ;
   tile *p = (tile *)n ;
   brick *br = (brick *)(p->b[(y>>3) & 3]) ;
   if (br == emptybrick)
      return -1 ;
   int i = ((x >> 2) & 7) ;
   int add = (generation.odd() ? 8 : 0) ;
   int sh = (7 - (y & 7)) * 4 ;
   int r = 0 ;
   x &= 3 ;
   int m = 15 >> x ;
   while (i < 8) {
     int t = (br->d[i+add] >> sh) & m ;
     if (t) {
       if (t & 8) return r - x ;
       if (t & 4) return r + 1 - x ;
       if (t & 2) return r + 2 - x ;
       return r + 3 - x ;
     }
     r += (4 - x) ;
     x = 0 ;
     m = 15 ;
     i++ ;
   }
   return -1 ;
}
/*
 *   This subroutine calculates the population count of the universe.  It
 *   uses dirty bits number 1 and 2 of supertiles.
 */
G_INT64 qlifealgo::find_set_bits(supertile *p, int lev, int gm1) {
   G_INT64 pop = 0 ;
   int i, j, b ;
   if (lev == 0) {
      tile *pp = (tile *)p ;
      b = 8 + gm1 * 12 ;
      pop = (pp->flags >> b) & 0xfff ;
      if (pop > 0x800) {
         pop = 0 ;
         for (i=0; i<4; i++) {
            if (pp->b[i] != emptybrick) {
               for (j=0; j<8; j++) {
#ifdef FASTPOPCOUNT
                  pop += FASTPOPCOUNT(pp->b[i]->d[j+gm1*8]) ;
#else
                  int k = pp->b[i]->d[j+gm1*8] ;
                  if (k)
                    pop += bc[k & 255] + bc[(k >> 8) & 255] +
                      bc[(k >> 16) & 255] + bc[(k >> 24) & 255] ;
#endif
               }
            }
         }
         pp->flags = (int)((pp->flags & ~(0xfff << b)) | (pop << b)) ;
      }
   } else {
      if (p->flags & (0x20000000 << gm1)) {
         for (i=0; i<8; i++)
            if (p->d[i] != nullroots[lev-1])
               pop += find_set_bits(p->d[i], lev-1, gm1) ;
         if (pop < 500000000) {
            p->pop[gm1] = (int)pop ;
            p->flags &= ~(0x20000000 << gm1) ;
         } else {
            p->pop[gm1] = 0xfffffff ; // placeholder; *some* bits are set
         }
      } else {
         pop = p->pop[gm1] ;
      }
   }
   return pop ;
}
/**
 *   A variation that tries to quickly answer:  *any* bits set?
 */
int qlifealgo::isEmpty(supertile *p, int lev, int gm1) {
   int i, j, k, b ;
   if (lev == 0) {
      tile *pp = (tile *)p ;
      b = 8 + gm1 * 12 ;
      int pop = (pp->flags >> b) & 0xfff ;
      if (pop > 0x800) {
         pop = 0 ;
         for (i=0; i<4; i++) {
            if (pp->b[i] != emptybrick) {
               for (j=0; j<8; j++) {
                  k = pp->b[i]->d[j+gm1*8] ;
                  if (k)
                     return 0 ;
               }
            }
         }
      }
      return pop ? 0 : 1 ;
   } else {
      if (p->flags & (0x20000000 << gm1)) {
         for (i=0; i<8; i++)
            if (p->d[i] != nullroots[lev-1])
               if (!isEmpty(p->d[i], lev-1, gm1))
                  return 0 ;
         return 1 ;
      } else {
         return p->pop[gm1] ? 0 : 1 ;
      }
   }
}
/*
 *   Another critical subroutine, this one cleans up the empty bricks,
 *   tiles, and supertiles as the generations go by.  This speeds things
 *   up by not using too much memory (minimizing cache misses and TLB
 *   misses).  We only try to delete bricks, tiles, and supertiles from
 *   regions of the universe that have been active since we last attempted
 *   to delete tiles.  We delete all possible tiles, even those near active
 *   regions; if necessary, 
 *
 *   We use dirty bit number 0 of supertiles, and dirty bits 0..3 of
 *   tiles.
 */
supertile *qlifealgo::mdelete(supertile *p, int lev, int odd) {
   int i ;
   if (lev == 0) {
      tile *pp = (tile *)p ;
      if (pp->flags & 0xf) {
         int seen = 0 ;
         for (i=0; i<4; i++) {
            brick *b = pp->b[i] ;
            if (b != emptybrick) {
               if ((pp->flags & (1 << i))) {
                  if (b->d[0] | b->d[1] | b->d[2] | b->d[3] | b->d[4] |
                      b->d[5] | b->d[6] | b->d[7] | b->d[8] | b->d[9] |
                      b->d[10] | b->d[11] | b->d[12] | b->d[13] | b->d[14] |
                      b->d[15]) {
                     seen++ ;
                  } else {
                     STAT(bricks--) ;
                     release(bricklist, 0, b) ;
                     pp->b[i] = emptybrick ;
                  }
               } else
                  seen++ ;
            }
         }
         if (seen || ((pp->c[1] | pp->c[2] | pp->c[3] | pp->c[4]) & 0xff) ||
                                 (odd ? pp->c[5] : pp->c[0]))
            pp->flags &= 0xfffffff0 ;
         else {
            STAT(tiles--) ;
            memset(pp, 0, sizeof(tile)) ;
            release(tilelist, 1, pp) ;
            return nullroots[lev] ;
         }
      }
   } else {
      if (p->flags & 0x10000000) {
         int keep = 0 ;
         for (i=0; i<8; i++)
            if (p->d[i] != nullroots[lev-1])
               if ((p->d[i] = mdelete(p->d[i], lev-1, odd)) !=
                                                       nullroots[lev-1])
                  keep++ ;
         if (keep || p == root || (p->flags & 0x3ffff))
            p->flags &= 0xefffffff ;
         else {
            STAT(supertiles--) ;
            memset(p, 0, sizeof(supertile)) ;
            release(supertilelist, 2, p) ;
            return nullroots[lev] ;
         }
      }
   }
   return p ;
}
G_INT64 qlifealgo::popcount() {
   return find_set_bits(root, rootlev, generation.odd()) ;
}
const bigint &qlifealgo::getPopulation() {
   if (!popValid) {
#ifdef ENABLE_CUDA
      if (gpuahead)
         population = bigint(qlgpu_population(gpu, generation.odd())) ;
      else
#endif
      population = bigint(popcount()) ;
      popValid = 1 ;
      poller->reset_countdown() ;
   }
   return population ;
}
int qlifealgo::isEmpty() {
   if (gpuahead)
      return getPopulation() == 0 ;
   return isEmpty(root, rootlev, generation.odd()) ;
}
/*
 *   Here we look at the root node and see if activity is getting
 *   uncomfortably close to the current edges.  If so, we add another
 *   level onto the top.
 */
int qlifealgo::uproot_needed() {
   int i ;
   if (root->d[0] != nullroots[rootlev-1] ||
       root->d[7] != nullroots[rootlev-1])
      return 1 ;
   for (i=1; i<7; i++)
      if (root->d[i]->d[0] != nullroots[rootlev-2] ||
          root->d[i]->d[7] != nullroots[rootlev-2])
         return 1 ;
   return 0 ;
}
/*
 *   The new generation code is simple.  We uproot if needed.  Then, we call
 *   the appropriate top-level slice code depending on the generation number.
 *   Finally, if we are generations 64, 128, 192, and so on, we clean up
 *   the tree.
 *
 *   Note that this 64 was carefully chosen to balance extraneous bricks left
 *   behind by gliders against the computational cost of deletion.
 */
void qlifealgo::dogen() {
   poller->reset_countdown() ;
#ifdef STATS
   ds = 0 ; dq = 0 ; rcc = 0 ;
#endif
   // AKT: if grid is bounded then we should never need to call uproot() here
   // because setrule() has already expanded the universe to enclose the grid
   if (gridwd == 0 || gridht == 0) {
      while (uproot_needed())
         uproot() ;
   }
   if (generation.odd())
      doquad10<0>(root, nullroot, nullroot, nullroot, rootlev, 7) ;
   else
      doquad01<0>(root, nullroot, nullroot, nullroot, rootlev, 7) ;
   deltaforward = 0 ;
   generation += bigint::one ;
   popValid = 0 ;
   if (--cleandowncounter == 0) {
      cleandowncounter = 63 ;
      mdelete(root, rootlev, generation.odd()) ;
   }
#ifdef STATS
   dss += ds ; dqs += dq ; rccs += rcc ;
#endif
}
/**
 *   Step.  Do increment generations.
 */
void qlifealgo::step() {
   poller->bailIfCalculating() ;
   if (gpuok()) {
      gpustep() ;
      return ;
   }
#ifdef __linux__
   // the calling thread is worker 0 of parallel generations: keep it in
   // its L3 group during the step (see qlplace)
   cpu_set_t saved ;
   int restore = 0 ;
   if (par && !par->cpus.empty() && sched_getaffinity(0, sizeof(saved), &saved) == 0) {
      cpu_set_t set ;
      CPU_ZERO(&set) ;
      for (size_t i=0; i<par->cpus[0].size(); i++)
         CPU_SET(par->cpus[0][i], &set) ;
      restore = sched_setaffinity(0, sizeof(set), &set) == 0 ;
   }
#endif
   bigint t = increment ;
   while (t != 0) {
      // with nthreads > 1, run up to parmaxgens() generations at a time in
      // the mode that took the least time per generation lately (see
      // parpick)
      int e = 1, mode = 0 ;
      if (parok()) {
         e = parmaxgens() ;
         if (t < bigint(e))
            e = t.toint() ;
         if (e >= QLMINGENS)
            mode = parpick() ;
         else
            e = 1 ;
      }
      std::chrono::steady_clock::time_point t0 ;
      if (e > 1)
         t0 = std::chrono::steady_clock::now() ;
      long long n0 = qltiles ;
      if (mode) {
         parepoch(e, mode) ;
      } else {
         for (int i=0; i<e; i++) {
            setruletable(generation.odd()) ;
            dogen() ;
            if (poller->isInterrupted())
               break ;
         }
      }
      if (poller->isInterrupted())
         break ;
      if (e > 1) {
         double tg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / e ;
         if (mode == 0)
            work = (double)(qltiles - n0) / e ;
         partimed(mode, tg * e, work * e) ;
      }
      t -= e ;
      if (t > increment) // might change; make it happen now
         t = increment ;
   }
#ifdef __linux__
   if (restore)
      sched_setaffinity(0, sizeof(saved), &saved) ;
#endif
}

/*
 *   GPU generations.  A universe made as "QuickLife CUDA" (usegpu) runs its
 *   generations on a CUDA GPU (qlifecuda.cu) when there is one, the grid
 *   is unbounded and the rule is outer-totalistic (vecok, see qlvecrule);
 *   otherwise on one thread as QuickLife.
 *
 *   The GPU keeps its own copy of the cells.  step() loads the tree's
 *   cells into it when they changed since (gpuvalid == 0) and leaves the
 *   tree behind (gpuahead); every function that reads or changes cells
 *   first rebuilds the tree from the GPU's cells (gpusync).  So a run of
 *   steps never copies cells between the two.
 */
#ifdef ENABLE_CUDA
// generations between checks for interrupts
static const int QLGPUGENS = 4096 ;
int qlifealgo::gpuok() {
   if (!usegpu || gpufailed || gridwd > 0 || gridht > 0 || !vecok[0] ||
       (qliferules.alternate_rules && !vecok[1]))
      return 0 ;
   if (gpu == 0 && (gpu = qlgpu_new()) == 0) {
      gpufailed = 1 ;
      return 0 ;
   }
   return 1 ;
}
// the non-empty bricks of the generation of parity odd below p
void qlifealgo::gpubricks(supertile *p, int lev, int xdel, int ydel, int odd,
                          std::vector<qlgbrick> &v) {
   if (lev == 0) {
      tile *t = (tile *)p ;
      if (t == emptytile)
         return ;
      for (int i=0; i<4; i++) {
         brick *b = t->b[i] ;
         if (b == emptybrick)
            continue ;
         qlgbrick k ;
         unsigned int any = 0 ;
         for (int j=0; j<8; j++)
            any |= (k.d[j] = b->d[8 * odd + j]) ;
         if (any == 0)
            continue ;
         k.x = (xdel + minlow32) * 32 ;
         k.y = (ydel + minlow32) * 32 + 8 * i ;
         v.push_back(k) ;
      }
   } else {
      if (p == nullroots[lev])
         return ;
      int s = (lev & 1) ? (lev >> 1) + lev - 1 : (lev >> 1) + lev - 3 ;
      for (int i=0; i<8; i++)
         if (lev & 1)
            gpubricks(p->d[i], lev-1, xdel + (i << s), ydel, odd, v) ;
         else
            gpubricks(p->d[i], lev-1, xdel, ydel + (i << s), odd, v) ;
   }
}
void qlifealgo::gpustep() {
   if (!gpuvalid) {
      std::vector<qlgbrick> v ;
      gpubricks(root, rootlev, 0, 0, generation.odd(), v) ;
      qlgpu_load(gpu, v, generation.odd()) ;
      gpuvalid = 1 ;
   }
   unsigned int born[2], stays[2] ;
   for (int p=0; p<2; p++) {
      const qlvrule &ru = vrule[qliferules.alternate_rules ? p : 0] ;
      born[p] = stays[p] = 0 ;
      for (int t=0; t<10; t++) {
         if (ru.mb[t])
            born[p] |= 1 << t ;
         if (ru.ms[t])
            stays[p] |= 1 << t ;
      }
   }
   bigint t = increment ;
   while (t != 0) {
      int n = t > bigint(QLGPUGENS) ? QLGPUGENS : t.toint() ;
      qlgpu_run(gpu, n, generation.odd(), born, stays) ;
      gpuahead = 1 ;
      generation += bigint(n) ;
      popValid = 0 ;
      t -= n ;
      poller->reset_countdown() ;
      if (poller->poll())
         break ;
      if (t > increment)
         t = increment ;
   }
}
// free every brick, tile and supertile below p
void qlifealgo::freetree(supertile *p, int lev) {
   if (lev == 0) {
      tile *t = (tile *)p ;
      if (t == emptytile)
         return ;
      for (int i=0; i<4; i++)
         if (t->b[i] != emptybrick)
            release(bricklist, 0, t->b[i]) ;
      memset(t, 0, sizeof(tile)) ;
      release(tilelist, 1, t) ;
   } else {
      if (p == nullroots[lev])
         return ;
      for (int i=0; i<8; i++)
         freetree(p->d[i], lev-1) ;
      memset(p, 0, sizeof(supertile)) ;
      release(supertilelist, 2, p) ;
   }
}
// store a brick of the generation of parity odd, as setcell stores a cell
void qlifealgo::putbrick(const qlgbrick &k, int odd) {
   int x = k.x, y = k.y ;
   while (x < min || x + 31 > max || y < min || y + 7 > max)
      uproot() ;
   int xdel = (x >> 5) - minlow32 ;
   int ydel = (y >> 5) - minlow32 ;
   if (root == nullroot)
      root = newsupertile(rootlev) ;
   supertile *b = root ;
   int lev = rootlev ;
   while (lev > 0) {
      int i ;
      if (lev & 1)
         i = (xdel >> ((lev >> 1) + lev - 1)) & 7 ;
      else
         i = (ydel >> ((lev >> 1) + lev - 3)) & 7 ;
      b->flags |= 0xf0000000 ;
      if (b->d[i] == nullroots[lev-1])
         b->d[i] = (lev == 1 ? (supertile *)newtile() : newsupertile(lev-1)) ;
      lev -= 1 ;
      b = b->d[i] ;
   }
   tile *p = (tile *)b ;
   int j = (y >> 3) & 3 ;
   if (p->b[j] == emptybrick)
      p->b[j] = newbrick() ;
   for (int i=0; i<8; i++)
      p->b[j]->d[8 * odd + i] = k.d[i] ;
}
// rebuild the tree from the GPU's cells; every tile counts as changed
void qlifealgo::gpusync() {
   std::vector<qlgbrick> v ;
   qlgpu_save(gpu, v, generation.odd()) ;
   freetree(root, rootlev) ;
   root = nullroot ;
   for (size_t i=0; i<v.size(); i++)
      putbrick(v[i], generation.odd()) ;
   markglobalchange() ;
   popValid = 0 ;
   gpuahead = 0 ;
}
#else
int qlifealgo::gpuok() { return 0 ; }
void qlifealgo::gpustep() {}
void qlifealgo::gpusync() {}
#endif

// Flip bits in given rule table.
// This is a tad tricky because we want to turn both the input
// and the output of this table upside down.
static void fliprule(char *rptr) {
   for (int i=0; i<65536; i++) {
      int j = ((i & 0xf) << 12) +
               ((i & 0xf0) << 4) + ((i & 0xf00) >> 4) + ((i & 0xf000) >> 12) ;
      if (i <= j) {
         char fi = rptr[i] ;
         char fj = rptr[j] ;
         fi = ((fi & 0x30) >> 4) + ((fi & 0x3) << 4) ;
         fj = ((fj & 0x30) >> 4) + ((fj & 0x3) << 4) ;
         rptr[i] = fj ;
         rptr[j] = fi ;
      }
   }
}

/**
 *   If we change the rule we need to mark everything dirty.
 */
const char *qlifealgo::setrule(const char *s) {
   if (gpuahead)
      gpusync() ;
   gpuvalid = 0 ;
   const char* err = qliferules.setrule(s, this);
   if (err) return err;

   markglobalchange() ;
   
   // AKT: qlifealgo has an opposite interpretation of the orientation of
   // a rule table assumed by qliferules.setrule.  For vertically symmetrical
   // rules such as the Moore or von Neumann neighborhoods this doesn't matter,
   // but for hexagonal rules and Wolfram rules we need to flip the rule table(s)
   // upside down.
   if ( qliferules.isHexagonal() || qliferules.isWolfram() ) {
      if (qliferules.alternate_rules) {
         // hex rule has B0 but not S6 so we'll be using rule1 for odd gens
         fliprule(qliferules.rule1);
      }
      fliprule(qliferules.rule0);
   }
   
   vecok[0] = qlvecrule(qliferules.rule0, vrule[0]) ;
   vecok[1] = qliferules.alternate_rules && qlvecrule(qliferules.rule1, vrule[1]) ;
   
   if (qliferules.isHexagonal())
      grid_type = HEX_GRID;
   else if (qliferules.isVonNeumann())
      grid_type = VN_GRID;
   else
      grid_type = SQUARE_GRID;

   // AKT: if the grid is bounded then call uproot() if necessary so that
   // dogen() never needs to call it
   if (gridwd > 0 && gridht > 0) {
      // use the top left and bottom right corners of the grid, but expanded by 2
      // to allow for growth in the borders when the grid edges are joined
      int xmin = -int(gridwd/2) - 2;
      int ymin = -int(gridht/2) - 2;
      int xmax = xmin + gridwd + 3;
      int ymax = ymin + gridht + 3;
      // duplicate the expansion code in setcell()
      ymin = -ymin;
      ymax = -ymax;
      if (generation.odd()) {
         xmin--;
         ymin--;
         xmax--;
         ymax--;
      }
      // min is -ve, max is +ve, xmin is -ve, xmax is +ve, ymin is +ve, ymax is -ve
      while (xmin < min || xmax > max || ymin > max || ymax < min)
         uproot();
   }
   
   return 0;
}

static lifealgo *creator() { return new qlifealgo() ; }

void qlifealgo::doInitializeAlgoInfo(staticAlgoInfo &ai) {
   ai.setAlgorithmName("QuickLife") ;
   ai.setAlgorithmCreator(&creator) ;
   ai.setDefaultBaseStep(10) ;
   ai.setDefaultMaxMem(0) ;
   ai.minstates = 2 ;
   ai.maxstates = 2 ;
   // init default color scheme
   ai.defgradient = false;
   ai.defr1 = ai.defg1 = ai.defb1 = 255;        // start color = white
   ai.defr2 = ai.defg2 = ai.defb2 = 255;        // end color = white
   ai.defr[0] = ai.defg[0] = ai.defb[0] = 48;   // 0 state = dark gray
   ai.defr[1] = ai.defg[1] = ai.defb[1] = 255;  // 1 state = white
}

static lifealgo *parcreator() {
   qlifealgo *a = new qlifealgo() ;
   a->nthreads = qlifealgo::parthreads ;
   return a ;
}

void qlifealgo::doInitializeParAlgoInfo(staticAlgoInfo &ai) {
   doInitializeAlgoInfo(ai) ;
   ai.setAlgorithmName("QuickLife Parallel") ;
   ai.setAlgorithmCreator(&parcreator) ;
}

static lifealgo *cudacreator() {
#ifdef ENABLE_CUDA
   qlgpu_warmup() ;
#endif
   qlifealgo *a = new qlifealgo() ;
   a->nthreads = 1 ;
   a->usegpu = 1 ;
   return a ;
}

void qlifealgo::doInitializeCudaAlgoInfo(staticAlgoInfo &ai) {
   doInitializeAlgoInfo(ai) ;
   ai.setAlgorithmName("QuickLife CUDA") ;
   ai.setAlgorithmCreator(&cudacreator) ;
}
