// This file is part of Golly.
// See docs/License.html for the copyright notice.

/*
 * hlife 0.99 by Radical Eye Software.
 *
 *   All good ideas here were originated by Gosper or Bell or others, I'm
 *   sure, and all bad ones by yours truly.
 *
 *   The main reason I wrote this program was to attempt to push out the
 *   evaluation of metacatacryst as far as I could.  So this program
 *   really does very little other than compute life as far into the
 *   future as possible, using as little memory as possible (and reusing
 *   it if necessary).  No UI, few options.
 */
#include "hlifealgo.h"
#include "util.h"
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>
#include <string>
#ifdef __linux__
#include <sched.h>
#include <pthread.h>
#include <fstream>
#include <sys/mman.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define cpu_relax() _mm_pause()
#elif defined(__aarch64__)
#define cpu_relax() __asm__ __volatile__("yield")
#else
#define cpu_relax() do {} while (0)
#endif
using namespace std ;
/*
 *   Note that all the places we represent 4-squares by short, we use
 *   unsigned shorts; this is so we can directly index into these arrays.
 */
static unsigned char shortpop[65536] ;
/*
 *   During a parallel step, a res field with the low bit set means the
 *   result is being computed (see the parallel stepping section).
 */
#define ISPARBUSY(r) (((g_uintptr_t)(r)) & 1)
/*
 *   The cached result of an 8-square is a new 4-square representing
 *   two generations into the future.  This subroutine calculates that
 *   future, assuming ruletable is calculated (see below).  The code
 *   that it uses is similar to code you'll see again, so we explain
 *   what's going on in some detail.
 *
 *   Each time we build a leaf node, we compute the result, because it
 *   is reasonably quick.
 *
 *   The first generation result of an 8-square is a 6-square, which
 *   we represent as nine 2-squares.  The nine 2-squares are called
 *   t00 through t22, and are arranged in a matrix:
 *
 *      t00   t01   t02
 *      t10   t11   t12
 *      t20   t21   t22
 *
 *   To compute each of these, we need to extract the relevant bits
 *   from the four 4-square values n->nw, n->ne, n->sw, and n->ne.
 *   We can use these values to directly index into the ruletable
 *   array.
 *
 *   Then, given the nine values, we can compute a resulting 4-square
 *   by computing four 2-square results, and combining these into a
 *   single 4-square.
 *
 *   It's a bit intricate, but it's not really overwhelming.
 */
#define combine9(t00,t01,t02,t10,t11,t12,t20,t21,t22) \
       ((t00) << 15) | ((t01) << 13) | (((t02) << 11) & 0x1000) | \
       (((t10) << 7) & 0x880) | ((t11) << 5) | (((t12) << 3) & 0x110) | \
       (((t20) >> 1) & 0x8) | ((t21) >> 3) | ((t22) >> 5)
void hlifealgo::leafres(leaf *n) {
   n->leafpop = bigint((short)(shortpop[n->nw] + shortpop[n->ne] +
                               shortpop[n->sw] + shortpop[n->se])) ;
   if (leafnk >= 0)   // res1 and res2 are only used by the table-driven leaf code
      return ;
   unsigned short
   t00 = ruletable[n->nw],
   t01 = ruletable[((n->nw << 2) & 0xcccc) | ((n->ne >> 2) & 0x3333)],
   t02 = ruletable[n->ne],
   t10 = ruletable[((n->nw << 8) & 0xff00) | ((n->sw >> 8) & 0x00ff)],
   t11 = ruletable[((n->nw << 10) & 0xcc00) | ((n->ne << 6) & 0x3300) |
                   ((n->sw >> 6) & 0x00cc) | ((n->se >> 10) & 0x0033)],
   t12 = ruletable[((n->ne << 8) & 0xff00) | ((n->se >> 8) & 0x00ff)],
   t20 = ruletable[n->sw],
   t21 = ruletable[((n->sw << 2) & 0xcccc) | ((n->se >> 2) & 0x3333)],
   t22 = ruletable[n->se] ;
   n->res1 = combine9(t00,t01,t02,t10,t11,t12,t20,t21,t22) ;
   n->res2 =
   (ruletable[(t00 << 10) | (t01 << 8) | (t10 << 2) | t11] << 10) |
   (ruletable[(t01 << 10) | (t02 << 8) | (t11 << 2) | t12] << 8) |
   (ruletable[(t10 << 10) | (t11 << 8) | (t20 << 2) | t21] << 2) |
    ruletable[(t11 << 10) | (t12 << 8) | (t21 << 2) | t22] ;
}
/*
 *   Direct leaf stepping.  For an outer-totalistic rule (a cell's next
 *   state depends only on its own state and its number of live
 *   neighbours), the 8-square result of a 16-square made of four leaves
 *   is computed from the cells themselves, with bitwise operations on
 *   rows of 16 cells (one cell per bit), instead of from the results of
 *   nine and then four 8-squares looked up in the hash.  Only the result
 *   leaf is looked up.
 *
 *   setleafrule() finds out from ruletable whether the rule is
 *   outer-totalistic and, if it is, lists the neighbour counts k that
 *   give a live cell: leafbm[j] is all ones if a dead cell with
 *   leafk[j] neighbours is born, leafsm[j] if a live one survives.
 *   Otherwise leafnk is -1 and the table-driven code is used.
 */
void hlifealgo::setleafrule() {
   int f[2][9] ;   // next state of a cell for (own state, live neighbours)
   for (int a=0; a<2; a++)
      for (int k=0; k<9; k++)
         f[a][k] = -1 ;
   leafnk = -1 ;
   for (int i=0; i<65536; i++) {
      int out = ruletable[i] ;
      for (int r=1; r<3; r++)
         for (int c=1; c<3; c++) {
            int b = 15 - 4 * r - c ;
            int cnt = 0 ;
            for (int dr=-1; dr<=1; dr++)
               for (int dc=-1; dc<=1; dc++)
                  if (dr || dc)
                     cnt += (i >> (b - 4 * dr - dc)) & 1 ;
            int a = (i >> b) & 1 ;
            int nx = (out >> (b - 5)) & 1 ;
            if (f[a][cnt] < 0)
               f[a][cnt] = nx ;
            else if (f[a][cnt] != nx)
               return ;
         }
   }
   leafnk = 0 ;
   for (int k=0; k<9; k++)
      if (f[0][k] || f[1][k]) {
         leafk[leafnk] = (unsigned char)k ;
         leafbm[leafnk] = f[0][k] ? 0xffff : 0 ;
         leafsm[leafnk] = f[1][k] ? 0xffff : 0 ;
         leafnk++ ;
      }
}
/*
 *   Compute the center 8-square of the 16-square made of leaves n, ne, t
 *   and e (nw, ne, sw, se), gens (1, 2 or 4) generations ahead, into the
 *   four 4-squares q[0..3] (nw, ne, sw, se).  Row r of the 16-square is
 *   a[r+1], with the westmost cell in bit 15; a[0] and a[17] stay zero.
 *   Each generation, the cells near the edge become wrong, one more row
 *   and column each time; the center 8-square is still right after 4.
 */
void hlifealgo::leafstep(const leaf *n, const leaf *ne, const leaf *t,
                         const leaf *e, int gens, unsigned short *q) const {
   unsigned short a[18], b[18] ;
   a[0] = a[17] = b[0] = b[17] = 0 ;
   for (int k=0; k<4; k++) {
      int s = 12 - 4 * k ;
      a[1+k] = (unsigned short)((((n->nw >> s) & 0xf) << 12) | (((n->ne >> s) & 0xf) << 8) |
                                (((ne->nw >> s) & 0xf) << 4) | ((ne->ne >> s) & 0xf)) ;
      a[5+k] = (unsigned short)((((n->sw >> s) & 0xf) << 12) | (((n->se >> s) & 0xf) << 8) |
                                (((ne->sw >> s) & 0xf) << 4) | ((ne->se >> s) & 0xf)) ;
      a[9+k] = (unsigned short)((((t->nw >> s) & 0xf) << 12) | (((t->ne >> s) & 0xf) << 8) |
                                (((e->nw >> s) & 0xf) << 4) | ((e->ne >> s) & 0xf)) ;
      a[13+k] = (unsigned short)((((t->sw >> s) & 0xf) << 12) | (((t->se >> s) & 0xf) << 8) |
                                 (((e->sw >> s) & 0xf) << 4) | ((e->se >> s) & 0xf)) ;
   }
   unsigned short *cur = a, *nxt = b ;
   for (int g=0; g<gens; g++) {
      // live neighbour count of every cell, as bit planes c0..c3
      unsigned short c0[16], c1[16], c2[16], c3[16], res[16] ;
      for (int r=0; r<16; r++) {
         unsigned short u = cur[r], m = cur[r+1], d = cur[r+2] ;
         unsigned short ul = (unsigned short)(u << 1), ur = (unsigned short)(u >> 1) ;
         unsigned short dl = (unsigned short)(d << 1), dr = (unsigned short)(d >> 1) ;
         unsigned short ml = (unsigned short)(m << 1), mr = (unsigned short)(m >> 1) ;
         unsigned short ux = ul ^ ur, u0 = ux ^ u, u1 = (ul & ur) | (u & ux) ;
         unsigned short dx = dl ^ dr, d0 = dx ^ d, d1 = (dl & dr) | (d & dx) ;
         unsigned short m0 = ml ^ mr, m1 = ml & mr ;
         unsigned short x = u0 ^ d0, k1 = (u0 & d0) | (m0 & x) ;
         unsigned short y = u1 ^ d1, p0 = y ^ m1, p1 = (u1 & d1) | (m1 & y) ;
         unsigned short q4 = p0 & k1 ;
         c0[r] = x ^ m0 ;
         c1[r] = p0 ^ k1 ;
         c2[r] = p1 ^ q4 ;
         c3[r] = p1 & q4 ;
         res[r] = 0 ;
      }
      for (int j=0; j<leafnk; j++) {
         int k = leafk[j] ;
         unsigned short x0 = (k & 1) ? 0 : 0xffff, x1 = (k & 2) ? 0 : 0xffff,
                        x2 = (k & 4) ? 0 : 0xffff, x3 = (k & 8) ? 0 : 0xffff ;
         unsigned short bm = leafbm[j], sm = leafsm[j] ;
         for (int r=0; r<16; r++) {
            unsigned short m = cur[r+1] ;
            res[r] |= (c0[r] ^ x0) & (c1[r] ^ x1) & (c2[r] ^ x2) & (c3[r] ^ x3) &
                      ((m & sm) | (~m & bm)) ;
         }
      }
      for (int r=0; r<16; r++)
         nxt[r+1] = res[r] ;
      unsigned short *tmp = cur ;
      cur = nxt ;
      nxt = tmp ;
   }
   q[0] = q[1] = q[2] = q[3] = 0 ;
   for (int k=0; k<4; k++) {
      int s = 12 - 4 * k ;
      unsigned short top = cur[5+k], bot = cur[9+k] ;
      q[0] |= ((top >> 8) & 0xf) << s ;
      q[1] |= ((top >> 4) & 0xf) << s ;
      q[2] |= ((bot >> 8) & 0xf) << s ;
      q[3] |= ((bot >> 4) & 0xf) << s ;
   }
}
/*
 *   We do now support garbage collection, but there are some routines we
 *   call frequently to help us.
 */
#ifdef PRIMEMOD
#define node_hash(a,b,c,d) (65537*(g_uintptr_t)(d)+257*(g_uintptr_t)(c)+17*(g_uintptr_t)(b)+5*(g_uintptr_t)(a))
#else
g_uintptr_t node_hash(void *a, void *b, void *c, void *d) {
   g_uintptr_t r = (65537*(g_uintptr_t)(d)+257*(g_uintptr_t)(c)+17*(g_uintptr_t)(b)+5*(g_uintptr_t)(a)) ;
   r += (r >> 11) ;
   return r ;
}
#endif
#define leaf_hash(a,b,c,d) (65537*(d)+257*(c)+17*(b)+5*(a))
double hlifealgo::maxloadfactor = 0.7 ;
/*
 *   The hash table.  Nodes and leaves are allocated in blocks (newnode)
 *   and never move, so nodes made one after another sit next to each
 *   other in memory.  The table is an index of 64-bit entries
 *   (open addressing, linear probing): 0 for an empty entry, otherwise a
 *   pointer to a hashed node or leaf with 16 bits of its hash (the tag)
 *   above the pointer bits, so a lookup reads only the nodes whose tag
 *   matches.  Entries are added during steps and only removed by gc,
 *   which clears the index and puts back the nodes it keeps.  When more
 *   than limit (maxloadfactor of the entries) are used, the index is
 *   replaced by one INDEXGROW times as large; that can happen at any time,
 *   since nodes do not move.  Growing by 4 rather than 2 halves the number
 *   of times every node is read to find its new entry, and the entries
 *   are small (8 bytes, against 48 for a node).  A hashed leaf has isnode
 *   SLOTLEAF, so is_node() tells it from a node.
 */
static const g_uintptr_t SLOTLEAF = 1 ;
static const g_uintptr_t INDEXMIN = 1 << 12 ;    // entries of the smallest index
static const g_uintptr_t INDEXMINPAR = 1 << 17 ; // and for parallel steps
static const g_uintptr_t INDEXGROW = 4 ;         // growth factor
typedef unsigned long long hlentry ;
#ifdef GOLLY64BIT
static const int HLPTRBITS = 48 ;   // user-space pointers fit in 48 bits
#else
static const int HLPTRBITS = 32 ;
#endif
static const hlentry HLPTRMASK = (1ULL << HLPTRBITS) - 1 ;
static inline g_uintptr_t hlmix(g_uintptr_t h) {
   return h * (g_uintptr_t)0x9E3779B97F4A7C15ULL ;
}
// the first entry to look at for mixed hash x, in an index of cap entries
static inline g_uintptr_t hlslot(g_uintptr_t x, g_uintptr_t cap) {
#ifdef __SIZEOF_INT128__
   return (g_uintptr_t)(((unsigned __int128)x * cap) >> 64) ;
#else
   return x % cap ;
#endif
}
static inline hlentry hltag(g_uintptr_t x) {
   return (hlentry)((x >> 16) & 0xffff) << HLPTRBITS ;
}
static inline node *hlptr(hlentry e) {
   return (node *)(g_uintptr_t)(e & HLPTRMASK) ;
}
static inline g_uintptr_t hlnext(g_uintptr_t i, g_uintptr_t cap) {
   return i + 1 == cap ? 0 : i + 1 ;
}
static inline g_uintptr_t hlhash(node *p) {
   if (is_node(p))
      return node_hash(p->nw, p->ne, p->sw, p->se) ;
   leaf *l = (leaf *)p ;
   return leaf_hash(l->nw, l->ne, l->sw, l->se) ;
}
/*
 *   Ask for 2 MB pages for the whole pages in p..p+bytes-1 (Linux); the
 *   index and the nodes are read in no particular order, and with 4 KB
 *   pages most of those reads also miss the TLB.
 */
static void hladvise(void *p, g_uintptr_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
   g_uintptr_t lo = ((g_uintptr_t)p + 4095) & ~(g_uintptr_t)4095 ;
   g_uintptr_t hi = ((g_uintptr_t)p + bytes) & ~(g_uintptr_t)4095 ;
   if (hi > lo)
      madvise((void *)lo, hi - lo, MADV_HUGEPAGE) ;
#endif
}
void hlifealgo::index_alloc(hlindex &x, g_uintptr_t cap) {
   x.e = (hlentry *)calloc(cap, sizeof(hlentry)) ;
   if (x.e == 0)
      lifefatal("Out of memory; try reducing the hash memory limit.") ;
   hladvise(x.e, cap * sizeof(hlentry)) ;
   x.cap = cap ;
   x.used = 0 ;
   x.limit = (g_uintptr_t)(maxloadfactor * cap) ;
}
/*
 *   Add p, which is not in x yet.
 */
void hlifealgo::index_put(hlindex &x, node *p) {
   g_uintptr_t m = hlmix(hlhash(p)) ;
   g_uintptr_t i = hlslot(m, x.cap) ;
   while (x.e[i])
      i = hlnext(i, x.cap) ;
   x.e[i] = (hlentry)(g_uintptr_t)p | hltag(m) ;
   x.used++ ;
}
/*
 *   Put the nodes of entries lo..hi-1 of x into nx (with PAR, while other
 *   threads put other nodes into nx).  Finding where a node goes means
 *   reading the node, and the nodes are spread over memory, so this goes
 *   in batches of HLBATCH: prefetch the nodes of the batch, then compute
 *   their hashes and prefetch the entries of nx they go to, then put them
 *   there.
 */
static const int HLBATCH = 128 ;
template<int PAR> static void index_move(const hlindex &x, g_uintptr_t lo,
                                         g_uintptr_t hi, hlindex &nx) {
   hlentry b[HLBATCH] ;
   g_uintptr_t m[HLBATCH] ;
   for (g_uintptr_t i=lo; i<hi; ) {
      int k = 0 ;
      for (; i<hi && k<HLBATCH; i++)
         if (x.e[i]) {
            b[k] = x.e[i] ;
            node *p = hlptr(b[k++]) ;
            PREFETCH((char *)p + 8) ;
            PREFETCH((char *)p + 39) ;
         }
      for (int t=0; t<k; t++) {
         m[t] = hlmix(hlhash(hlptr(b[t]))) ;
         PREFETCH(nx.e + hlslot(m[t], nx.cap)) ;
      }
      for (int t=0; t<k; t++) {
         hlentry v = (b[t] & HLPTRMASK) | hltag(m[t]) ;
         for (g_uintptr_t j=hlslot(m[t], nx.cap); ; j=hlnext(j, nx.cap)) {
            if (PAR) {
               hlentry e = 0 ;
               if (__atomic_load_n(nx.e + j, __ATOMIC_RELAXED) == 0 &&
                   __atomic_compare_exchange_n(nx.e + j, &e, v, false,
                                               __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                  break ;
            } else if (nx.e[j] == 0) {
               nx.e[j] = v ;
               break ;
            }
         }
      }
   }
}
/*
 *   Replace the index by one of ncap entries.
 */
void hlifealgo::index_grow(g_uintptr_t ncap) {
   if (verbose) {
     sprintf(statusline, "Resizing hash to %" PRIuPTR "...", ncap) ;
     lifestatus(statusline) ;
   }
   hlindex nx ;
   index_alloc(nx, ncap) ;
   index_move<0>(idx, 0, idx.cap, nx) ;
   nx.used = idx.used ;
   free(idx.e) ;
   alloced += (ncap - idx.cap) * sizeof(hlentry) ;
   idx = nx ;
   if (verbose) {
     strcpy(statusline+strlen(statusline), " done.") ;
     lifestatus(statusline) ;
   }
}
/*
 *   More than limit entries are used: grow the index.  If the larger
 *   index would not fit in maxmem, gc first (during a step), and grow
 *   only if that leaves the index more than 80% of the limit full (as
 *   newnode() allocates over maxmem when gc does not free enough).
 */
void hlifealgo::index_full() {
   if (okaytogc && alloced + (INDEXGROW - 1) * idx.cap * sizeof(hlentry) > maxmem) {
      do_gc(0) ;
      if (idx.used <= idx.limit - idx.limit / 5)
         return ;
   }
   index_grow(INDEXGROW * idx.cap) ;
}
/*
 *   These next two routines are (nearly) our only hash table access
 *   routines; we simply look up the passed in information.  If we
 *   find it in the hash table, we return it; otherwise, we build a
 *   new node and store it in the hash table, and return that.  Building
 *   the node is a separate function, so the lookup itself calls nothing
 *   and needs few saved registers.  A node found after the first entry
 *   looked at (its home) is swapped with the node at its home, so nodes
 *   that are looked up often are found at once.  This keeps every node
 *   reachable: the entries between the two are all used.
 */
node *hlifealgo::find_node_h(g_uintptr_t h, node *nw, node *ne, node *sw, node *se) {
   g_uintptr_t x = hlmix(h) ;
   hlentry tag = hltag(x) ;
   hlentry *home = idx.e + hlslot(x, idx.cap), *q = home ;
   for (;;) {
      hlentry e = *q ;
      if (e == 0)
         return find_node_new(nw, ne, sw, se) ;
      if (((e ^ tag) >> HLPTRBITS) == 0) {
         node *p = (node *)(g_uintptr_t)(e ^ tag) ;
         if (p->nw == nw && p->ne == ne && p->sw == sw && p->se == se) {
            if (q != home) {
               *q = *home ;
               *home = e ;
            }
            return save(p) ;
         }
      }
      if (++q == idx.e + idx.cap)
         q = idx.e ;
   }
}
/*
 *   Make the node that find_node_h() did not find.
 */
node *hlifealgo::find_node_new(node *nw, node *ne, node *sw, node *se) {
   g_uintptr_t x = hlmix(node_hash(nw, ne, sw, se)) ;
   hlentry tag = hltag(x) ;
   node *p = newnode() ;   // may gc, which rebuilds the index
   g_uintptr_t i = hlslot(x, idx.cap) ;
   while (idx.e[i])
      i = hlnext(i, idx.cap) ;
   p->nw = nw ;
   p->ne = ne ;
   p->sw = sw ;
   p->se = se ;
   p->res = 0 ;
   p->next = 0 ;
   idx.e[i] = (hlentry)(g_uintptr_t)p | tag ;
   idx.used++ ;
   hashpop++ ;
   save(p) ;
   if (idx.used > idx.limit)
      index_full() ;
   return p ;
}
node *hlifealgo::find_node(node *nw, node *ne, node *sw, node *se) {
   return find_node_h(node_hash(nw, ne, sw, se), nw, ne, sw, se) ;
}
leaf *hlifealgo::find_leaf(unsigned short nw, unsigned short ne,
                                  unsigned short sw, unsigned short se) {
   g_uintptr_t x = hlmix(leaf_hash(nw, ne, sw, se)) ;
   hlentry tag = hltag(x) ;
   hlentry *home = idx.e + hlslot(x, idx.cap), *q = home ;
   for (;;) {
      hlentry e = *q ;
      if (e == 0)
         return find_leaf_new(nw, ne, sw, se) ;
      if (((e ^ tag) >> HLPTRBITS) == 0) {
         leaf *p = (leaf *)(g_uintptr_t)(e ^ tag) ;
         if ((g_uintptr_t)p->isnode == SLOTLEAF &&
             p->nw == nw && p->ne == ne && p->sw == sw && p->se == se) {
            if (q != home) {
               *q = *home ;
               *home = e ;
            }
            return (leaf *)save((node *)p) ;
         }
      }
      if (++q == idx.e + idx.cap)
         q = idx.e ;
   }
}
leaf *hlifealgo::find_leaf_new(unsigned short nw, unsigned short ne,
                               unsigned short sw, unsigned short se) {
   g_uintptr_t x = hlmix(leaf_hash(nw, ne, sw, se)) ;
   hlentry tag = hltag(x) ;
   leaf *p = newleaf() ;
   g_uintptr_t i = hlslot(x, idx.cap) ;
   while (idx.e[i])
      i = hlnext(i, idx.cap) ;
   p->isnode = (node *)SLOTLEAF ;
   p->nw = nw ;
   p->ne = ne ;
   p->sw = sw ;
   p->se = se ;
   leafres(p) ;
   p->wnum = 0 ;
   p->next = 0 ;
   idx.e[i] = (hlentry)(g_uintptr_t)p | tag ;
   idx.used++ ;
   hashpop++ ;
   save((node *)p) ;
   if (idx.used > idx.limit)
      index_full() ;
   return p ;
}
/*
 *   The following routine does the same, but first it checks to see if
 *   the cached result is any good.  If it is, it directly returns that.
 *   Otherwise, it figures out whether to call the leaf routine or the
 *   non-leaf routine by whether two nodes down is a leaf node or not.
 *   (We'll understand why this is a bit later.)  All the sp stuff is
 *   stack pointer and garbage collection stuff.
 */
node *hlifealgo::getres(node *n, int depth) {
   if (n->res)
     return n->res ;
   node *res = 0 ;
   /**
    *   This routine be the only place we assign to res.  We use
    *   the fact that the poll routine is *sticky* to allow us to
    *   manage unwinding the stack without munging our data
    *   structures.  Note that there may be many find_nodes
    *   and getres called before we finally actually exit from
    *   here, because the stack is deep and we don't want to
    *   put checks throughout the code.  Instead we need two
    *   calls here, one to prevent us going deeper, and another
    *   to prevent us from destroying the cache field.
    */
   if (poller->poll() || softinterrupt)
     return zeronode(depth-1) ;
   int sp = gsp ;
   if (running_hperf.fastinc(depth, ngens < depth))
      running_hperf.report(inc_hperf, verbose) ;
   depth-- ;
   if (ngens >= depth) {
     if (is_node(n->nw)) {
       res = dorecurs(n->nw, n->ne, n->sw, n->se, depth) ;
     } else {
       res = (node *)dorecurs_leaf((leaf *)n->nw, (leaf *)n->ne,
                                   (leaf *)n->sw, (leaf *)n->se) ;
     }
   } else {
     if (is_node(n->nw)) {
       res = dorecurs_half(n->nw, n->ne, n->sw, n->se, depth) ;
     } else if (ngens == 0) {
       res = (node *)dorecurs_leaf_quarter((leaf *)n->nw, (leaf *)n->ne,
                                           (leaf *)n->sw, (leaf *)n->se) ;
     } else {
       res = (node *)dorecurs_leaf_half((leaf *)n->nw, (leaf *)n->ne,
                                        (leaf *)n->sw, (leaf *)n->se) ;
     }
   }
   pop(sp) ;
   if (softinterrupt ||
       poller->isInterrupted()) // don't assign this to the cache field!
     res = zeronode(depth) ;
   else {
     if (ngens < depth && halvesdone < 1000)
       halvesdone++ ;
     n->res = res ;
   }
   return res ;
}
#ifdef USEPREFETCH
void hlifealgo::setupprefetch(setup_t &su, node *nw, node *ne, node *sw, node *se) {
   su.h = node_hash(nw,ne,sw,se) ;
   su.nw = nw ;
   su.ne = ne ;
   su.sw = sw ;
   su.se = se ;
   PREFETCH(idx.e + hlslot(hlmix(su.h), idx.cap)) ;
}
node *hlifealgo::find_node(setup_t &su) {
   return find_node_h(su.h, su.nw, su.ne, su.sw, su.se) ;
}
node *hlifealgo::dorecurs(node *n, node *ne, node *t, node *e, int depth) {
   int sp = gsp ;
   setup_t su[5] ;
   setupprefetch(su[2], n->se, ne->sw, t->ne, e->nw) ;
   setupprefetch(su[0], n->ne, ne->nw, n->se, ne->sw) ;
   setupprefetch(su[1], ne->sw, ne->se, e->nw, e->ne) ;
   setupprefetch(su[3], n->sw, n->se, t->nw, t->ne) ;
   setupprefetch(su[4], t->ne, e->nw, t->se, e->sw) ;
   node
   *t00 = getres(n, depth),
   *t01 = getres(find_node(su[0]), depth),
   *t02 = getres(ne, depth),
   *t12 = getres(find_node(su[1]), depth),
   *t11 = getres(find_node(su[2]), depth),
   *t10 = getres(find_node(su[3]), depth),
   *t20 = getres(t, depth),
   *t21 = getres(find_node(su[4]), depth),
   *t22 = getres(e, depth) ;
   setupprefetch(su[0], t11, t12, t21, t22) ;
   setupprefetch(su[1], t10, t11, t20, t21) ;
   setupprefetch(su[2], t00, t01, t10, t11) ;
   setupprefetch(su[3], t01, t02, t11, t12) ;
   node
   *t44 = getres(find_node(su[0]), depth),
   *t43 = getres(find_node(su[1]), depth),
   *t33 = getres(find_node(su[2]), depth),
   *t34 = getres(find_node(su[3]), depth) ;
   n = find_node(t33, t34, t43, t44) ;
   pop(sp) ;
   return save(n) ;
}
#else
/*
 *   So let's say the cached way failed.  How do we do it the slow way?
 *   Recursively, of course.  For an n-square (composed of the four
 *   n/2-squares passed in, compute the n/2-square that is n/4
 *   generations ahead.
 *
 *   This routine works exactly the same as the leafres() routine, only
 *   instead of working on an 8-square, we're working on an n-square,
 *   returning an n/2-square, and we build that n/2-square by first building
 *   9 n/4-squares, use those to calculate 4 more n/4-squares, and
 *   then put these together into a new n/2-square.  Simple, eh?
 */
node *hlifealgo::dorecurs(node *n, node *ne, node *t, node *e, int depth) {
   int sp = gsp ;
   node
   *t11 = getres(find_node(n->se, ne->sw, t->ne, e->nw), depth),
   *t00 = getres(n, depth),
   *t01 = getres(find_node(n->ne, ne->nw, n->se, ne->sw), depth),
   *t02 = getres(ne, depth),
   *t12 = getres(find_node(ne->sw, ne->se, e->nw, e->ne), depth),
   *t10 = getres(find_node(n->sw, n->se, t->nw, t->ne), depth),
   *t20 = getres(t, depth),
   *t21 = getres(find_node(t->ne, e->nw, t->se, e->sw), depth),
   *t22 = getres(e, depth),
   *t44 = getres(find_node(t11, t12, t21, t22), depth),
   *t43 = getres(find_node(t10, t11, t20, t21), depth),
   *t33 = getres(find_node(t00, t01, t10, t11), depth),
   *t34 = getres(find_node(t01, t02, t11, t12), depth) ;
   n = find_node(t33, t34, t43, t44) ;
   pop(sp) ;
   return save(n) ;
}
#endif
/*
 *   Same as above, but we only do one step instead of 2.
 */
node *hlifealgo::dorecurs_half(node *n, node *ne, node *t,
                               node *e, int depth) {
   int sp = gsp ;
   node
   *t00 = getres(n, depth),
   *t01 = getres(find_node(n->ne, ne->nw, n->se, ne->sw), depth),
   *t10 = getres(find_node(n->sw, n->se, t->nw, t->ne), depth),
   *t11 = getres(find_node(n->se, ne->sw, t->ne, e->nw), depth),
   *t02 = getres(ne, depth),
   *t12 = getres(find_node(ne->sw, ne->se, e->nw, e->ne), depth),
   *t20 = getres(t, depth),
   *t21 = getres(find_node(t->ne, e->nw, t->se, e->sw), depth),
   *t22 = getres(e, depth) ;
   if (depth > 3) {
      n = find_node(find_node(t00->se, t01->sw, t10->ne, t11->nw),
                    find_node(t01->se, t02->sw, t11->ne, t12->nw),
                    find_node(t10->se, t11->sw, t20->ne, t21->nw),
                    find_node(t11->se, t12->sw, t21->ne, t22->nw)) ;
   } else {
      n = find_node((node *)find_leaf(((leaf *)t00)->se,
                                             ((leaf *)t01)->sw,
                                             ((leaf *)t10)->ne,
                                             ((leaf *)t11)->nw),
                    (node *)find_leaf(((leaf *)t01)->se,
                                             ((leaf *)t02)->sw,
                                             ((leaf *)t11)->ne,
                                             ((leaf *)t12)->nw),
                    (node *)find_leaf(((leaf *)t10)->se,
                                             ((leaf *)t11)->sw,
                                             ((leaf *)t20)->ne,
                                             ((leaf *)t21)->nw),
                    (node *)find_leaf(((leaf *)t11)->se,
                                             ((leaf *)t12)->sw,
                                             ((leaf *)t21)->ne,
                                             ((leaf *)t22)->nw)) ;
   }
   pop(sp) ;
   return save(n) ;
}
/*
 *   If the node is a 16-node, then the constituents are leaves, so we
 *   need a very similar but still somewhat different subroutine.  Since
 *   we do not (yet) garbage collect leaves, we don't need all that
 *   save/pop mumbo-jumbo.
 */
leaf *hlifealgo::dorecurs_leaf(leaf *n, leaf *ne, leaf *t, leaf *e) {
   if (leafnk >= 0) {
      unsigned short q[4] ;
      leafstep(n, ne, t, e, 4, q) ;
      return find_leaf(q[0], q[1], q[2], q[3]) ;
   }
   unsigned short
   t00 = n->res2,
   t01 = find_leaf(n->ne, ne->nw, n->se, ne->sw)->res2,
   t02 = ne->res2,
   t10 = find_leaf(n->sw, n->se, t->nw, t->ne)->res2,
   t11 = find_leaf(n->se, ne->sw, t->ne, e->nw)->res2,
   t12 = find_leaf(ne->sw, ne->se, e->nw, e->ne)->res2,
   t20 = t->res2,
   t21 = find_leaf(t->ne, e->nw, t->se, e->sw)->res2,
   t22 = e->res2 ;
   return find_leaf(find_leaf(t00, t01, t10, t11)->res2,
                    find_leaf(t01, t02, t11, t12)->res2,
                    find_leaf(t10, t11, t20, t21)->res2,
                    find_leaf(t11, t12, t21, t22)->res2) ;
}
/*
 *   Same as above but we only do two generations.
 */
#define combine4(t00,t01,t10,t11) (unsigned short)\
((((t00)<<10)&0xcc00)|(((t01)<<6)&0x3300)|(((t10)>>6)&0xcc)|(((t11)>>10)&0x33))
leaf *hlifealgo::dorecurs_leaf_half(leaf *n, leaf *ne, leaf *t, leaf *e) {
   if (leafnk >= 0) {
      unsigned short q[4] ;
      leafstep(n, ne, t, e, 2, q) ;
      return find_leaf(q[0], q[1], q[2], q[3]) ;
   }
   unsigned short
   t00 = n->res2,
   t01 = find_leaf(n->ne, ne->nw, n->se, ne->sw)->res2,
   t02 = ne->res2,
   t10 = find_leaf(n->sw, n->se, t->nw, t->ne)->res2,
   t11 = find_leaf(n->se, ne->sw, t->ne, e->nw)->res2,
   t12 = find_leaf(ne->sw, ne->se, e->nw, e->ne)->res2,
   t20 = t->res2,
   t21 = find_leaf(t->ne, e->nw, t->se, e->sw)->res2,
   t22 = e->res2 ;
   return find_leaf(combine4(t00, t01, t10, t11),
                    combine4(t01, t02, t11, t12),
                    combine4(t10, t11, t20, t21),
                    combine4(t11, t12, t21, t22)) ;
}
/*
 *   Same as above but we only do one generation.
 */
leaf *hlifealgo::dorecurs_leaf_quarter(leaf *n, leaf *ne,
                                   leaf *t, leaf *e) {
   if (leafnk >= 0) {
      unsigned short q[4] ;
      leafstep(n, ne, t, e, 1, q) ;
      return find_leaf(q[0], q[1], q[2], q[3]) ;
   }
   unsigned short
   t00 = n->res1,
   t01 = find_leaf(n->ne, ne->nw, n->se, ne->sw)->res1,
   t02 = ne->res1,
   t10 = find_leaf(n->sw, n->se, t->nw, t->ne)->res1,
   t11 = find_leaf(n->se, ne->sw, t->ne, e->nw)->res1,
   t12 = find_leaf(ne->sw, ne->se, e->nw, e->ne)->res1,
   t20 = t->res1,
   t21 = find_leaf(t->ne, e->nw, t->se, e->sw)->res1,
   t22 = e->res1 ;
   return find_leaf(combine4(t00, t01, t10, t11),
                    combine4(t01, t02, t11, t12),
                    combine4(t10, t11, t20, t21),
                    combine4(t11, t12, t21, t22)) ;
}
/*
 *   We keep free nodes in a linked list for allocation, and we allocate
 *   them 1360 at a time.  We used to do 1000, but this was interacting
 *   with certain (MacOS) memory allocators and instead of allocating the
 *   requested 48,000 bytes, they were allocating 65,536 bytes and
 *   wasting a good fraction of the memory available.  Using 1360 gets us
 *   closer to 65536, while still leaving some memory for any headers the
 *   allocator might need.
 */
static const int NODECHUNKCOUNT = 1360 ;
/*
 *   The blocks are cut from regions of HLREGION bytes, aligned to
 *   HLREGION, so the system can back them with 2 MB pages (hladvise).
 *   The first bytes of a region hold the previous region and the pointer
 *   to free.
 */
static const g_uintptr_t HLREGION = 2 << 20 ;
static const g_uintptr_t HLREGIONHEAD = 64 ;
node *hlifealgo::newblock() {
   g_uintptr_t bytes = (NODECHUNKCOUNT+1) * sizeof(node) ;
   if (regionnext == 0 || regionnext + bytes > regionend) {
      char *raw = (char *)malloc(2 * HLREGION) ;
      if (raw == 0)
         return 0 ;
      char *r = (char *)(((g_uintptr_t)raw + HLREGION - 1) & ~(HLREGION - 1)) ;
      hladvise(r, HLREGION) ;
      ((char **)r)[0] = regions ;
      ((char **)r)[1] = raw ;
      regions = r ;
      regionnext = r + HLREGIONHEAD ;
      regionend = r + HLREGION ;
   }
   node *b = (node *)regionnext ;
   regionnext += bytes ;
   memset(b, 0, bytes) ;
   return b ;
}
node *hlifealgo::newnode() {
   node *r ;
   if (freenodes == 0) {
      int i ;
      freenodes = newblock() ;
      if (freenodes == 0)
         lifefatal("Out of memory; try reducing the hash memory limit.") ;
      alloced += NODECHUNKCOUNT * sizeof(node) ;
      freenodes->next = nodeblocks ;
      freenodes->res = (struct node *)NODECHUNKCOUNT ;
      nodeblocks = freenodes++ ;
      for (i=0; i<(NODECHUNKCOUNT-1); i++) {
         freenodes[1].next = freenodes ;
         freenodes++ ;
      }
      totalthings += NODECHUNKCOUNT ;
   }
   if (freenodes->next == 0 && alloced + NODECHUNKCOUNT * sizeof(node) > maxmem &&
       okaytogc) {
      do_gc(0) ;
   }
   r = freenodes ;
   freenodes = freenodes->next ;
   return r ;
}
/*
 *   Leaves are the same.
 */
leaf *hlifealgo::newleaf() {
   leaf *r = (leaf *)newnode() ;
   new(&(r->leafpop))bigint ;
   return r ;
}
/*
 *   Sometimes we want the new node or leaf to be automatically cleared
 *   for us.
 */
node *hlifealgo::newclearednode() {
   return (node *)memset(newnode(), 0, sizeof(node)) ;
}
leaf *hlifealgo::newclearedleaf() {
   leaf *r = (leaf *)newclearednode() ;
   new(&(r->leafpop))bigint ;
   return r ;
}
hlifealgo::hlifealgo() {
   int i ;
   nthreads = numthreads ;
/*
 *   The population of one-bits in an integer is one more than the
 *   population of one-bits in the integer with one fewer bit set,
 *   and we can turn off a bit by anding an integer with the next
 *   lower integer.
 */
   if (shortpop[1] == 0)
      for (i=1; i<65536; i++)
         shortpop[i] = shortpop[i & (i - 1)] + 1 ;
   hashpop = 0 ;
   index_alloc(idx, INDEXMIN) ;
   alloced = INDEXMIN * sizeof(hlentry) ;
   ngens = 0 ;
   stacksize = 0 ;
   halvesdone = 0 ;
   nzeros = 0 ;
   stack = 0 ;
   gsp = 0 ;
   maxmem = 256 * 1024 * 1024 ;
   freenodes = 0 ;
   okaytogc = 0 ;
   totalthings = 0 ;
   nodeblocks = 0 ;
   regions = regionnext = regionend = 0 ;
   zeronodea = 0 ;
   ruletable = hliferules.rule0 ;
   setleafrule() ;
/*
 *   We initialize our universe to be a 16-square.  We are in drawing
 *   mode at this point.
 */
   root = (node *)newclearednode() ;
   population = 0 ;
   generation = 0 ;
   increment = 1 ;
   setincrement = 1 ;
   nonpow2 = 1 ;
   pow2step = 1 ;
   llsize = 0 ;
   depth = 3 ;
   hashed = 0 ;
   popValid = 0 ;
   needPop = 0 ;
   inGC = 0 ;
   cacheinvalid = 0 ;
   gccount = 0 ;
   gcstep = 0 ;
   running_hperf.clear() ;
   inc_hperf = running_hperf ;
   step_hperf = running_hperf ;
   softinterrupt = 0 ;
   par = 0 ;
}
/**
 *   Destructor frees memory.
 */
hlifealgo::~hlifealgo() {
   par_stop() ;
   free(idx.e) ;
   while (regions) {
      char *r = regions ;
      regions = ((char **)r)[0] ;
      free(((char **)r)[1]) ;
   }
   if (zeronodea)
      free(zeronodea) ;
   if (stack)
      free(stack) ;
   if (llsize) {
      delete [] llxb ;
      delete [] llyb ;
   }
}
/**
 *   Set increment.
 */
void hlifealgo::setIncrement(bigint inc) {
   if (inc < increment)
      softinterrupt = 1 ;
   increment = inc ;
}
/**
 *   Do a step.
 */
void hlifealgo::step() {
   poller->bailIfCalculating() ;
   // we use while here because the increment may be changed while we are
   // doing the hashtable sweep; if that happens, we may need to sweep
   // again.
   while (1) {
      int cleareddownto = 1000000000 ;
      softinterrupt = 0 ;
      while (increment != setincrement) {
         bigint pendingincrement = increment ;
         int newpow2 = 0 ;
         bigint t = pendingincrement ;
         while (t > 0 && t.even()) {
            newpow2++ ;
            t.div2() ;
         }
         nonpow2 = t.low31() ;
         if (t != nonpow2)
            lifefatal("bad increment") ;
         int downto = newpow2 ;
         if (ngens < newpow2)
            downto = ngens ;
         if (newpow2 != ngens && cleareddownto > downto) {
            new_ngens(newpow2) ;
            cleareddownto = downto ;
         } else {
            ngens = newpow2 ;
         }
         setincrement = pendingincrement ;
         pow2step = 1 ;
         while (newpow2--)
            pow2step += pow2step ;
      }
      gcstep = 0 ;
      running_hperf.genval = generation.todouble() ;
      for (int i=0; i<nonpow2; i++) {
         node *newroot = runpattern() ;
         if (newroot == 0 || softinterrupt || poller->isInterrupted()) // we *were* interrupted
            break ;
         popValid = 0 ;
         root = newroot ;
         depth = node_depth(root) ;
      }
      running_hperf.reportStep(step_hperf, inc_hperf, generation.todouble(), verbose) ;
      if (poller->isInterrupted() || !softinterrupt)
         break ;
   }
}
void hlifealgo::setcurrentstate(void *n) {
   if (root != (node *)n) {
      root = (node *)n ;
      depth = node_depth(root) ;
      popValid = 0 ;
   }
}
/*
 *   Set the max memory
 */
void hlifealgo::setMaxMemory(int newmemlimit) {
   if (newmemlimit < 10)
     newmemlimit = 10 ;
#ifndef GOLLY64BIT
   else if (newmemlimit > 4000)
     newmemlimit = 4000 ;
#endif
   g_uintptr_t newlimit = ((g_uintptr_t)newmemlimit) << 20 ;
   if (alloced > newlimit) {
      lifewarning("Sorry, more memory currently used than allowed.") ;
      return ;
   }
   maxmem = newlimit ;
}
/*
 *   This routine expands our universe by a factor of two, maintaining
 *   centering.  We use four new nodes, and *reuse* the root so this cannot
 *   be called after we've started hashing.
 */
void hlifealgo::pushroot_1() {
   node *t ;
   t = newclearednode() ;
   t->se = root->nw ;
   root->nw = t ;
   t = newclearednode() ;
   t->sw = root->ne ;
   root->ne = t ;
   t = newclearednode() ;
   t->ne = root->sw ;
   root->sw = t ;
   t = newclearednode() ;
   t->nw = root->se ;
   root->se = t ;
   depth++ ;
}
/*
 *   Return the depth of this node (2 is 8x8).
 */
int hlifealgo::node_depth(node *n) {
   int depth = 2 ;
   while (is_node(n)) {
      depth++ ;
      n = n->nw ;
   }
   return depth ;
}
/*
 *   This routine returns the canonical clear space node at a particular
 *   depth.
 */
node *hlifealgo::zeronode(int depth) {
   while (depth >= nzeros) {
      int nnzeros = 2 * nzeros + 10 ;
      zeronodea = (node **)realloc(zeronodea,
                                          nnzeros * sizeof(node *)) ;
      if (zeronodea == 0)
        lifefatal("Out of memory (2).") ;
      alloced += (nnzeros - nzeros) * sizeof(node *) ;
      while (nzeros < nnzeros)
         zeronodea[nzeros++] = 0 ;
   }
   if (zeronodea[depth] == 0) {
      if (depth == 2) {
         zeronodea[depth] = (node *)find_leaf(0, 0, 0, 0) ;
      } else {
         node *z = zeronode(depth-1) ;
         zeronodea[depth] = find_node(z, z, z, z) ;
      }
   }
   return zeronodea[depth] ;
}
/*
 *   Same, but with hashed nodes.
 */
node *hlifealgo::pushroot(node *n) {
   int depth = node_depth(n) ;
   zeronode(depth+1) ; // ensure enough zero nodes for rendering
   node *z = zeronode(depth-1) ;
   return find_node(find_node(z, z, z, n->nw),
                    find_node(z, z, n->ne, z),
                    find_node(z, n->sw, z, z),
                    find_node(n->se, z, z, z)) ;
}
/* Returns an internal (i.e. non-leaf) node containing the given node. */
node *hlifealgo::make_internal_node(node *n) {
   if (is_node(n)) return n ;
   leaf *l=(leaf *)n ;
   return find_node((node *)find_leaf(0, 0, 0, l->nw),
                    (node *)find_leaf(0, 0, l->ne, 0),
                    (node *)find_leaf(0, l->sw, 0, 0),
                    (node *)find_leaf(l->se, 0, 0, 0));
}
/*
 *   Here is our recursive routine to set a bit in our universe.  We
 *   pass in a depth, and walk the space.  Again, a lot of bit twiddling,
 *   but really not all that complicated.  We allocate new nodes and
 *   leaves on our way down.
 *
 *   Note that at this point our universe lives outside the hash table
 *   and has not been canonicalized, and that many of the pointers in
 *   the nodes can be null.  We'll patch this up in due course.
 */
node *hlifealgo::gsetbit(node *n, int x, int y, int newstate, int depth) {
   if (depth == 2) {
      leaf *l = (leaf *)n ;
      if (hashed) {
         unsigned short nw = l->nw ;
         unsigned short sw = l->sw ;
         unsigned short ne = l->ne ;
         unsigned short se = l->se ;
         if (newstate) {
            if (x < 0)
               if (y < 0)
                  sw |= 1 << (3 - (x & 3) + 4 * (y & 3)) ;
               else
                  nw |= 1 << (3 - (x & 3) + 4 * (y & 3)) ;
            else
               if (y < 0)
                  se |= 1 << (3 - (x & 3) + 4 * (y & 3)) ;
               else
                  ne |= 1 << (3 - (x & 3) + 4 * (y & 3)) ;
         } else {
            if (x < 0)
               if (y < 0)
                  sw &= ~(1 << (3 - (x & 3) + 4 * (y & 3))) ;
               else
                  nw &= ~(1 << (3 - (x & 3) + 4 * (y & 3))) ;
            else
               if (y < 0)
                  se &= ~(1 << (3 - (x & 3) + 4 * (y & 3))) ;
               else
                  ne &= ~(1 << (3 - (x & 3) + 4 * (y & 3))) ;
         }
         return save((node *)find_leaf(nw, ne, sw, se)) ;
      }
      if (newstate) {
         if (x < 0)
            if (y < 0)
               l->sw |= 1 << (3 - (x & 3) + 4 * (y & 3)) ;
            else
               l->nw |= 1 << (3 - (x & 3) + 4 * (y & 3)) ;
         else
            if (y < 0)
               l->se |= 1 << (3 - (x & 3) + 4 * (y & 3)) ;
            else
               l->ne |= 1 << (3 - (x & 3) + 4 * (y & 3)) ;
      } else {
         if (x < 0)
            if (y < 0)
               l->sw &= ~(1 << (3 - (x & 3) + 4 * (y & 3))) ;
            else
               l->nw &= ~(1 << (3 - (x & 3) + 4 * (y & 3))) ;
         else
            if (y < 0)
               l->se &= ~(1 << (3 - (x & 3) + 4 * (y & 3))) ;
            else
               l->ne &= ~(1 << (3 - (x & 3) + 4 * (y & 3))) ;
      }
      return (node *)l ;
   } else {
      unsigned int w = 0, wh = 0 ;
      if (depth >= 32) {
         if (depth == 32)
            wh = 0x80000000 ;
      } else {
         w = 1 << depth ;
         wh = 1 << (depth - 1) ;
      }
      depth-- ;
      node **nptr ;
      if (depth+1 == this->depth || depth < 31) {
         if (x < 0) {
            if (y < 0)
               nptr = &(n->sw) ;
            else
               nptr = &(n->nw) ;
         } else {
            if (y < 0)
               nptr = &(n->se) ;
            else
               nptr = &(n->ne) ;
         }
      } else {
         if (x >= 0) {
            if (y >= 0)
               nptr = &(n->sw) ;
            else
               nptr = &(n->nw) ;
         } else {
            if (y >= 0)
               nptr = &(n->se) ;
            else
               nptr = &(n->ne) ;
         }
      }
      if (*nptr == 0) {
         if (depth == 2)
            *nptr = (node *)newclearedleaf() ;
         else
            *nptr = newclearednode() ;
      }
      node *s = gsetbit(*nptr, (x & (w - 1)) - wh,
                               (y & (w - 1)) - wh, newstate, depth) ;
      if (hashed) {
         node *nw = (nptr == &(n->nw) ? s : n->nw) ;
         node *sw = (nptr == &(n->sw) ? s : n->sw) ;
         node *ne = (nptr == &(n->ne) ? s : n->ne) ;
         node *se = (nptr == &(n->se) ? s : n->se) ;
         n = save(find_node(nw, ne, sw, se)) ;
      } else {
         *nptr = s ;
      }
      return n ;
   }
}
/*
 *   Here is our recursive routine to get a bit in our universe.  We
 *   pass in a depth, and walk the space.  Again, a lot of bit twiddling,
 *   but really not all that complicated.
 */
int hlifealgo::getbit(node *n, int x, int y, int depth) {
   struct node tnode ;
   while (depth >= 32) {
      tnode.nw = n->nw->se ;
      tnode.ne = n->ne->sw ;
      tnode.sw = n->sw->ne ;
      tnode.se = n->se->nw ;
      n = &tnode ;
      depth-- ;
   }
   if (depth == 2) {
      leaf *l = (leaf *)n ;
      int test = 0 ;
      if (x < 0)
         if (y < 0)
            test = (l->sw & (1 << (3 - (x & 3) + 4 * (y & 3)))) ;
         else
            test = (l->nw & (1 << (3 - (x & 3) + 4 * (y & 3)))) ;
      else
         if (y < 0)
            test = (l->se & (1 << (3 - (x & 3) + 4 * (y & 3)))) ;
         else
            test = (l->ne & (1 << (3 - (x & 3) + 4 * (y & 3)))) ;
      if (test)
         return 1 ;
      return 0 ;
   } else {
      unsigned int w = 0, wh = 0 ;
      if (depth >= 32) {
         if (depth == 32)
            wh = 0x80000000 ;
      } else {
         w = 1 << depth ;
         wh = 1 << (depth - 1) ;
      }
      depth-- ;
      node *nptr ;
      if (x < 0) {
         if (y < 0)
            nptr = n->sw ;
         else
            nptr = n->nw ;
      } else {
         if (y < 0)
            nptr = n->se ;
         else
            nptr = n->ne ;
      }
      if (nptr == 0 || nptr == zeronode(depth))
         return 0 ;
      return getbit(nptr, (x & (w - 1)) - wh, (y & (w - 1)) - wh, depth) ;
   }
}
/*
 *   Here is our recursive routine to get the next bit in our universe.  We
 *   pass in a depth, and walk the space.  Again, a lot of bit twiddling,
 *   but really not all that complicated.
 */
int hlifealgo::nextbit(node *n, int x, int y, int depth) {
   if (n == 0 || n == zeronode(depth))
      return -1 ;
   if (depth == 2) {
      leaf *l = (leaf *)n ;
      int test = 0 ;
      if (y < 0)
        test = (((l->sw >> (4 * (y & 3))) & 15) << 4) |
                ((l->se >> (4 * (y & 3))) & 15) ;
      else
        test = (((l->nw >> (4 * (y & 3))) & 15) << 4) |
                ((l->ne >> (4 * (y & 3))) & 15) ;
      test &= (1 << (4 - x)) - 1 ;
      if (test) {
        int r = 0 ;
        int b = 1 << (3 - x) ;
        while ((test & b) == 0) {
          r++ ;
          b >>= 1 ;
        }
        return r ;
      }
      return -1 ; // none found
   } else {
      unsigned int w = 0, wh = 0 ;
      w = 1 << depth ;
      wh = 1 << (depth - 1) ;
      node *lft, *rght ;
      depth-- ;
      if (y < 0) {
        lft = n->sw ;
        rght = n->se ;
      } else {
        lft = n->nw ;
        rght = n->ne ;
      }
      int r = 0 ;
      if (x < 0) {
        int t = nextbit(lft, (x & (w-1)) - wh,
                        (y & (w - 1)) - wh, depth) ;
        if (t >= 0)
          return t ;
        r = -x ;
        x = 0 ;
      }
      int t = nextbit(rght, (x & (w-1)) - wh,
                      (y & (w - 1)) - wh, depth) ;
      if (t >= 0)
        return r + t ;
      return -1 ;
   }
}
/*
 *   Our nonrecurse top-level bit setting routine simply expands the
 *   universe as necessary to encompass the passed-in coordinates, and
 *   then invokes the recursive setbit.  Right now it works hashed or
 *   unhashed (but it's faster when unhashed).  We also turn on the inGC
 *   flag to inhibit popcount.
 */
int hlifealgo::setcell(int x, int y, int newstate) {
   if (newstate & ~1)
      return -1 ;
   if (hashed) {
      clearstack() ;
      save(root) ;
      okaytogc = 1 ;
   }
   inGC = 1 ;
   y = - y ;
   int sx = x ;
   int sy = y ;
   if (depth <= 31) {
     sx >>= depth ;
     sy >>= depth ;
   } else {
     sx >>= 31 ;
     sy >>= 31 ;
   }
   while (sx > 0 || sx < -1 || sy > 0 || sy < -1) {
      if (hashed) {
         root = save(pushroot(root)) ;
         depth++ ;
      } else {
         pushroot_1() ;
      }
      sx >>= 1 ;
      sy >>= 1 ;
   }
   root = gsetbit(root, x, y, newstate, depth) ;
   if (hashed) {
      okaytogc = 0 ;
   }
   return 0 ;
}
/*
 *   Our nonrecurse top-level bit getting routine.
 */
int hlifealgo::getcell(int x, int y) {
   y = - y ;
   int sx = x ;
   int sy = y ;
   if (depth <= 31) {
     sx >>= depth ;
     sy >>= depth ;
   } else {
     sx >>= 31 ;
     sy >>= 31 ;
   }
   if (sx > 0 || sx < -1 || sy > 0 || sy < -1)
      return 0 ;
   return getbit(root, x, y, depth) ;
}
/*
 *   A recursive bit getting routine, but this one returns the
 *   number of pixels to the right to the next set cell in the
 *   current universe, or -1 if none set to the right, or if
 *   the next set pixel is out of range.
 */
int hlifealgo::nextcell(int x, int y, int &v) {
   v = 1 ;
   y = - y ;
   int sx = x ;
   int sy = y ;
   if (depth <= 31) {
     sx >>= depth ;
     sy >>= depth ;
   } else {
     sx >>= 31 ;
     sy >>= 31 ;
   }
   while (sx > 0 || sx < -1 || sy > 0 || sy < -1) {
      if (hashed) {
         root = save(pushroot(root)) ;
         depth++ ;
      } else {
         pushroot_1() ;
      }
      sx >>= 1 ;
      sy >>= 1 ;
   }
   if (depth > 30) {
      struct node tnode = *root ;
      int mdepth = depth ;
      while (mdepth > 30) {
         tnode.nw = tnode.nw->se ;
         tnode.ne = tnode.ne->sw ;
         tnode.sw = tnode.sw->ne ;
         tnode.se = tnode.se->nw ;
         mdepth-- ;
      }
      return nextbit(&tnode, x, y, mdepth) ;
   }
   return nextbit(root, x, y, depth) ;
}
/*
 *   Canonicalize a universe by filling in the null pointers and then
 *   invoking find_node on each node.  Drops the original universe on
 *   the floor [big deal, it's probably small anyway].
 */
node *hlifealgo::hashpattern(node *root, int depth) {
   node *r ;
   if (root == 0) {
      r = zeronode(depth) ;
   } else if (depth == 2) {
      leaf *n = (leaf *)root ;
      r = (node *)find_leaf(n->nw, n->ne, n->sw, n->se) ;
      n->next = freenodes ;
      freenodes = root ;
   } else {
      depth-- ;
      r = find_node(hashpattern(root->nw, depth),
                    hashpattern(root->ne, depth),
                    hashpattern(root->sw, depth),
                    hashpattern(root->se, depth)) ;
      root->next = freenodes ;
      freenodes = root ;
   }
   return r ;
}
void hlifealgo::endofpattern() {
   poller->bailIfCalculating() ;
   if (!hashed) {
      root = hashpattern(root, depth) ;
      zeronode(depth) ;
      hashed = 1 ;
   }
   popValid = 0 ;
   needPop = 0 ;
   inGC = 0 ;
}
void hlifealgo::ensure_hashed() {
   if (!hashed)
      endofpattern() ;
}
/*
 *   Pop off any levels we don't need.
 */
node *hlifealgo::popzeros(node *n) {
   int depth = node_depth(n) ;
   while (depth > 3) {
      node *z = zeronode(depth-2) ;
      if (n->nw->nw == z && n->nw->ne == z && n->nw->sw == z &&
          n->ne->nw == z && n->ne->ne == z && n->ne->se == z &&
          n->sw->nw == z && n->sw->sw == z && n->sw->se == z &&
          n->se->ne == z && n->se->sw == z && n->se->se == z) {
         depth-- ;
         n = find_node(n->nw->se, n->ne->sw, n->sw->ne, n->se->nw) ;
      } else {
         break ;
      }
   }
   return n ;
}
/*
 *   A lot of the routines from here on down traverse the universe, hanging
 *   information off the nodes.  The way they generally do so is by using
 *   (or abusing) the cache (res) field, and the least significant bit of
 *   the hash next field (as a visited bit).
 */
#define marked(n) (1 & (g_uintptr_t)(n)->next)
#define mark(n) ((n)->next = (node *)(1 | (g_uintptr_t)(n)->next))
#define clearmark(n) ((n)->next = (node *)(~1 & (g_uintptr_t)(n)->next))
#define clearmarkbit(p) ((node *)(~1 & (g_uintptr_t)(p)))
/*
 *   Sometimes we want to use *res* instead of next to mark.  You cannot
 *   do this to leaves, though.
 */
#define marked2(n) (3 & (g_uintptr_t)(n)->res)
#define mark2(n) ((n)->res = (node *)(1 | (g_uintptr_t)(n)->res))
#define mark2v(n,v) ((n)->res = (node *)(v | (g_uintptr_t)(n)->res))
#define clearmark2(n) ((n)->res = (node *)(~3 & (g_uintptr_t)(n)->res))
/*
 *   calcpop() and writecell() use the next field of nodes for their own
 *   data; nothing else uses it between steps except the gc mark bit.
 */
void hlifealgo::unhash_node(node *) {
}
void hlifealgo::unhash_node2(node *) {
}
void hlifealgo::rehash_node(node *n) {
   n->next = 0 ;
}
/*
 *   This recursive routine calculates the population by hanging the
 *   population on marked nodes.
 */
const bigint &hlifealgo::calcpop(node *root, int depth) {
   if (root == zeronode(depth))
      return bigint::zero ;
   if (depth == 2)
      return ((leaf *)root)->leafpop ;
   if (marked2(root))
      return *(bigint*)&(root->next) ;
   depth-- ;
   if (root->next == 0)
      mark2v(root, 3) ;
   else {
      unhash_node(root) ;
      mark2(root) ;
   }
/**
 *   We use allocate-in-place bigint constructor here to initialize the
 *   node.  This should compile to a single instruction.
 */
   new(&(root->next))bigint(
        calcpop(root->nw, depth), calcpop(root->ne, depth),
        calcpop(root->sw, depth), calcpop(root->se, depth)) ;
   return *(bigint *)&(root->next) ;
}
/*
 *   Call this after doing something that unhashes nodes in order to
 *   use the next field as a temp pointer.
 */
void hlifealgo::aftercalcpop2(node *root, int depth) {
   if (depth == 2 || root == zeronode(depth))
      return ;
   int v = marked2(root) ;
   if (v) {
      clearmark2(root) ;
      depth-- ;
      if (depth > 2) {
         aftercalcpop2(root->nw, depth) ;
         aftercalcpop2(root->ne, depth) ;
         aftercalcpop2(root->sw, depth) ;
         aftercalcpop2(root->se, depth) ;
      }
      ((bigint *)&(root->next))->~bigint() ;
      if (v == 3)
         root->next = 0 ;
      else
         rehash_node(root) ;
   }
}
/*
 *   Call this after writing macrocell.
 */
void hlifealgo::afterwritemc(node *root, int depth) {
   if (root == zeronode(depth))
      return ;
   if (depth == 2) {
      ((leaf *)root)->wnum = 0 ;
      return ;
   }
   if (marked2(root)) {
      clearmark2(root) ;
      depth-- ;
      afterwritemc(root->nw, depth) ;
      afterwritemc(root->ne, depth) ;
      afterwritemc(root->sw, depth) ;
      afterwritemc(root->se, depth) ;
      rehash_node(root) ;
   }
}
/*
 *   This top level routine calculates the population of a universe.
 */
void hlifealgo::calcPopulation() {
   int depth ;
   ensure_hashed() ;
   depth = node_depth(root) ;
   population = calcpop(root, depth) ;
   aftercalcpop2(root, depth) ;
}
/*
 *   Is the universe empty?
 */
int hlifealgo::isEmpty() {
   ensure_hashed() ;
   return root == zeronode(depth) ;
}
/*
 *   This routine marks a node as needed to be saved.
 */
node *hlifealgo::save(node *n) {
   if (gsp >= stacksize)
      return savegrow(n) ;
   stack[gsp++] = n ;
   return n ;
}
// not inlined into save(), so the callers of save() need no saved registers
__attribute__((noinline)) node *hlifealgo::savegrow(node *n) {
   int nstacksize = stacksize * 2 + 100 ;
   alloced += sizeof(node *)*(nstacksize-stacksize) ;
   stack = (node **)realloc(stack, nstacksize * sizeof(node *)) ;
   if (stack == 0)
     lifefatal("Out of memory (3).") ;
   stacksize = nstacksize ;
   stack[gsp++] = n ;
   return n ;
}
/*
 *   This routine pops the stack back to a previous depth.
 */
void hlifealgo::pop(int n) {
   gsp = n ;
}
/*
 *   This routine clears the stack altogether.
 */
void hlifealgo::clearstack() {
   gsp = 0 ;
}
/*
 *   Do a gc.  Walk down from all nodes reachable on the stack, saveing
 *   them by setting the odd bit on the next link.  Then, walk the hash,
 *   eliminating the res from everything that's not saveed, and moving
 *   the nodes from the hash to the freelist as appropriate.  Finally,
 *   walk the hash again, clearing the low order bits in the next pointers.
 */
void hlifealgo::gc_mark(node *root, int invalidate) {
   if (!marked(root)) {
      mark(root) ;
      if (is_node(root)) {
         gc_mark(root->nw, invalidate) ;
         gc_mark(root->ne, invalidate) ;
         gc_mark(root->sw, invalidate) ;
         gc_mark(root->se, invalidate) ;
         if (root->res && !ISPARBUSY(root->res)) {
            if (invalidate)
              root->res = 0 ;
            else
              gc_mark(root->res, invalidate) ;
         }
      }
   }
}
/**
 *   If the invalidate flag is set, we want to kill *all* cache entries
 *   and recalculate all leaves.
 */
void hlifealgo::do_gc(int invalidate) {
   int i ;
   g_uintptr_t freed_nodes=0 ;
   inGC = 1 ;
   gccount++ ;
   gcstep++ ;
   if (verbose) {
     if (gcstep > 1)
       sprintf(statusline, "GC #%d(%d)", gccount, gcstep) ;
     else
       sprintf(statusline, "GC #%d", gccount) ;
     lifestatus(statusline) ;
   }
   for (i=nzeros-1; i>=0; i--)
      if (zeronodea[i] != 0)
         break ;
   if (i >= 0)
      gc_mark(zeronodea[i], 0) ; // never invalidate zeronode
   if (root != 0)
      gc_mark(root, invalidate) ; // pick up the root
   for (i=0; i<gsp; i++) {
      poller->poll() ;
      gc_mark(stack[i], invalidate) ;
   }
   for (i=0; i<timeline.framecount; i++)
      gc_mark((node *)timeline.frames[i], invalidate) ;
   if (par)
      par_gc_roots(invalidate) ;
   // the marked nodes go back into the cleared index, the others (with
   // the threads' free lists) onto the free list
   memset(idx.e, 0, idx.cap * sizeof(hlentry)) ;
   idx.used = 0 ;
   g_uintptr_t oldpop = hashpop ;
   hashpop = 0 ;
   freenodes = 0 ;
   for (node *b=nodeblocks; b; b=b->next) {
      poller->poll() ;
      node *pp = b + 1 ;
      for (g_uintptr_t j=0; j<(g_uintptr_t)b->res; j++, pp++) {
         if (marked(pp)) {
            clearmark(pp) ;
            if (invalidate && !is_node(pp))
               leafres((leaf *)pp) ;
            index_put(idx, pp) ;
            hashpop++ ;
         } else {
            pp->next = freenodes ;
            freenodes = pp ;
         }
      }
   }
   freed_nodes = oldpop > hashpop ? oldpop - hashpop : 0 ;
   inGC = 0 ;
   if (verbose) {
     double perc = (double)freed_nodes / (double)(hashpop + freed_nodes) * 100.0 ;
     sprintf(statusline+strlen(statusline), " freed %g percent (%" PRIuPTR ").",
                                                   perc, freed_nodes) ;
     lifestatus(statusline) ;
   }
   if (needPop) {
      calcPopulation() ;
      popValid = 1 ;
      needPop = 0 ;
      poller->updatePop() ;
   }
}
/*
 *   Clear the cache bits down to the appropriate level, marking the
 *   nodes we've handled.
 */
void hlifealgo::clearcache(node *n, int depth, int clearto) {
   if (!marked(n)) {
      mark(n) ;
      if (depth > 3) {
         depth-- ;
         poller->poll() ;
         clearcache(n->nw, depth, clearto) ;
         clearcache(n->ne, depth, clearto) ;
         clearcache(n->sw, depth, clearto) ;
         clearcache(n->se, depth, clearto) ;
         if (n->res)
            clearcache(n->res, depth, clearto) ;
      }
      if (depth >= clearto)
         n->res = 0 ;
   }
}
/*
 *   Clear the entire cache of everything, and recalculate all leaves.
 *   This can be very expensive.
 */
void hlifealgo::clearcache() {
   cacheinvalid = 1 ;
}
/*
 *   Change the ngens value.  Requires us to walk the hash, clearing
 *   the cache fields of any nodes that do not have the appropriate
 *   values.
 */
void hlifealgo::new_ngens(int newval) {
   int clearto = ngens ;
   if (newval > ngens && halvesdone == 0) {
      ngens = newval ;
      return ;
   }
#ifndef NOGCBEFOREINC
   do_gc(0) ;
#endif
   if (verbose) {
     strcpy(statusline, "Changing increment...") ;
     lifestatus(statusline) ;
   }
   if (newval < clearto)
      clearto = newval ;
   clearto++ ; /* clear this depth and above */
   if (clearto < 3)
      clearto = 3 ;
   ngens = newval ;
   inGC = 1 ;
   for (g_uintptr_t j=0; j<idx.cap; j++)
      if (idx.e[j]) {
         node *p = hlptr(idx.e[j]) ;
         if (is_node(p) && !marked(p))
            clearcache(p, node_depth(p), clearto) ;
      }
   poller->poll() ;
   for (g_uintptr_t j=0; j<idx.cap; j++)
      if (idx.e[j])
         clearmark(hlptr(idx.e[j])) ;
   halvesdone = 0 ;
   inGC = 0 ;
   if (needPop) {
      calcPopulation() ;
      popValid = 1 ;
      needPop = 0 ;
      poller->updatePop() ;
   }
   if (verbose) {
     strcpy(statusline+strlen(statusline), " done.") ;
     lifestatus(statusline) ;
   }
}
/*
 *   Return log2.
 */
int hlifealgo::log2(unsigned int n) {
   int r = 0 ;
   while ((n & 1) == 0) {
      n >>= 1 ;
      r++ ;
   }
   if (n != 1) {
      lifefatal("Expected power of two!") ;
   }
   return r ;
}
static bigint negone = -1 ;
const bigint &hlifealgo::getPopulation() {
   // note:  if called during gc, then we cannot call calcPopulation
   // since that will mess up the gc.
   if (!popValid) {
      if (inGC) {
        needPop = 1 ;
        return negone ;
      } else if (poller->isCalculating()) {
        // AKT: avoid calling poller->bailIfCalculating
        return negone ;
      } else {
        calcPopulation() ;
        popValid = 1 ;
        needPop = 0 ;
      }
   }
   return population ;
}
/*
 *   Finally, we get to run the pattern.  We first ensure that all
 *   clearspace nodes and the input pattern is never garbage
 *   collected; we turn on garbage collection, and then we invoke our
 *   magic top-level routine passing in clearspace borders that are
 *   guaranteed large enough.
 */
node *hlifealgo::runpattern() {
   node *n = root ;
   save(root) ; // do this in case we interrupt generation
   ensure_hashed() ;
   okaytogc = 1 ;
   if (cacheinvalid) {
      do_gc(1) ; // invalidate the entire cache and recalc leaves
      cacheinvalid = 0 ;
   }
   int depth = node_depth(n) ;
   node *n2 ;
   n = pushroot(n) ;
   depth++ ;
   n = pushroot(n) ;
   depth++ ;
   while (ngens + 2 > depth) {
      n = pushroot(n) ;
      depth++ ;
   }
   save(zeronode(nzeros-1)) ;
   save(n) ;
   if (nthreads > 0)
      n2 = runpattern_par(n, depth) ;
   else
      n2 = getres(n, depth) ;
   okaytogc = 0 ;
   clearstack() ;
   if (halvesdone == 1 && n->res != 0) {
      n->res = 0 ;
      halvesdone = 0 ;
   }
   if (poller->isInterrupted() || softinterrupt)
      return 0 ; // indicate it was interrupted
   n = popzeros(n2) ;
   generation += pow2step ;
   return n ;
}
/* Returns the center 4-square of an 8x8 leaf node. */
static unsigned short unpack4x4center(leaf *leaf) {
   return combine4(leaf->nw, leaf->ne, leaf->sw, leaf->se);
}
const char *hlifealgo::readmacrocell(char *line) {
   int n=0 ;
   g_uintptr_t i=1, nw=0, ne=0, sw=0, se=0, indlen=0 ;
   int r, d ;
   node **ind = 0 ;
   root = 0 ;
   while (getline(line, 10000)) {
      if (i >= indlen) {
         g_uintptr_t nlen = i + indlen + 10 ;
         ind = (node **)realloc(ind, sizeof(node*) * nlen) ;
         if (ind == 0)
           lifefatal("Out of memory (4).") ;
         while (indlen < nlen)
            ind[indlen++] = 0 ;
      }
      if (line[0] == '.' || line[0] == '*' || line[0] == '$') {
         int x=0, y=7 ;
         unsigned short lnw=0, lne=0, lsw=0, lse=0 ;
         char *p = 0 ;
         for (p=line; *p > ' '; p++) {
            switch(*p) {
case '*':      if (x > 7 || y < 0)
                  return "Illegal coordinates in readmacrocell." ;
               if (x < 4)
                  if (y < 4)
                     lsw |= 1 << (3 - (x & 3) + 4 * (y & 3)) ;
                  else
                     lnw |= 1 << (3 - (x & 3) + 4 * (y & 3)) ;
               else
                  if (y < 4)
                     lse |= 1 << (3 - (x & 3) + 4 * (y & 3)) ;
                  else
                     lne |= 1 << (3 - (x & 3) + 4 * (y & 3)) ;
               /* note: fall through here */
case '.':      x++ ;
               break ;
case '$':      x = 0 ;
               y-- ;
               break ;
default:       return "Illegal character in readmacrocell." ;
            }
         }
         clearstack() ;
         root = ind[i++] = (node *)find_leaf(lnw, lne, lsw, lse) ;
         depth = 2;
      } else if (line[0] == '#') {
         char *p, *pp ;
         const char *err ;
         switch (line[1]) {
         case 'R':
            p = line + 2 ;
            while (*p && *p <= ' ') p++ ;
            pp = p ;
            while (*pp > ' ') pp++ ;
            *pp = 0 ;
            
            // AKT: need to check for B0-not-Smax rule
            err = setrule(p);
            if (err)
               return err;
            if (hliferules.alternate_rules)
               return "B0-not-Smax rules are not allowed in HashLife.";
            
            break ;
         case 'G':
            p = line + 2 ;
            while (*p && *p <= ' ') p++ ;
            pp = p ;
            while (*pp >= '0' && *pp <= '9') pp++ ;
            *pp = 0 ;
            generation = bigint(p) ;
            break ;
	    // either:
	    //   #FRAMES count base inc
	    // or
	    //   #FRAME index node
	 case 'F':
	    if (strncmp(line, "#FRAMES ", 8) == 0) {
	       p = line + 8 ;
	       while (*p && *p <= ' ')
		 p++ ;
	       long cnt = atol(p) ;
	       if (cnt < 0 || cnt > MAX_FRAME_COUNT)
		  return "Bad FRAMES line" ;
	       destroytimeline() ;
	       while ('0' <= *p && *p <= '9')
		 p++ ;
	       while (*p && *p <= ' ')
		 p++ ;
	       pp = p ;
	       while ((*pp >= '0' && *pp <= '9') || *pp == ',') pp++ ;
	       if (*pp == 0)
		  return "Bad FRAMES line" ;
	       *pp = 0 ;
	       timeline.start = bigint(p) ;
	       timeline.end = timeline.start ;
               timeline.next = timeline.start ;
	       p = pp + 1 ;
	       while (*p && *p <= ' ')
		 p++ ;
	       pp = p ;
	       while (*pp > ' ')
                  pp++ ;
	       *pp = 0 ;
               if (strchr(p, '^')) {
                  int tbase=0, texpo=0 ;
                  if (sscanf(p, "%d^%d", &tbase, &texpo) != 2 ||
                      tbase < 2 || texpo < 0)
                     return "Bad FRAMES line" ;
                  timeline.base = tbase ;
                  timeline.expo = texpo ;
                  timeline.inc = 1 ;
                  while (texpo--)
                     timeline.inc.mul_smallint(tbase) ;
               } else {
	          timeline.inc = bigint(p) ;
                  // if it's a power of two, we're good
                  int texpo = timeline.inc.lowbitset() ;
                  int tbase = 2 ;
                  bigint test = 1 ;
                  for (int i=0; i<texpo; i++)
                     test += test ;
                  if (test != timeline.inc)
                     return "Bad increment (missing ^) in FRAMES" ;
                  timeline.base = tbase ;
                  timeline.expo = texpo ;
               }
	    } else if (strncmp(line, "#FRAME ", 7) == 0) {
	       int frameind = 0 ;
	       g_uintptr_t nodeind = 0 ;
	       n = sscanf(line+7, "%d %" PRIuPTR, &frameind, &nodeind) ;
	       if (n != 2 || frameind > MAX_FRAME_COUNT || frameind < 0 ||
		   nodeind > i || timeline.framecount != frameind)
		  return "Bad FRAME line" ;
	       timeline.frames.push_back(make_internal_node(ind[nodeind])) ;
	       timeline.framecount++ ;
	       timeline.end = timeline.next ;
	       timeline.next += timeline.inc ;
	    }
	    break ;
         }
      } else {
         n = sscanf(line, "%d %" PRIuPTR " %" PRIuPTR " %" PRIuPTR " %" PRIuPTR " %d", &d, &nw, &ne, &sw, &se, &r) ;
         if (n < 0) // blank line; permit
            continue ;
         if (n == 0) {
            // conversion error in first argument; we allow only if the only
            // content on the line is whitespace.
            char *ws = line ;
            while (*ws && *ws <= ' ')
               ws++ ;
            if (*ws > 0)
               return "Parse error in macrocell format." ;
            continue ;
         }
         if (n < 5)
            // AKT: best not to use lifefatal here because user won't see any
            // error message when reading clipboard data starting with "[..."
            return "Parse error in readmacrocell." ;
         if (d < 1)
            return "Oops; bad depth in readmacrocell." ;
         if (d > 1) {
            ind[0] = zeronode(d <= 4 ? 2 : d-2) ; /* allow zeros to work right */
            if (nw >= i || ind[nw] == 0 || ne >= i || ind[ne] == 0 ||
               sw >= i || ind[sw] == 0 || se >= i || ind[se] == 0) {
               return "Node out of range in readmacrocell." ;
            }
         }
         if (d < 4) {
            /* Support macrocell nodes in multicell format (i.e. with 2-square
               leaf nodes) for compatibility. Cell values must be 0 or 1! */
            unsigned short lnw=0, lne=0, lsw=0, lse=0 ;
            if (d == 1) {
               if (nw > 1 || ne > 1 || sw > 1 || se > 1) {
                  return "Cell value out of range in readmacrocell." ;
               }
               lnw = nw ? 1 <<  0 : 0 ;
               lne = ne ? 1 <<  3 : 0 ;
               lsw = sw ? 1 << 12 : 0 ;
               lse = se ? 1 << 15 : 0 ;
            } else { // d == 2 || d == 3
               node *pnw=ind[nw], *pne=ind[ne], *psw=ind[sw], *pse=ind[se] ;
               if (is_node(pnw) || is_node(pne) || is_node(psw) || is_node(pse)) {
                  return "Invalid leaf node reference in readmacrocell." ;
               }
               lnw = unpack4x4center((leaf *)pnw) ;
               lne = unpack4x4center((leaf *)pne) ;
               lsw = unpack4x4center((leaf *)psw) ;
               lse = unpack4x4center((leaf *)pse) ;
               if (d == 2) {
                  lnw >>= 5 ;
                  lne >>= 3 ;
                  lsw <<= 3 ;
                  lse <<= 5 ;
               }
            }
            clearstack() ;
            root = ind[i++] = (node *)find_leaf(lnw, lne, lsw, lse) ;
         } else {  // d >= 4
            clearstack() ;
            root = ind[i++] = find_node(ind[nw], ind[ne], ind[sw], ind[se]) ;
         }
         depth = d - 1 ;
      }
   }
   if (ind)
      free(ind) ;
   if (root == 0) {
      // AKT: allow empty macrocell pattern; note that endofpattern()
      // will be called soon so don't set hashed here
      // return "Invalid macrocell file: no nodes." ;
      return 0 ;
   }
   if (depth < 3) {
      root = make_internal_node(root) ;
      depth = 3;
   }
   hashed = 1 ;
   return 0 ;
}

// Flip bits in given rule table.
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

const char *hlifealgo::setrule(const char *s) {
   poller->bailIfCalculating() ;
   const char* err = hliferules.setrule(s, this);
   if (err) return err;

   // invert orientation if not hex or Wolfram
   if (!(hliferules.isHexagonal() || hliferules.isWolfram())) {
      fliprule(hliferules.rule0);
   }
   setleafrule() ;

   clearcache() ;
   
   if (hliferules.alternate_rules)
      return "B0-not-Smax rules are not allowed in HashLife.";
      
   if (hliferules.isHexagonal())
      grid_type = HEX_GRID;
   else if (hliferules.isVonNeumann())
      grid_type = VN_GRID;
   else
      grid_type = SQUARE_GRID;
      
   return 0 ;
}
void hlifealgo::unpack8x8(unsigned short nw, unsigned short ne,
                          unsigned short sw, unsigned short se,
                          unsigned int *top, unsigned int *bot) {
   *top = ((nw & 0xf000) << 16) | (((ne & 0xf000) | (nw & 0xf00)) << 12) |
          (((ne & 0xf00) | (nw & 0xf0)) << 8) |
          (((ne & 0xf0) | (nw & 0xf)) << 4) | (ne & 0xf) ;
   *bot = ((sw & 0xf000) << 16) | (((se & 0xf000) | (sw & 0xf00)) << 12) |
          (((se & 0xf00) | (sw & 0xf0)) << 8) |
          (((se & 0xf0) | (sw & 0xf)) << 4) | (se & 0xf) ;
}
/**
 *   Write out the native macrocell format.  This is the one we use when
 *   we're not interactive and displaying a progress dialog.
 */
g_uintptr_t hlifealgo::writecell(std::ostream &os, node *root, int depth) {
   g_uintptr_t thiscell = 0 ;
   if (root == zeronode(depth))
      return 0 ;
   if (depth == 2) {
      if (((leaf *)root)->wnum != 0)
         return ((leaf *)root)->wnum ;
   } else {
      if (marked2(root))
         return (g_uintptr_t)(root->next) ;
      unhash_node2(root) ;
      mark2(root) ;
   }
   if (depth == 2) {
      int i, j ;
      unsigned int top, bot ;
      leaf *n = (leaf *)root ;
      thiscell = ++cellcounter ;
      ((leaf *)root)->wnum = thiscell ;
      unpack8x8(n->nw, n->ne, n->sw, n->se, &top, &bot) ;
      for (j=7; (top | bot) && j>=0; j--) {
         int bits = (top >> 24) ;
         top = (top << 8) | (bot >> 24) ;
         bot = (bot << 8) ;
         for (i=0; bits && i<8; i++, bits = (bits << 1) & 255)
            if (bits & 128)
               os << '*' ;
            else
               os << '.' ;
         os << '$' ;
      }
      os << '\n' ;
   } else {
      g_uintptr_t nw = writecell(os, root->nw, depth-1) ;
      g_uintptr_t ne = writecell(os, root->ne, depth-1) ;
      g_uintptr_t sw = writecell(os, root->sw, depth-1) ;
      g_uintptr_t se = writecell(os, root->se, depth-1) ;
      thiscell = ++cellcounter ;
      root->next = (node *)thiscell ;
      os << depth+1 << ' ' << nw << ' ' << ne << ' ' << sw << ' ' << se << '\n';
   }
   return thiscell ;
}
/**
 *   This new two-pass method works by doing a prepass that numbers the
 *   nodes and counts the number of nodes that should be sent, so we can
 *   display an accurate progress dialog.
 */
g_uintptr_t hlifealgo::writecell_2p1(node *root, int depth) {
   g_uintptr_t thiscell = 0 ;
   if (root == zeronode(depth))
      return 0 ;
   if (depth == 2) {
      if (((leaf *)root)->wnum != 0)
         return ((leaf *)root)->wnum ;
   } else {
      if (marked2(root))
         return (g_uintptr_t)(root->next) ;
      unhash_node2(root) ;
      mark2(root) ;
   }
   if (depth == 2) {
      thiscell = ++cellcounter ;
      // note:  we *must* not abort this prescan
      if ((cellcounter & 4095) == 0)
         lifeabortprogress(0, "Scanning tree") ;
      ((leaf *)root)->wnum = thiscell ;
   } else {
      writecell_2p1(root->nw, depth-1) ;
      writecell_2p1(root->ne, depth-1) ;
      writecell_2p1(root->sw, depth-1) ;
      writecell_2p1(root->se, depth-1) ;
      thiscell = ++cellcounter ;
      // note:  we *must* not abort this prescan
      if ((cellcounter & 4095) == 0)
         lifeabortprogress(0, "Scanning tree") ;
      root->next = (node *)thiscell ;
   }
   return thiscell ;
}
/**
 *   This one writes the cells, but assuming they've already been
 *   numbered, and displaying a progress dialog.
 */
static char progressmsg[80] ;
g_uintptr_t hlifealgo::writecell_2p2(std::ostream &os, node *root, int depth) {
   g_uintptr_t thiscell = 0 ;
   if (root == zeronode(depth))
      return 0 ;
   if (depth == 2) {
      if (cellcounter + 1 != ((leaf *)root)->wnum)
         return ((leaf *)root)->wnum ;
      thiscell = ++cellcounter ;
      if ((cellcounter & 4095) == 0) {
         std::streampos siz = os.tellp();
         sprintf(progressmsg, "File size: %.2f MB", double(siz) / 1048576.0) ;
         lifeabortprogress(thiscell/(double)writecells, progressmsg) ;
      }
      int i, j ;
      unsigned int top, bot ;
      leaf *n = (leaf *)root ;
      ((leaf *)root)->wnum = thiscell ;
      unpack8x8(n->nw, n->ne, n->sw, n->se, &top, &bot) ;
      for (j=7; (top | bot) && j>=0; j--) {
         int bits = (top >> 24) ;
         top = (top << 8) | (bot >> 24) ;
         bot = (bot << 8) ;
         for (i=0; bits && i<8; i++, bits = (bits << 1) & 255)
            if (bits & 128)
               os << '*' ;
            else
               os << '.' ;
         os << '$' ;
      }
      os << '\n' ;
   } else {
      if (cellcounter + 1 > (g_uintptr_t)(root->next) || isaborted())
         return (g_uintptr_t)(root->next) ;
      g_uintptr_t nw = writecell_2p2(os, root->nw, depth-1) ;
      g_uintptr_t ne = writecell_2p2(os, root->ne, depth-1) ;
      g_uintptr_t sw = writecell_2p2(os, root->sw, depth-1) ;
      g_uintptr_t se = writecell_2p2(os, root->se, depth-1) ;
      if (!isaborted() &&
          cellcounter + 1 != (g_uintptr_t)(root->next)) { // this should never happen
         lifefatal("Internal in writecell_2p2") ;
         return (g_uintptr_t)(root->next) ;
      }
      thiscell = ++cellcounter ;
      if ((cellcounter & 4095) == 0) {
         std::streampos siz = os.tellp();
         sprintf(progressmsg, "File size: %.2f MB", double(siz) / 1048576.0) ;
         lifeabortprogress(thiscell/(double)writecells, progressmsg) ;
      }
      root->next = (node *)thiscell ;
      os << depth+1 << ' ' << nw << ' ' << ne << ' ' << sw << ' ' << se << '\n';
   }
   return thiscell ;
}
#define STRINGIFY(arg) STR2(arg)
#define STR2(arg) #arg
const char *hlifealgo::writeNativeFormat(std::ostream &os, char *comments) {
   int depth = node_depth(root) ;
   os << "[M2] (golly " STRINGIFY(VERSION) ")\n" ;

   // AKT: always write out explicit rule
   os << "#R " << hliferules.getrule() << '\n' ;

   if (generation > bigint::zero) {
      // write non-zero gen count
      os << "#G " << generation.tostring('\0') << '\n' ;
   }
   
    if (comments) {
        // write given comment line(s), but we can't just do "os << comments" because the
        // lines might not start with #C (eg. if they came from the end of a .rle file),
        // so we must ensure that each comment line in the .mc file starts with #C
        char *p = comments;
        while (*p != '\0') {
            char *line = p;
            // note that readcomments() in readpattern.cpp ensures each line ends with \n
            while (*p != '\n') p++;
            if (line[0] != '#' || line[1] != 'C') {
                os << "#C ";
            }
            if (line != p) {
                *p = '\0';
                os << line;
                *p = '\n';
            }
            os << '\n';
            p++;
        }
    }
   
   inGC = 1 ;
   /* this is the old way:
   cellcounter = 0 ;
   writecell(f, root, depth) ;
   */
   /* this is the new two-pass way */
   cellcounter = 0 ;
   vector<int> depths(timeline.framecount) ;
   int framestosave = timeline.framecount ;
   if (timeline.savetimeline == 0)
     framestosave = 0 ;
   if (framestosave) {
     for (int i=0; i<timeline.framecount; i++) {
       node *frame = (node*)timeline.frames[i] ;
       depths[i] = node_depth(frame) ;
     }
     for (int i=0; i<timeline.framecount; i++) {
       node *frame = (node*)timeline.frames[i] ;
       writecell_2p1(frame, depths[i]) ;
     }
   }
   writecell_2p1(root, depth) ;
   writecells = cellcounter ;
   cellcounter = 0 ;
   if (framestosave) {
      os << "#FRAMES"
         << ' ' << timeline.framecount
         << ' ' << timeline.start.tostring()
         << ' ' << timeline.base << '^' << timeline.expo << '\n' ;
     for (int i=0; i<timeline.framecount; i++) {
       node *frame = (node*)timeline.frames[i] ;
       writecell_2p2(os, frame, depths[i]) ;
       os << "#FRAME " << i << ' ' << (g_uintptr_t)frame->next << '\n' ;
     }
   }
   writecell_2p2(os, root, depth) ;
   /* end new two-pass way */
   if (framestosave) {
     for (int i=0; i<timeline.framecount; i++) {
       node *frame = (node*)timeline.frames[i] ;
       afterwritemc(frame, depths[i]) ;
     }
   }
   afterwritemc(root, depth) ;
   inGC = 0 ;
   return 0 ;
}
char hlifealgo::statusline[200] ;
static lifealgo *creator() { return new hlifealgo() ; }
void hlifealgo::doInitializeAlgoInfo(staticAlgoInfo &ai) {
   ai.setAlgorithmName("HashLife") ;
   ai.setAlgorithmCreator(&creator) ;
   ai.setDefaultBaseStep(8) ;
   ai.setDefaultMaxMem(500) ; // MB
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
   hlifealgo *a = new hlifealgo() ;
   a->nthreads = hlifealgo::parthreads ;
   return a ;
}
void hlifealgo::doInitializeParAlgoInfo(staticAlgoInfo &ai) {
   doInitializeAlgoInfo(ai) ;
   ai.setAlgorithmName("HashLife Parallel") ;
   ai.setAlgorithmCreator(&parcreator) ;
}
/*
 *   Parallel stepping.
 *
 *   With nthreads >= 1, runpattern() computes the root's result with
 *   runpattern_par() instead of getres().  Every result is the same node
 *   the serial code computes; what changes is the order of the work.
 *
 *   Each thread runs the serial recursion (getres_h()/calc_h(), the same
 *   steps as getres()/dorecurs()).  Parallel work appears only when an
 *   idle thread asks for it, and no thread ever waits for a result that
 *   another thread computes:
 *
 *   - Asking for work: an idle thread posts a request to a working thread,
 *     which answers at its next safe point (the start of getres_h()).  It
 *     gives a sub-result its oldest unfinished computation has not started
 *     yet: it claims that node and the asking thread computes it.  If there
 *     is none, it gives a ready job stage (below), if it has one.
 *   - Claiming: a node at level parcutoff or above is claimed by changing
 *     its res field from 0 to 1 with compare-and-swap.  While it is
 *     claimed, res has the low bit set, and the rest of res is the list of
 *     waiting slots (hlwait) of jobs that need the result.  Storing the
 *     result takes the list in the same atomic exchange, and each waiting
 *     job gets the result.
 *   - When a computation finds a sub-result claimed by another thread (or
 *     left pending by one of its own sub-computations, see next), it goes
 *     on with the other sub-results of the same phase.  If some are still
 *     missing at the end of the phase, it becomes a job (hljob): it puts
 *     its slots on the waiting lists of the missing sub-results and returns
 *     "pending" to its caller, which then does the same.  A job builds
 *     each phase-2 node as soon as its 4 phase-1 inputs are there, and the
 *     result as soon as the 4 phase-2 results are there (a half step: the 9
 *     phase-1 results).  The thread that delivers the last missing input
 *     of a stage puts the stage on its list of ready stages and runs it
 *     later, unless another thread asks for it first.
 *   - Below parcutoff, nodes are computed by the plain serial recursion
 *     (getres_in()), without safe points; res == 1 while a thread computes
 *     the result, and another thread that needs it spins until it is
 *     stored.  That node has a lower level than the waiter's own, and its
 *     thread is running, so the wait ends.
 *   - Hash table lookups take no lock; a new node takes its index entry
 *     with compare-and-swap (see find_node_par).  Each thread takes free
 *     nodes from the global free list PARGRAB at a time (par_newnode).
 *   - When the index gets too full, or memory reaches maxmem, worker 0
 *     stops the other threads at safe points (they keep adding nodes
 *     until then) and garbage collects or grows the index, with every
 *     stopped thread helping (par_gc, par_index_grow).  The gc roots are
 *     the threads' unfinished computations (hlframe) and the jobs.  A
 *     thread that finds the index nearly full before that waits, and
 *     grows it if no other thread does (par_index_full).
 *
 *   Not handled yet: interrupting a step (the poller), and the hyperspeed
 *   perf counters.
 */
int hlifealgo::numthreads = 0 ;
int hlifealgo::parthreads = (int)std::thread::hardware_concurrency() ;
int hlifealgo::parcutoff = 0 ;
/*
 *   With parcutoff 0, the cutoff is chosen by timing steps: no level suits
 *   every pattern (a pattern whose results are mostly cached wants large
 *   tasks; one with much new work, more and smaller tasks).  Starting at
 *   PARCUTOFF, and again every pcinterval steps, a trial sequence of steps
 *   runs at the current cutoff c, at c+1 and at c-1 (pcarm); each trial
 *   step is timed against the mean of the steps at c just before and
 *   after it (which cancels a steady change in step times as the pattern
 *   evolves).  If c+1 or c-1 was faster by PCGAIN, it becomes the cutoff
 *   and the next trials come after PCFIRST steps; otherwise the interval
 *   doubles, up to PCMAXWAIT steps.  The timings start again when the
 *   step size (ngens) changes.
 */
static const int PARCUTOFF = 6 ;
static const int PCROUNDS = 4 ;                 // trial steps at c+1 and at c-1
static const int PCSEQ = 4 * PCROUNDS + 1 ;     // steps of a trial sequence
static inline int pcarm(int i) {                 // level of step i: c+1, c, c-1, c, ...
   return (i & 1) == 0 ? 0 : ((i & 2) ? -1 : 1) ;
}
static const int PCFIRST = 32, PCMAXWAIT = 512 ;
static const int PCWINS = PCROUNDS - 1 ;        // trial steps that must be faster
static const double PCGAIN = 0.93 ;
template<class T> static inline T load_acq(T *p) {
   return __atomic_load_n(p, __ATOMIC_ACQUIRE) ;
}
template<class T> static inline T load_rlx(T *p) {
   return __atomic_load_n(p, __ATOMIC_RELAXED) ;
}
template<class T> static inline void store_rel(T *p, T v) {
   __atomic_store_n(p, v, __ATOMIC_RELEASE) ;
}
template<class T> static inline void store_rlx(T *p, T v) {
   __atomic_store_n(p, v, __ATOMIC_RELAXED) ;
}
template<class T> static inline bool cas_ptr(T *p, T *expected, T desired) {
   return __atomic_compare_exchange_n(p, expected, desired, false,
                                      __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE) ;
}
static const int PARMINCUTOFF = 5 ;  // jobs' results must be nodes, not leaves
static const int PARGRAB = 1024 ;    // nodes moved to a thread's free list at once
static const int PARCOUNT = 256 ;    // new nodes counted per thread before adding to hashpop
static const int PARSPIN = 200000 ;  // idle spins between steps before sleeping
static const int PARBACKOFF = 32 ;   // pauses after a failed request for work,
static const int PARBACKOFFMAX = 1024 ;   // doubling up to this after each further one
static const int PARJOBCHUNK = 64 ;  // jobs allocated at once
static const unsigned int PARREMOTE = 4 ;   // see par_steal
#ifndef USEPREFETCH
struct setup_t {
   g_uintptr_t h ;
   struct node *nw, *ne, *sw, *se ;
} ;
#endif
/*
 *   A job's slot for one result it needs: 0..8 the phase-1 results t00 ..
 *   t22 (row by row), 9..12 the phase-2 results t33, t34, t43, t44.
 */
struct hlwait {
   hlwait *next ;
   hljob *job ;      // 0 for the root of the step
   int slot ;
} ;
struct alignas(64) hljob {
   node *n ;            // the node whose result this computes
   int depth ;          // level of n's children
   int half ;           // half step (as dorecurs_half)
   int live ;           // allocated; gc marks the nodes of live jobs
   hljob *nextfree ;
   node *in1[9], *r1[9] ;   // phase-1 nodes and their results
   node *in2[4], *r2[4] ;   // phase-2 nodes and their results
   std::atomic<int> cnt[4] ;  // missing inputs of each phase-2 node (half step: cnt[0], missing results)
   std::atomic<int> fin ;     // missing phase-2 results
   hlwait wt[13] ;
} ;
/*
 *   A computation in progress on a thread's stack (calc_h()); the thread
 *   gives other threads sub-results at positions end..k-1 of the phase's
 *   order (k = 9 or 4) that it has not reached yet (next).
 */
struct hlframe {
   node *n ;
   int depth ;          // level of n's children
   int half ;
   node *in1[9], *r1[9] ;
   node *in2[4], *r2[4] ;
   int phase ;          // 1 or 2
   int next, end ;
   unsigned int pend ;  // slots whose result is pending
} ;
/*
 *   Phase-1 result s is an input of the phase-2 nodes q with bit q set in
 *   parfeeds[s]; phase-2 node q is built from phase-1 results parin2[q].
 *   parorder1 and parorder2 are the orders dorecurs() computes them in.
 */
static const int parfeeds[9] = { 1, 3, 2, 5, 15, 10, 4, 12, 8 } ;
static const int parin2[4][4] = { {0, 1, 3, 4}, {1, 2, 4, 5}, {3, 4, 6, 7}, {4, 5, 7, 8} } ;
static const int parorder1[9] = { 0, 1, 2, 5, 4, 3, 6, 7, 8 } ;
static const int parorder2[4] = { 3, 2, 0, 1 } ;
/*
 *   Work handed between threads: a >= 0: compute the result of claimed
 *   node p (level a); a < 0: run stage -1-a of job p (0..3: phase-2 node
 *   q; PARSTAGEC: the result).
 */
struct hlitem {
   void *p ;
   long a ;
} ;
static const int PARSTAGEC = 4 ;
static const int PARCLOSED = -1 ;   // request slot of a thread that does not answer
static const int PARSTW = 1, PAREXCL = 2 ;   // bits of hlpar::attn
// phases of par_gc_work
static const int PARGC_MARK = 1, PARGC_CLEAR = 2, PARGC_SWEEP = 3, PARGC_REHASH = 4 ;
/*
 *   Fields written by one thread and read by others are kept on separate
 *   cache lines (alignas(64)).
 */
struct alignas(64) hlworker {
   // used by the owning thread (and by gc while all threads are stopped)
   int id ;
   int newnodes ;          // nodes added to the hash, not yet in hashpop
   node *freenodes ;       // this thread's free nodes (see par_newnode)
   int halves ;            // half-step results computed this step
   unsigned int rng ;
   unsigned int fails ;    // requests for work in a row that got none
   hljob *freejobs ;
   std::vector<hlframe *> frames ;   // computations in progress, oldest first
   std::vector<hlitem> ready ;       // ready job stages, a ring of ready.size() items
   size_t rhead, rcount ;
   // parallel gc (see par_gc)
   int gcseen ;                     // last gc phase this thread took part in
   std::vector<node *> gcstack ;
   node *gcfree, *gcfreelast ;      // nodes this thread's sweep freed
   g_uintptr_t gckept ;             // and the number it kept
   // 1 while the thread works (others only ask working threads for work)
   alignas(64) std::atomic<int> busy ;
   // 0 at safe points while the world is stopped, and when idle
   alignas(64) std::atomic<int> active ;
   // id+1 of the thread asking this one for work, 0, or PARCLOSED
   alignas(64) std::atomic<int> request ;
   // answer to this thread's request: 0 while waiting, 1 work in reply, 2 none
   alignas(64) std::atomic<int> replied ;
   hlitem reply ;
   hlworker() : id(0), newnodes(0), freenodes(0), halves(0), rng(1), fails(0), freejobs(0),
                ready(64), rhead(0), rcount(0), gcseen(0), gcfree(0), gcfreelast(0), gckept(0),
                busy(0), active(0), request(0), replied(0) {}
} ;
struct alignas(64) hlpar {
   int n ;
   int cutoff ;
   hlworker *w ;
   std::atomic<int> attn ;      // PARSTW: stop at safe points; PAREXCL: gc or a larger index wanted
   alignas(64) std::atomic<int> stepactive, quit ;
   // parallel gc: worker 0 publishes a phase (gcgen), stopped threads help;
   // gcnext is (gcgen << 32) | next item
   // the task cutoff chosen by timing steps (par_cutoff): the current one,
   // ngens it is for, the position in the trial sequence (-1: none
   // running), steps until the next trials, steps between trials, and the
   // times of the steps of the trial sequence
   int pcut, pcngens, pcpos, pcwait, pcinterval, pcdepth ;
   double pctime[17] ;
   alignas(64) std::atomic<int> gcgen ;
   int gcphase ;
   std::atomic<long> gctotal ;
   std::vector<node *> gcitems ;    // mark: subtrees; sweep: node blocks
   std::vector<g_uintptr_t> gcpos ; // clear, rehash: first entries of ranges
   hlindex newidx ;                 // rehash: the larger index
   alignas(64) std::atomic<long> gcnext ;
   alignas(64) std::atomic<long> gcdone ;
   alignas(64) node *rootres ;
   hlwait rootwait ;
   alignas(64) std::mutex alloclock ;   // global free list, nodeblocks, alloced, jobchunks
   // gc wanted because memory is full (par_newnode); threads paused where
   // they may hold nodes that are no gc roots (par_index_full, par_pause);
   // 1 while a stop of the world may only grow the index; index replacements
   std::atomic<int> gcwanted, paused, stwgrow, idxgen ;
   std::vector<hljob *> jobchunks ;
   alignas(64) std::mutex sleepmx ;
   std::condition_variable sleepcv ;
   std::vector<std::thread> threads ;
   // cache domains (see par_topology): domain of each worker, the workers
   // of each domain, and the CPUs of each domain (empty: no affinity set)
   std::vector<int> wdomain ;
   std::vector<std::vector<int> > dworkers ;
   std::vector<std::vector<int> > dcpus ;
   hlpar(int nthreads) : n(nthreads), cutoff(0), attn(0),
                         stepactive(0), quit(0), gcgen(0), gcphase(0), gctotal(0),
                         gcnext(0), gcdone(0), rootres(0),
                         gcwanted(0), paused(0), stwgrow(0), idxgen(0) {
      pcut = pcngens = pcpos = pcwait = pcinterval = 0 ;
      w = new hlworker[n] ;
      for (int i=0; i<n; i++) {
         w[i].id = i ;
         w[i].rng = 2463534242u + 7919u * i ;
      }
      w[0].request.store(PARCLOSED) ;
      rootwait.next = 0 ;
      rootwait.job = 0 ;
      rootwait.slot = 0 ;
   }
} ;
/*
 *   The ready stages of a thread: it runs the newest, and gives the oldest
 *   to a thread that asks.
 */
static void rq_push(hlworker &w, void *p, long a) {
   if (w.rcount == w.ready.size()) {
      std::vector<hlitem> nr(2 * w.ready.size()) ;
      for (size_t i=0; i<w.rcount; i++)
         nr[i] = w.ready[(w.rhead + i) % w.ready.size()] ;
      w.ready.swap(nr) ;
      w.rhead = 0 ;
   }
   hlitem &it = w.ready[(w.rhead + w.rcount++) % w.ready.size()] ;
   it.p = p ;
   it.a = a ;
}
static hlitem rq_pop_newest(hlworker &w) {
   return w.ready[(w.rhead + --w.rcount) % w.ready.size()] ;
}
static hlitem rq_pop_oldest(hlworker &w) {
   hlitem it = w.ready[w.rhead] ;
   w.rhead = (w.rhead + 1) % w.ready.size() ;
   w.rcount-- ;
   return it ;
}
/*
 *   Stopping the world.  A thread sets active before looking at attn, and
 *   worker 0 sets attn before looking at active (both sequentially
 *   consistent), so worker 0 never runs gc while another thread is
 *   between safe points.
 */
static void par_enter(hlpar *par, hlworker &w) {
   for (;;) {
      w.active.store(1) ;
      if (!(par->attn.load() & PARSTW))
         return ;
      w.active.store(0) ;
      while (par->attn.load(std::memory_order_acquire) & PARSTW)
         cpu_relax() ;
   }
}
/*
 *   Wait until every other thread has stopped: at a safe point, or paused
 *   (par_index_full, par_pause) where it may hold nodes that are no gc
 *   roots.  While a thread is paused, the stop may only grow the index
 *   (stwgrow); threads that spin on a result then pause too (the result
 *   may be one a paused thread computes).
 */
static void par_wait_stopped(hlpar *par, hlworker &w) {
   for (int i=0; i<par->n; i++)
      if (i != w.id)
         while (par->w[i].active.load()) {
            if (par->paused.load() && !par->stwgrow.load())
               par->stwgrow.store(1) ;
            cpu_relax() ;
         }
}
void hlifealgo::par_attention(hlworker &w) {
   int a = par->attn.load() ;
   if (a & PARSTW) {
      par_flush(w) ;
      w.active.store(0) ;
      while (par->attn.load(std::memory_order_acquire) & PARSTW) {
         int g = par->gcgen.load(std::memory_order_acquire) ;
         if (g != w.gcseen) {
            w.gcseen = g ;
            par_gc_work(w, g) ;
         }
         cpu_relax() ;
      }
      par_enter(par, w) ;
   } else if ((a & PAREXCL) && w.id == 0 && par->attn.compare_exchange_strong(a, PARSTW)) {
      par_flush(w) ;
      par_wait_stopped(par, w) ;
      // the same decisions as newnode() and index_full(); no gc while a
      // thread is paused
      int cangc = okaytogc && par->paused.load() == 0 ;
      int gced = 0 ;
      if (cangc && par->gcwanted.load()) {
         par_gc(w) ;
         par->gcwanted.store(0) ;
         gced = 1 ;
      }
      if (idx.used > idx.limit) {
         if (cangc && !gced &&
             alloced + (INDEXGROW - 1) * idx.cap * sizeof(hlentry) > maxmem) {
            par_gc(w) ;
            par->gcwanted.store(0) ;
            gced = 1 ;
         }
         if (!gced || idx.used > idx.limit - idx.limit / 5)
            par_index_grow(w) ;
      }
      par->stwgrow.store(0) ;
      par->attn.store(par->gcwanted.load() ? PAREXCL : 0, std::memory_order_release) ;
   }
}
#define PAR_SAFEPOINT(w) \
   do { if (par->attn.load(std::memory_order_relaxed)) par_attention(w) ; } while (0)
#define PAR_ANSWER(w) \
   do { if (w.request.load(std::memory_order_relaxed) > 0) par_answer(w) ; } while (0)
/*
 *   Add this thread's new nodes to hashpop and to the used entries of the
 *   index (par_flush), and ask worker 0 for a larger index when it is too
 *   full.  Threads flush before they stop or go idle, so the counts are
 *   exact while the world is stopped.
 */
void hlifealgo::par_flush(hlworker &w) {
   if (w.newnodes == 0)
      return ;
   __atomic_add_fetch(&hashpop, (g_uintptr_t)w.newnodes, __ATOMIC_RELAXED) ;
   __atomic_add_fetch(&idx.used, (g_uintptr_t)w.newnodes, __ATOMIC_RELAXED) ;
   w.newnodes = 0 ;
}
void hlifealgo::par_count_nodes(hlworker &w) {
   par_flush(w) ;
   if (load_rlx(&idx.used) > idx.limit &&
       !(par->attn.load(std::memory_order_relaxed) & PAREXCL))
      par->attn.fetch_or(PAREXCL) ;
}
/*
 *   A node for this thread: PARGRAB nodes at a time come from the global
 *   free list.  Same policy as newnode(): when the free list is empty and
 *   growing would exceed maxmem, garbage collect; the gc waits until
 *   worker 0 reaches a safe point, and until then the memory grows.
 */
node *hlifealgo::par_newnode(hlworker &w) {
   node *p = w.freenodes ;
   if (p) {
      w.freenodes = p->next ;
      return p ;
   }
   par->alloclock.lock() ;
   if (freenodes == 0) {
      if (okaytogc && alloced + NODECHUNKCOUNT * sizeof(node) > maxmem &&
          !par->gcwanted.load(std::memory_order_relaxed)) {
         par->gcwanted.store(1) ;
         par->attn.fetch_or(PAREXCL) ;
      }
      node *blk = newblock() ;
      if (blk == 0) {
         par->alloclock.unlock() ;
         lifefatal("Out of memory; try reducing the hash memory limit.") ;
      }
      alloced += NODECHUNKCOUNT * sizeof(node) ;
      blk->next = nodeblocks ;
      blk->res = (node *)(g_uintptr_t)NODECHUNKCOUNT ;
      nodeblocks = blk ;
      for (int i=1; i<NODECHUNKCOUNT; i++)
         blk[i].next = blk + i + 1 ;
      freenodes = blk + 1 ;
      totalthings += NODECHUNKCOUNT ;
   }
   node *first = freenodes, *last = first ;
   for (int k=1; k<PARGRAB && last->next; k++)
      last = last->next ;
   freenodes = last->next ;
   last->next = 0 ;
   par->alloclock.unlock() ;
   w.freenodes = first->next ;
   return first ;
}
/*
 *   The index is nearly full, and worker 0 has not grown it yet (it grows
 *   it at a safe point, when the index is above its limit): stop adding
 *   nodes and wait, paused, until the index has been replaced.  If no
 *   other thread has stopped the world, this thread stops it and grows
 *   the index (without gc: paused threads hold nodes that are no roots).
 */
void hlifealgo::par_index_full(hlworker &w) {
   int g0 = par->idxgen.load(std::memory_order_acquire) ;
   par_flush(w) ;
   par->paused.fetch_add(1) ;
   w.active.store(0) ;
   while (par->idxgen.load(std::memory_order_acquire) == g0) {
      int a = par->attn.load() ;
      if (!(a & PARSTW) && par->attn.compare_exchange_strong(a, a | PARSTW)) {
         par->stwgrow.store(1) ;
         par_wait_stopped(par, w) ;
         if (par->idxgen.load() == g0)
            par_index_grow(w) ;
         par->stwgrow.store(0) ;
         par->attn.fetch_and(~PARSTW) ;
         break ;
      }
      int g = par->gcgen.load(std::memory_order_acquire) ;
      if (g != w.gcseen) {
         w.gcseen = g ;
         par_gc_work(w, g) ;
      }
      cpu_relax() ;
   }
   par->paused.fetch_sub(1) ;
   par_enter(par, w) ;
}
/*
 *   A thread that spins on a result another thread computes (getres_in)
 *   pauses while the world is stopped to grow the index.
 */
static void par_pause(hlpar *par, hlworker &w) {
   par->paused.fetch_add(1) ;
   w.active.store(0) ;
   while (par->attn.load(std::memory_order_acquire) & PARSTW)
      cpu_relax() ;
   par->paused.fetch_sub(1) ;
   par_enter(par, w) ;
}
/*
 *   Put p, which is not in x yet, into x, while other threads may do the
 *   same with other nodes.
 */
static void index_put_par(hlindex &x, node *p) {
   g_uintptr_t m = hlmix(hlhash(p)) ;
   hlentry v = (hlentry)(g_uintptr_t)p | hltag(m) ;
   for (g_uintptr_t i=hlslot(m, x.cap); ; i=hlnext(i, x.cap)) {
      hlentry e = 0 ;
      if (load_rlx(x.e + i) == 0 && cas_ptr(x.e + i, &e, v))
         return ;
   }
}
/*
 *   Grow the index, with the stopped threads helping (par_gc_work).
 */
static const g_uintptr_t PARGCRANGE = 1 << 16 ;   // entries per item of clear and rehash
void hlifealgo::par_index_grow(hlworker &w) {
   g_uintptr_t ncap = INDEXGROW * idx.cap ;
   if (verbose) {
     sprintf(statusline, "Resizing hash to %" PRIuPTR "...", ncap) ;
     lifestatus(statusline) ;
   }
   index_alloc(par->newidx, ncap) ;
   par->gcpos.clear() ;
   for (g_uintptr_t j=0; j<idx.cap; j+=PARGCRANGE)
      par->gcpos.push_back(j) ;
   par_gc_phase(w, PARGC_REHASH, (long)par->gcpos.size()) ;
   par->newidx.used = idx.used ;
   free(idx.e) ;
   alloced += (ncap - idx.cap) * sizeof(hlentry) ;
   idx = par->newidx ;
   par->idxgen.fetch_add(1, std::memory_order_release) ;
   if (verbose) {
     strcpy(statusline+strlen(statusline), " done.") ;
     lifestatus(statusline) ;
   }
}
/*
 *   Like find_node_h(), for any number of threads.  The index is only
 *   replaced or cleared while every thread is stopped or paused.  A new
 *   node is filled in first and then put into the first empty entry of
 *   its probe sequence with compare-and-swap; entries only go from empty
 *   to used during a step, so two threads adding the same node try the
 *   same entry, and the one that loses finds the other's node there and
 *   keeps its own for later.
 */
node *hlifealgo::find_node_par(hlworker &w, g_uintptr_t h,
                               node *nw, node *ne, node *sw, node *se) {
   if (load_rlx(&idx.used) + w.newnodes > idx.cap - idx.cap / 10)
      par_index_full(w) ;
   g_uintptr_t x = hlmix(h) ;
   hlentry tag = hltag(x) ;
   hlentry *ie = idx.e ;
   g_uintptr_t cap = idx.cap ;
   node *p = 0 ;
   for (g_uintptr_t i=hlslot(x, cap); ; i=hlnext(i, cap)) {
      hlentry e = load_acq(ie + i) ;
      if (e == 0) {
         if (p == 0) {
            p = par_newnode(w) ;
            p->nw = nw ;
            p->ne = ne ;
            p->sw = sw ;
            p->se = se ;
            p->res = 0 ;
            p->next = 0 ;
         }
         if (cas_ptr(ie + i, &e, (hlentry)(g_uintptr_t)p | tag)) {
            if (++w.newnodes >= PARCOUNT)
               par_count_nodes(w) ;
            return p ;
         }
         // e is now the entry another thread put here
      }
      if ((e & ~HLPTRMASK) == tag) {
         node *q = hlptr(e) ;
         if (q->nw == nw && q->ne == ne && q->sw == sw && q->se == se) {
            if (p) {
               p->next = w.freenodes ;
               w.freenodes = p ;
            }
            return q ;
         }
      }
   }
}
node *hlifealgo::find_node_par(hlworker &w, node *nw, node *ne, node *sw, node *se) {
   return find_node_par(w, node_hash(nw, ne, sw, se), nw, ne, sw, se) ;
}
leaf *hlifealgo::find_leaf_par(hlworker &w, unsigned short nw, unsigned short ne,
                               unsigned short sw, unsigned short se) {
   if (load_rlx(&idx.used) + w.newnodes > idx.cap - idx.cap / 10)
      par_index_full(w) ;
   g_uintptr_t x = hlmix(leaf_hash(nw, ne, sw, se)) ;
   hlentry tag = hltag(x) ;
   hlentry *ie = idx.e ;
   g_uintptr_t cap = idx.cap ;
   leaf *p = 0 ;
   for (g_uintptr_t i=hlslot(x, cap); ; i=hlnext(i, cap)) {
      hlentry e = load_acq(ie + i) ;
      if (e == 0) {
         if (p == 0) {
            p = (leaf *)par_newnode(w) ;
            new(&(p->leafpop))bigint ;
            p->isnode = (node *)SLOTLEAF ;
            p->nw = nw ;
            p->ne = ne ;
            p->sw = sw ;
            p->se = se ;
            leafres(p) ;
            p->wnum = 0 ;
            p->next = 0 ;
         }
         if (cas_ptr(ie + i, &e, (hlentry)(g_uintptr_t)p | tag)) {
            if (++w.newnodes >= PARCOUNT)
               par_count_nodes(w) ;
            return p ;
         }
      }
      if ((e & ~HLPTRMASK) == tag) {
         leaf *q = (leaf *)hlptr(e) ;
         if ((g_uintptr_t)q->isnode == SLOTLEAF &&
             q->nw == nw && q->ne == ne && q->sw == sw && q->se == se) {
            if (p) {
               p->next = (node *)w.freenodes ;
               w.freenodes = (node *)p ;
            }
            return q ;
         }
      }
   }
}
/*
 *   The serial recursion below parcutoff.  It has no safe points, so it
 *   does not save nodes for the gc.
 */
node *hlifealgo::getres_in(hlworker &w, node *n, int depth) {
   node *r = load_acq(&n->res) ;
   if (r == 0) {
      store_rlx(&n->res, (node *)1) ;
      r = calc_in(w, n, depth) ;
      store_rel(&n->res, r) ;
      return r ;
   }
   while (ISPARBUSY(r)) {
      if ((par->attn.load(std::memory_order_relaxed) & PARSTW) &&
          par->stwgrow.load(std::memory_order_relaxed))
         par_pause(par, w) ;
      cpu_relax() ;
      r = load_acq(&n->res) ;
   }
   return r ;
}
node *hlifealgo::calc_in(hlworker &w, node *n, int depth) {
   depth-- ;
   if (ngens >= depth) {
      if (is_node(n->nw))
         return dorecurs_in(w, n->nw, n->ne, n->sw, n->se, depth) ;
      return (node *)dorecurs_leaf_in(w, (leaf *)n->nw, (leaf *)n->ne,
                                      (leaf *)n->sw, (leaf *)n->se) ;
   }
   w.halves++ ;
   if (is_node(n->nw))
      return dorecurs_half_in(w, n->nw, n->ne, n->sw, n->se, depth) ;
   if (ngens == 0)
      return (node *)dorecurs_leaf_quarter_in(w, (leaf *)n->nw, (leaf *)n->ne,
                                              (leaf *)n->sw, (leaf *)n->se) ;
   return (node *)dorecurs_leaf_half_in(w, (leaf *)n->nw, (leaf *)n->ne,
                                        (leaf *)n->sw, (leaf *)n->se) ;
}
#ifdef USEPREFETCH
#define PARKEY(k, a, b, c, d) setupprefetch(k, a, b, c, d)
#else
#define PARKEY(k, a, b, c, d) ((k).h = node_hash(a, b, c, d), (k).nw = a, \
                               (k).ne = b, (k).sw = c, (k).se = d)
#endif
#define PARFIND(k) find_node_par(w, (k).h, (k).nw, (k).ne, (k).sw, (k).se)
node *hlifealgo::dorecurs_in(hlworker &w, node *n, node *ne, node *t, node *e, int depth) {
   setup_t su[5] ;
   PARKEY(su[2], n->se, ne->sw, t->ne, e->nw) ;
   PARKEY(su[0], n->ne, ne->nw, n->se, ne->sw) ;
   PARKEY(su[1], ne->sw, ne->se, e->nw, e->ne) ;
   PARKEY(su[3], n->sw, n->se, t->nw, t->ne) ;
   PARKEY(su[4], t->ne, e->nw, t->se, e->sw) ;
   node
   *t00 = getres_in(w, n, depth),
   *t01 = getres_in(w, PARFIND(su[0]), depth),
   *t02 = getres_in(w, ne, depth),
   *t12 = getres_in(w, PARFIND(su[1]), depth),
   *t11 = getres_in(w, PARFIND(su[2]), depth),
   *t10 = getres_in(w, PARFIND(su[3]), depth),
   *t20 = getres_in(w, t, depth),
   *t21 = getres_in(w, PARFIND(su[4]), depth),
   *t22 = getres_in(w, e, depth) ;
   PARKEY(su[0], t11, t12, t21, t22) ;
   PARKEY(su[1], t10, t11, t20, t21) ;
   PARKEY(su[2], t00, t01, t10, t11) ;
   PARKEY(su[3], t01, t02, t11, t12) ;
   node
   *t44 = getres_in(w, PARFIND(su[0]), depth),
   *t43 = getres_in(w, PARFIND(su[1]), depth),
   *t33 = getres_in(w, PARFIND(su[2]), depth),
   *t34 = getres_in(w, PARFIND(su[3]), depth) ;
   return find_node_par(w, t33, t34, t43, t44) ;
}
node *hlifealgo::dorecurs_half_in(hlworker &w, node *n, node *ne, node *t,
                                  node *e, int depth) {
   node
   *t00 = getres_in(w, n, depth),
   *t01 = getres_in(w, find_node_par(w, n->ne, ne->nw, n->se, ne->sw), depth),
   *t10 = getres_in(w, find_node_par(w, n->sw, n->se, t->nw, t->ne), depth),
   *t11 = getres_in(w, find_node_par(w, n->se, ne->sw, t->ne, e->nw), depth),
   *t02 = getres_in(w, ne, depth),
   *t12 = getres_in(w, find_node_par(w, ne->sw, ne->se, e->nw, e->ne), depth),
   *t20 = getres_in(w, t, depth),
   *t21 = getres_in(w, find_node_par(w, t->ne, e->nw, t->se, e->sw), depth),
   *t22 = getres_in(w, e, depth) ;
   if (depth > 3)
      return find_node_par(w, find_node_par(w, t00->se, t01->sw, t10->ne, t11->nw),
                              find_node_par(w, t01->se, t02->sw, t11->ne, t12->nw),
                              find_node_par(w, t10->se, t11->sw, t20->ne, t21->nw),
                              find_node_par(w, t11->se, t12->sw, t21->ne, t22->nw)) ;
   return find_node_par(w, (node *)find_leaf_par(w, ((leaf *)t00)->se,
                                                    ((leaf *)t01)->sw,
                                                    ((leaf *)t10)->ne,
                                                    ((leaf *)t11)->nw),
                           (node *)find_leaf_par(w, ((leaf *)t01)->se,
                                                    ((leaf *)t02)->sw,
                                                    ((leaf *)t11)->ne,
                                                    ((leaf *)t12)->nw),
                           (node *)find_leaf_par(w, ((leaf *)t10)->se,
                                                    ((leaf *)t11)->sw,
                                                    ((leaf *)t20)->ne,
                                                    ((leaf *)t21)->nw),
                           (node *)find_leaf_par(w, ((leaf *)t11)->se,
                                                    ((leaf *)t12)->sw,
                                                    ((leaf *)t21)->ne,
                                                    ((leaf *)t22)->nw)) ;
}
leaf *hlifealgo::dorecurs_leaf_in(hlworker &w, leaf *n, leaf *ne, leaf *t, leaf *e) {
   if (leafnk >= 0) {
      unsigned short q[4] ;
      leafstep(n, ne, t, e, 4, q) ;
      return find_leaf_par(w, q[0], q[1], q[2], q[3]) ;
   }
   unsigned short
   t00 = n->res2,
   t01 = find_leaf_par(w, n->ne, ne->nw, n->se, ne->sw)->res2,
   t02 = ne->res2,
   t10 = find_leaf_par(w, n->sw, n->se, t->nw, t->ne)->res2,
   t11 = find_leaf_par(w, n->se, ne->sw, t->ne, e->nw)->res2,
   t12 = find_leaf_par(w, ne->sw, ne->se, e->nw, e->ne)->res2,
   t20 = t->res2,
   t21 = find_leaf_par(w, t->ne, e->nw, t->se, e->sw)->res2,
   t22 = e->res2 ;
   return find_leaf_par(w, find_leaf_par(w, t00, t01, t10, t11)->res2,
                           find_leaf_par(w, t01, t02, t11, t12)->res2,
                           find_leaf_par(w, t10, t11, t20, t21)->res2,
                           find_leaf_par(w, t11, t12, t21, t22)->res2) ;
}
leaf *hlifealgo::dorecurs_leaf_half_in(hlworker &w, leaf *n, leaf *ne, leaf *t, leaf *e) {
   if (leafnk >= 0) {
      unsigned short q[4] ;
      leafstep(n, ne, t, e, 2, q) ;
      return find_leaf_par(w, q[0], q[1], q[2], q[3]) ;
   }
   unsigned short
   t00 = n->res2,
   t01 = find_leaf_par(w, n->ne, ne->nw, n->se, ne->sw)->res2,
   t02 = ne->res2,
   t10 = find_leaf_par(w, n->sw, n->se, t->nw, t->ne)->res2,
   t11 = find_leaf_par(w, n->se, ne->sw, t->ne, e->nw)->res2,
   t12 = find_leaf_par(w, ne->sw, ne->se, e->nw, e->ne)->res2,
   t20 = t->res2,
   t21 = find_leaf_par(w, t->ne, e->nw, t->se, e->sw)->res2,
   t22 = e->res2 ;
   return find_leaf_par(w, combine4(t00, t01, t10, t11),
                           combine4(t01, t02, t11, t12),
                           combine4(t10, t11, t20, t21),
                           combine4(t11, t12, t21, t22)) ;
}
leaf *hlifealgo::dorecurs_leaf_quarter_in(hlworker &w, leaf *n, leaf *ne,
                                          leaf *t, leaf *e) {
   if (leafnk >= 0) {
      unsigned short q[4] ;
      leafstep(n, ne, t, e, 1, q) ;
      return find_leaf_par(w, q[0], q[1], q[2], q[3]) ;
   }
   unsigned short
   t00 = n->res1,
   t01 = find_leaf_par(w, n->ne, ne->nw, n->se, ne->sw)->res1,
   t02 = ne->res1,
   t10 = find_leaf_par(w, n->sw, n->se, t->nw, t->ne)->res1,
   t11 = find_leaf_par(w, n->se, ne->sw, t->ne, e->nw)->res1,
   t12 = find_leaf_par(w, ne->sw, ne->se, e->nw, e->ne)->res1,
   t20 = t->res1,
   t21 = find_leaf_par(w, t->ne, e->nw, t->se, e->sw)->res1,
   t22 = e->res1 ;
   return find_leaf_par(w, combine4(t00, t01, t10, t11),
                           combine4(t01, t02, t11, t12),
                           combine4(t10, t11, t20, t21),
                           combine4(t11, t12, t21, t22)) ;
}
/*
 *   Jobs.
 */
hljob *hlifealgo::par_job_alloc(hlworker &w) {
   hljob *j = w.freejobs ;
   if (j == 0) {
      j = new hljob[PARJOBCHUNK] ;
      for (int i=0; i<PARJOBCHUNK; i++) {
         j[i].live = 0 ;
         j[i].nextfree = (i + 1 < PARJOBCHUNK) ? j + i + 1 : 0 ;
         for (int s=0; s<13; s++) {
            j[i].wt[s].job = j + i ;
            j[i].wt[s].slot = s ;
         }
      }
      std::lock_guard<std::mutex> lk(par->alloclock) ;
      par->jobchunks.push_back(j) ;
   }
   w.freejobs = j->nextfree ;
   j->live = 1 ;
   return j ;
}
static void par_job_free(hlworker &w, hljob *j) {
   j->live = 0 ;
   j->nextfree = w.freejobs ;
   w.freejobs = j ;
}
/*
 *   Put slot wt on the waiting list of claimed node c; returns c's result
 *   instead if it was stored meanwhile.
 */
static node *par_wait(node *c, hlwait *wt) {
   node *r = load_acq(&c->res) ;
   for (;;) {
      if (!ISPARBUSY(r))
         return r ;
      wt->next = (hlwait *)((g_uintptr_t)r & ~(g_uintptr_t)1) ;
      if (cas_ptr(&c->res, &r, (node *)((g_uintptr_t)wt | 1)))
         return 0 ;
   }
}
/*
 *   Store the result r of claimed node c and hand it to every waiting slot.
 */
void hlifealgo::par_complete(hlworker &w, node *c, node *r) {
   node *old = __atomic_exchange_n(&c->res, r, __ATOMIC_ACQ_REL) ;
   hlwait *wt = (hlwait *)((g_uintptr_t)old & ~(g_uintptr_t)1) ;
   while (wt) {
      hlwait *next = wt->next ;   // the job may be finished and reused after par_deliver
      par_deliver(w, wt, r) ;
      wt = next ;
   }
}
void hlifealgo::par_deliver(hlworker &w, hlwait *wt, node *r) {
   hljob *j = wt->job ;
   if (j == 0) {
      store_rel(&par->rootres, r) ;
      return ;
   }
   int s = wt->slot ;
   if (s < 9) {
      j->r1[s] = r ;
      if (j->half) {
         if (j->cnt[0].fetch_sub(1, std::memory_order_acq_rel) == 1)
            rq_push(w, j, -1 - PARSTAGEC) ;
      } else {
         // j stays alive until every counter reaches 0, so it is safe to
         // touch it after an earlier counter in this loop did
         int f = parfeeds[s] ;
         for (int q=0; q<4; q++)
            if (((f >> q) & 1) && j->cnt[q].fetch_sub(1, std::memory_order_acq_rel) == 1)
               rq_push(w, j, -1 - q) ;
      }
   } else {
      j->r2[s-9] = r ;
      if (j->fin.fetch_sub(1, std::memory_order_acq_rel) == 1)
         rq_push(w, j, -1 - PARSTAGEC) ;
   }
}
/*
 *   Turn computation f, with pending slots f.pend, into a job.  Its
 *   counters start one higher than the number of missing inputs, so no
 *   stage can start before this function has looked at all of them.
 */
void hlifealgo::par_convert(hlworker &w, hlframe &f) {
   hljob *j = par_job_alloc(w) ;
   j->n = f.n ;
   j->depth = f.depth ;
   j->half = f.half ;
   for (int i=0; i<9; i++) {
      j->in1[i] = f.in1[i] ;
      j->r1[i] = f.r1[i] ;
   }
   for (int q=0; q<4; q++) {
      j->in2[q] = f.in2[q] ;
      j->r2[q] = f.r2[q] ;
   }
   if (f.phase == 1) {
      int missq[4] = { 0, 0, 0, 0 }, miss = 0 ;
      for (int s=0; s<9; s++)
         if ((f.pend >> s) & 1) {
            miss++ ;
            for (int q=0; q<4; q++)
               missq[q] += (parfeeds[s] >> q) & 1 ;
         }
      if (j->half) {
         j->cnt[0].store(miss + 1, std::memory_order_relaxed) ;
      } else {
         for (int q=0; q<4; q++)
            j->cnt[q].store(missq[q] + 1, std::memory_order_relaxed) ;
         j->fin.store(4, std::memory_order_relaxed) ;
      }
      int found = 0, foundq[4] = { 0, 0, 0, 0 } ;
      for (int s=0; s<9; s++)
         if ((f.pend >> s) & 1) {
            node *r = par_wait(j->in1[s], &j->wt[s]) ;
            if (r) {
               j->r1[s] = r ;
               found++ ;
               for (int q=0; q<4; q++)
                  foundq[q] += (parfeeds[s] >> q) & 1 ;
            }
         }
      if (j->half) {
         if (j->cnt[0].fetch_sub(found + 1, std::memory_order_acq_rel) == found + 1)
            par_finish(w, j) ;
      } else {
         // j stays alive until the last of these counters reaches 0
         for (int q=0; q<4; q++)
            if (j->cnt[q].fetch_sub(foundq[q] + 1, std::memory_order_acq_rel) == foundq[q] + 1)
               par_stage2(w, j, q) ;
      }
   } else {
      int miss = 0, found = 0 ;
      for (int q=0; q<4; q++)
         miss += (f.pend >> (9 + q)) & 1 ;
      j->fin.store(miss + 1, std::memory_order_relaxed) ;
      for (int q=0; q<4; q++)
         if ((f.pend >> (9 + q)) & 1) {
            node *r = par_wait(j->in2[q], &j->wt[9+q]) ;
            if (r) {
               j->r2[q] = r ;
               found++ ;
            }
         }
      if (j->fin.fetch_sub(found + 1, std::memory_order_acq_rel) == found + 1)
         par_finish(w, j) ;
   }
}
/*
 *   Build phase-2 node q of job j from its 4 phase-1 results and get its
 *   result.
 */
void hlifealgo::par_stage2(hlworker &w, hljob *j, int q) {
   const int *in = parin2[q] ;
   node *c = find_node_par(w, j->r1[in[0]], j->r1[in[1]], j->r1[in[2]], j->r1[in[3]]) ;
   j->in2[q] = c ;
   node *r = getres_h(w, c, j->depth) ;
   if (r == 0)
      r = par_wait(c, &j->wt[9+q]) ;
   if (r) {
      j->r2[q] = r ;
      if (j->fin.fetch_sub(1, std::memory_order_acq_rel) == 1)
         par_finish(w, j) ;
   }
}
void hlifealgo::par_finish(hlworker &w, hljob *j) {
   node *r ;
   if (j->half) {
      node **t = j->r1 ;
      r = find_node_par(w, find_node_par(w, t[0]->se, t[1]->sw, t[3]->ne, t[4]->nw),
                           find_node_par(w, t[1]->se, t[2]->sw, t[4]->ne, t[5]->nw),
                           find_node_par(w, t[3]->se, t[4]->sw, t[6]->ne, t[7]->nw),
                           find_node_par(w, t[4]->se, t[5]->sw, t[7]->ne, t[8]->nw)) ;
      w.halves++ ;
   } else {
      r = find_node_par(w, j->r2[0], j->r2[1], j->r2[2], j->r2[3]) ;
   }
   par_complete(w, j->n, r) ;
   par_job_free(w, j) ;
}
/*
 *   The result of node n at level depth >= parcutoff, or 0 if it is
 *   pending (another thread computes it, or it became a job).  This is the
 *   safe point: gc may run here, and requests for work are answered.
 */
node *hlifealgo::getres_h(hlworker &w, node *n, int depth) {
   PAR_SAFEPOINT(w) ;
   PAR_ANSWER(w) ;
   node *r = load_acq(&n->res) ;
   if (r == 0 && cas_ptr(&n->res, &r, (node *)1))
      return calc_h(w, n, depth) ;
   return ISPARBUSY(r) ? 0 : r ;
}
/*
 *   Compute the result of claimed node n at level depth >= parcutoff, as
 *   getres() and dorecurs() do; returns 0 if it became a job.
 */
node *hlifealgo::calc_h(hlworker &w, node *n, int depth) {
   if (depth == par->cutoff) {
      node *r = calc_in(w, n, depth) ;
      par_complete(w, n, r) ;
      return r ;
   }
   hlframe f ;
   depth-- ;
   f.n = n ;
   f.depth = depth ;
   f.half = ngens < depth ;
   node *nw = n->nw, *ne = n->ne, *sw = n->sw, *se = n->se ;
   setup_t su[5] ;
   PARKEY(su[2], nw->se, ne->sw, sw->ne, se->nw) ;
   PARKEY(su[0], nw->ne, ne->nw, nw->se, ne->sw) ;
   PARKEY(su[1], ne->sw, ne->se, se->nw, se->ne) ;
   PARKEY(su[3], nw->sw, nw->se, sw->nw, sw->ne) ;
   PARKEY(su[4], sw->ne, se->nw, sw->se, se->sw) ;
   f.in1[0] = nw ;
   f.in1[1] = PARFIND(su[0]) ;
   f.in1[2] = ne ;
   f.in1[5] = PARFIND(su[1]) ;
   f.in1[4] = PARFIND(su[2]) ;
   f.in1[3] = PARFIND(su[3]) ;
   f.in1[6] = sw ;
   f.in1[7] = PARFIND(su[4]) ;
   f.in1[8] = se ;
   for (int i=0; i<9; i++)
      f.r1[i] = 0 ;
   for (int q=0; q<4; q++)
      f.in2[q] = f.r2[q] = 0 ;
   f.phase = 1 ;
   f.next = 0 ;
   f.end = 9 ;
   f.pend = 0 ;
   w.frames.push_back(&f) ;
   for (int pos=0; pos<9; pos++) {
      int s = parorder1[pos] ;
      f.next = pos + 1 ;
      node *r = getres_h(w, f.in1[s], depth) ;
      if (r)
         f.r1[s] = r ;
      else
         f.pend |= 1u << s ;
   }
   if (f.pend) {
      w.frames.pop_back() ;
      par_convert(w, f) ;
      return 0 ;
   }
   node *r ;
   if (f.half) {
      node **t = f.r1 ;
      r = find_node_par(w, find_node_par(w, t[0]->se, t[1]->sw, t[3]->ne, t[4]->nw),
                           find_node_par(w, t[1]->se, t[2]->sw, t[4]->ne, t[5]->nw),
                           find_node_par(w, t[3]->se, t[4]->sw, t[6]->ne, t[7]->nw),
                           find_node_par(w, t[4]->se, t[5]->sw, t[7]->ne, t[8]->nw)) ;
      w.halves++ ;
   } else {
      node **t = f.r1 ;
      PARKEY(su[3], t[4], t[5], t[7], t[8]) ;
      PARKEY(su[2], t[3], t[4], t[6], t[7]) ;
      PARKEY(su[0], t[0], t[1], t[3], t[4]) ;
      PARKEY(su[1], t[1], t[2], t[4], t[5]) ;
      for (int q=0; q<4; q++)
         f.in2[q] = PARFIND(su[q]) ;
      f.phase = 2 ;
      f.next = 0 ;
      f.end = 4 ;
      for (int pos=0; pos<4; pos++) {
         int q = parorder2[pos] ;
         f.next = pos + 1 ;
         node *r2 = getres_h(w, f.in2[q], depth) ;
         if (r2)
            f.r2[q] = r2 ;
         else
            f.pend |= 1u << (9 + q) ;
      }
      if (f.pend) {
         w.frames.pop_back() ;
         par_convert(w, f) ;
         return 0 ;
      }
      r = find_node_par(w, f.r2[0], f.r2[1], f.r2[2], f.r2[3]) ;
   }
   w.frames.pop_back() ;
   par_complete(w, n, r) ;
   return r ;
}
/*
 *   Find work for a thread that asks: a sub-result the oldest computation
 *   has not reached (claimed here for the asking thread), else the oldest
 *   ready stage.  A thread in another cache domain only gets a sub-result
 *   of the oldest computation (the largest pieces of work), since every
 *   node it reads or makes crosses domains.
 */
int hlifealgo::par_give(hlworker &w, hlitem &it, int remote) {
   size_t nframes = w.frames.size() ;
   if (remote && nframes > 1)
      nframes = 1 ;
   for (size_t i=0; i<nframes; i++) {
      hlframe *f = w.frames[i] ;
      while (f->end > f->next) {
         int pos = --f->end ;
         node *c ;
         if (f->phase == 1)
            c = f->in1[parorder1[pos]] ;
         else
            c = f->in2[parorder2[pos]] ;
         node *z = 0 ;
         if (load_rlx(&c->res) == 0 && cas_ptr(&c->res, &z, (node *)1)) {
            it.p = c ;
            it.a = f->depth ;
            return 1 ;
         }
      }
   }
   if (w.rcount && !remote) {
      it = rq_pop_oldest(w) ;
      return 1 ;
   }
   return 0 ;
}
void hlifealgo::par_answer(hlworker &w) {
   int req = w.request.load(std::memory_order_acquire) ;
   hlworker &t = par->w[req - 1] ;
   if (par_give(w, t.reply, par->wdomain[t.id] != par->wdomain[w.id]))
      t.replied.store(1, std::memory_order_release) ;
   else
      t.replied.store(2, std::memory_order_release) ;
   w.request.store(0, std::memory_order_release) ;
}
void hlifealgo::par_run(hlworker &w, void *p, long a) {
   if (a >= 0)
      calc_h(w, (node *)p, (int)a) ;
   else if (a == -1 - PARSTAGEC)
      par_finish(w, (hljob *)p) ;
   else
      par_stage2(w, (hljob *)p, (int)(-1 - a)) ;
}
/*
 *   Ask a random working thread for work; 0 if it had none.  While waiting
 *   for the answer, keep answering requests and stopping for gc.
 */
int hlifealgo::par_steal(hlworker &w, hlitem &it) {
   if (par->n == 1)
      return 0 ;
   w.rng ^= w.rng << 13 ;
   w.rng ^= w.rng >> 17 ;
   w.rng ^= w.rng << 5 ;
   // mostly ask a thread that shares our cache; every PARREMOTE-th try in
   // a row, any thread, but only from a domain whose threads are all busy
   // (otherwise its own threads can take the work, without the cost of
   // moving it between domains)
   const std::vector<int> &mates = par->dworkers[par->wdomain[w.id]] ;
   int v ;
   if (mates.size() > 1 && (w.fails++ % PARREMOTE) != PARREMOTE - 1) {
      v = mates[w.rng % (unsigned int)(mates.size() - 1)] ;
      if (v == w.id)
         v = mates.back() ;
   } else {
      v = (int)(w.rng % (unsigned int)(par->n - 1)) ;
      if (v >= w.id)
         v++ ;
      if (par->wdomain[v] != par->wdomain[w.id]) {
         const std::vector<int> &theirs = par->dworkers[par->wdomain[v]] ;
         for (size_t k=0; k<theirs.size(); k++)
            if (!par->w[theirs[k]].busy.load(std::memory_order_relaxed))
               return 0 ;
      }
   }
   hlworker &vw = par->w[v] ;
   // look before writing: a failed compare-and-swap also takes the cache
   // line away from the thread that polls it
   if (!vw.busy.load(std::memory_order_relaxed) ||
       vw.request.load(std::memory_order_relaxed) != 0)
      return 0 ;
   int expected = 0 ;
   w.replied.store(0, std::memory_order_relaxed) ;
   if (!vw.request.compare_exchange_strong(expected, w.id + 1))
      return 0 ;
   int got ;
   while ((got = w.replied.load(std::memory_order_acquire)) == 0) {
      PAR_SAFEPOINT(w) ;
      PAR_ANSWER(w) ;
      cpu_relax() ;
   }
   if (got == 2)
      return 0 ;
   w.fails = 0 ;
   it = w.reply ;
   return 1 ;
}
/*
 *   Between pieces of work: run a ready stage of our own, or work from
 *   another thread.  Returns 0 if there was none.
 */
int hlifealgo::par_one(hlworker &w) {
   PAR_SAFEPOINT(w) ;
   PAR_ANSWER(w) ;
   hlitem it ;
   if (w.rcount) {
      it = rq_pop_newest(w) ;
   } else {
      if (w.busy.load(std::memory_order_relaxed))
         w.busy.store(0, std::memory_order_relaxed) ;
      if (!par_steal(w, it))
         return 0 ;
      w.busy.store(1, std::memory_order_relaxed) ;
   }
   par_run(w, it.p, it.a) ;
   return 1 ;
}
/*
 *   Close this thread's request slot (answering a pending request first),
 *   before it stops answering.
 */
void hlifealgo::par_close(hlworker &w) {
   for (;;) {
      int expected = 0 ;
      if (w.request.compare_exchange_strong(expected, PARCLOSED))
         return ;
      par_answer(w) ;
   }
}
/*
 *   Cache domains: groups of CPUs that share a last-level cache.  Passing
 *   nodes between threads in different domains is much slower than within
 *   one (on a 64-core EPYC with 8 cores per L3, 8 threads on one L3 ran
 *   hexadecimal 1.8x faster than 8 threads on 8 L3s).  So workers fill the
 *   domain worker 0 runs in first, then the next domains; each worker
 *   thread is restricted to its domain's CPUs (worker 0 only during a
 *   step), and asks threads of its own domain for work first (par_steal).  On Linux the domains come from
 *   sysfs (the CPUs allowed to this process, grouped by L3); elsewhere, or
 *   without that information, all workers are in one domain and no
 *   affinity is set.
 */
void hlifealgo::par_topology() {
   int n = par->n ;
   par->wdomain.assign(n, 0) ;
#ifdef __linux__
   cpu_set_t allowed ;
   std::vector<std::string> keys ;
   std::vector<std::vector<int> > doms ;
   if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
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
   }
   if (doms.size() > 1) {
      int c0 = sched_getcpu() ;
      for (size_t d=0; d<doms.size(); d++)
         for (size_t k=0; k<doms[d].size(); k++)
            if (doms[d][k] == c0)
               std::swap(doms[0], doms[d]) ;
      size_t ncpus = 0 ;
      for (size_t d=0; d<doms.size(); d++)
         ncpus += doms[d].size() ;
      for (int i=0; i<n; i++) {
         size_t k = i % ncpus, d = 0 ;
         while (k >= doms[d].size())
            k -= doms[d++].size() ;
         par->wdomain[i] = (int)d ;
      }
      par->dcpus = doms ;
   }
#endif
   int ndom = 1 ;
   for (int i=0; i<n; i++)
      if (par->wdomain[i] + 1 > ndom)
         ndom = par->wdomain[i] + 1 ;
   par->dworkers.assign(ndom, std::vector<int>()) ;
   for (int i=0; i<n; i++)
      par->dworkers[par->wdomain[i]].push_back(i) ;
}
void hlifealgo::par_worker(hlworker &w) {
#ifdef __linux__
   if (!par->dcpus.empty()) {
      cpu_set_t set ;
      CPU_ZERO(&set) ;
      const std::vector<int> &cpus = par->dcpus[par->wdomain[w.id]] ;
      for (size_t i=0; i<cpus.size(); i++)
         CPU_SET(cpus[i], &set) ;
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set) ;
   }
#endif
   int idle = 0 ;
   for (;;) {
      if (par->quit.load(std::memory_order_acquire))
         return ;
      if (par->stepactive.load(std::memory_order_acquire)) {
         par_enter(par, w) ;
         int backoff = PARBACKOFF ;
         while (par->stepactive.load(std::memory_order_relaxed)) {
            if (par_one(w)) {
               backoff = PARBACKOFF ;
            } else {
               for (int i=0; i<backoff; i++)
                  cpu_relax() ;
               if (backoff < PARBACKOFFMAX)
                  backoff *= 2 ;
            }
         }
         w.busy.store(0, std::memory_order_relaxed) ;
         w.active.store(0) ;
         idle = 0 ;
      } else if (++idle < PARSPIN) {
         PAR_ANSWER(w) ;
         cpu_relax() ;
      } else {
         par_close(w) ;
         {
            std::unique_lock<std::mutex> lk(par->sleepmx) ;
            par->sleepcv.wait(lk, [this] { return par->stepactive.load() || par->quit.load() ; }) ;
         }
         w.request.store(0) ;
         idle = 0 ;
      }
   }
}
/*
 *   Compute the result of the (pushed-out) root n at level depth on all
 *   threads.  Starts the threads on first use.
 */
node *hlifealgo::runpattern_par(node *n, int depth) {
   if (par == 0) {
      par = new hlpar(nthreads) ;
      par_topology() ;
      for (int i=1; i<par->n; i++)
         par->threads.push_back(std::thread(&hlifealgo::par_worker, this,
                                            std::ref(par->w[i]))) ;
   }
   par->cutoff = par_cutoff(depth) ;
   double tstart = std::chrono::duration<double>(
                      std::chrono::steady_clock::now().time_since_epoch()).count() ;
   if (idx.cap < INDEXMINPAR)
      index_grow(INDEXMINPAR) ;
   hlworker &w = par->w[0] ;
   node *r ;
   if (depth <= par->cutoff) {
      r = getres_in(w, n, depth) ;
   } else {
#ifdef __linux__
      // keep worker 0 in its domain during the step (it is the caller's
      // thread, so its own affinity is restored afterwards)
      cpu_set_t saved ;
      int restore = 0 ;
      if (!par->dcpus.empty() && sched_getaffinity(0, sizeof(saved), &saved) == 0) {
         cpu_set_t set ;
         CPU_ZERO(&set) ;
         const std::vector<int> &cpus = par->dcpus[par->wdomain[0]] ;
         for (size_t i=0; i<cpus.size(); i++)
            CPU_SET(cpus[i], &set) ;
         restore = sched_setaffinity(0, sizeof(set), &set) == 0 ;
      }
#endif
      par->rootres = 0 ;
      {
         std::lock_guard<std::mutex> lk(par->sleepmx) ;
         par->stepactive.store(1) ;
      }
      par->sleepcv.notify_all() ;
      par_enter(par, w) ;
      w.busy.store(1, std::memory_order_relaxed) ;
      w.request.store(0) ;
      r = getres_h(w, n, depth) ;
      if (r == 0)
         r = par_wait(n, &par->rootwait) ;
      while (r == 0) {
         if (!par_one(w))
            cpu_relax() ;
         r = load_acq(&par->rootres) ;
      }
      par_close(w) ;
      w.busy.store(0, std::memory_order_relaxed) ;
      w.active.store(0) ;
      par->stepactive.store(0) ;
#ifdef __linux__
      if (restore)
         sched_setaffinity(0, sizeof(saved), &saved) ;
#endif
   }
   // every piece of work finished before the root's result was stored
   for (int i=0; i<par->n; i++) {
      hlworker &v = par->w[i] ;
      par_flush(v) ;
      halvesdone += v.halves ;
      v.halves = 0 ;
   }
   if (halvesdone > 1000)
      halvesdone = 1000 ;
   par_cutoff_time(std::chrono::duration<double>(
                      std::chrono::steady_clock::now().time_since_epoch()).count() - tstart) ;
   return r ;
}
int hlifealgo::par_cutoff(int depth) {
   if (parcutoff > 0)
      return parcutoff < PARMINCUTOFF ? PARMINCUTOFF : parcutoff ;
   par->pcdepth = depth ;
   if (par->pcut == 0 || par->pcngens != ngens) {
      par->pcut = PARCUTOFF ;
      par->pcngens = ngens ;
      par->pcpos = 0 ;
      par->pcinterval = PCFIRST ;
   }
   if (par->pcpos < 0)
      return par->pcut ;
   int c = par->pcut + pcarm(par->pcpos) ;
   if (c < PARMINCUTOFF || c > depth)   // time the current cutoff instead
      c = par->pcut ;
   return c ;
}
void hlifealgo::par_cutoff_time(double t) {
   if (parcutoff > 0)
      return ;
   if (par->pcpos < 0) {
      if (--par->pcwait <= 0)
         par->pcpos = 0 ;
      return ;
   }
   par->pctime[par->pcpos++] = t ;
   if (par->pcpos < PCSEQ)
      return ;
   // r[0]: trials at c+1, r[1]: at c-1; a level wins if PCWINS of its
   // trial steps were faster than the steps around them, and on average
   // by PCGAIN (it must be a level the trials ran at)
   double r[2] = { 0, 0 } ;
   int wins[2] = { 0, 0 } ;
   for (int i=1; i<PCSEQ; i+=2) {
      double q = 2 * par->pctime[i] / (par->pctime[i-1] + par->pctime[i+1]) ;
      int a = pcarm(i) > 0 ? 0 : 1 ;
      r[a] += q / PCROUNDS ;
      wins[a] += q < 1 ;
   }
   int ok[2] = { par->pcut + 1 <= par->pcdepth, par->pcut - 1 >= PARMINCUTOFF } ;
   double up = r[0], down = r[1] ;
   int move = 0 ;
   for (int a=0; a<2; a++)
      if (ok[a] && wins[a] >= PCWINS && r[a] < PCGAIN &&
          (move == 0 || r[a] < r[move > 0 ? 0 : 1]))
         move = a == 0 ? 1 : -1 ;
   if (move) {
      par->pcut += move ;
      par->pcinterval = PCFIRST ;
   } else if (par->pcinterval < PCMAXWAIT)
      par->pcinterval *= 2 ;
   if (verbose) {
      sprintf(statusline, "Task cutoff %d (trials: %.3f up, %.3f down)", par->pcut, up, down) ;
      lifestatus(statusline) ;
   }
   par->pcwait = par->pcinterval ;
   par->pcpos = -1 ;
}
/*
 *   Called by do_gc(): mark the nodes of unfinished computations and jobs,
 *   and empty the threads' free lists (the sweep puts those nodes back on
 *   the global free list).
 */
void hlifealgo::par_gc_mark(node **a, int k, int invalidate) {
   for (int i=0; i<k; i++)
      if (a[i])
         gc_mark(a[i], invalidate) ;
}
void hlifealgo::par_gc_roots(int invalidate) {
   for (size_t c=0; c<par->jobchunks.size(); c++)
      for (int i=0; i<PARJOBCHUNK; i++) {
         hljob *j = par->jobchunks[c] + i ;
         if (j->live) {
            gc_mark(j->n, invalidate) ;
            par_gc_mark(j->in1, 9, invalidate) ;
            par_gc_mark(j->r1, 9, invalidate) ;
            par_gc_mark(j->in2, 4, invalidate) ;
            par_gc_mark(j->r2, 4, invalidate) ;
         }
      }
   for (int i=0; i<par->n; i++) {
      hlworker &w = par->w[i] ;
      w.freenodes = 0 ;
      for (size_t k=0; k<w.frames.size(); k++) {
         hlframe *f = w.frames[k] ;
         gc_mark(f->n, invalidate) ;
         par_gc_mark(f->in1, 9, invalidate) ;
         par_gc_mark(f->r1, 9, invalidate) ;
         par_gc_mark(f->in2, 4, invalidate) ;
         par_gc_mark(f->r2, 4, invalidate) ;
      }
   }
}
/*
 *   Garbage collection on all threads, run by worker 0 while the others
 *   are stopped.  The same work as do_gc(): mark every node reachable from
 *   the roots, clear the index, then put the marked nodes back into it and
 *   the others on the free list.  Each phase is a list of items (subtrees
 *   to mark, parts of the index to clear or rehash, node blocks to sweep);
 *   the stopped threads take items until none are left.
 */
static inline int par_try_mark(node *n) {
   if (marked(n))
      return 0 ;
   return !(__atomic_fetch_or((g_uintptr_t *)&n->next, (g_uintptr_t)1,
                              __ATOMIC_RELAXED) & 1) ;
}
/*
 *   Mark n and push its children and result (the nodes gc_mark() visits
 *   from n) on stack.
 */
static inline void par_gc_visit(node *n, std::vector<node *> &stack) {
   if (!par_try_mark(n) || !is_node(n))
      return ;
   stack.push_back(n->nw) ;
   stack.push_back(n->ne) ;
   stack.push_back(n->sw) ;
   stack.push_back(n->se) ;
   if (n->res && !ISPARBUSY(n->res))
      stack.push_back(n->res) ;
}
void hlifealgo::par_gc_work(hlworker &w, int gen) {
   int phase = par->gcphase ;
   for (;;) {
      long v = par->gcnext.load(std::memory_order_relaxed) ;
      long i ;
      for (;;) {
         i = v & 0xffffffffL ;
         if ((v >> 32) != gen || i >= par->gctotal.load())
            return ;
         if (par->gcnext.compare_exchange_weak(v, v + 1))
            break ;
      }
      if (phase == PARGC_MARK) {
         std::vector<node *> &stack = w.gcstack ;
         stack.push_back(par->gcitems[i]) ;
         while (!stack.empty()) {
            node *n = stack.back() ;
            stack.pop_back() ;
            par_gc_visit(n, stack) ;
         }
      } else if (phase == PARGC_CLEAR) {
         g_uintptr_t lo = par->gcpos[i], hi = lo + PARGCRANGE ;
         if (hi > idx.cap)
            hi = idx.cap ;
         memset(idx.e + lo, 0, (hi - lo) * sizeof(hlentry)) ;
      } else if (phase == PARGC_SWEEP) {
         // marked nodes go back into the index, the others onto this
         // thread's list of freed nodes
         node *b = par->gcitems[i] ;
         node *pp = b + 1 ;
         for (g_uintptr_t j=0; j<(g_uintptr_t)b->res; j++, pp++) {
            if (marked(pp)) {
               clearmark(pp) ;
               index_put_par(idx, pp) ;
               w.gckept++ ;
            } else {
               if (w.gcfree == 0)
                  w.gcfreelast = pp ;
               pp->next = w.gcfree ;
               w.gcfree = pp ;
            }
         }
      } else {
         g_uintptr_t lo = par->gcpos[i], hi = lo + PARGCRANGE ;
         if (hi > idx.cap)
            hi = idx.cap ;
         index_move<1>(idx, lo, hi, par->newidx) ;
      }
      par->gcdone.fetch_add(1, std::memory_order_release) ;
   }
}
void hlifealgo::par_gc_phase(hlworker &w, int phase, long total) {
   int gen = par->gcgen.load(std::memory_order_relaxed) + 1 ;
   // gcnext gets the new generation before gctotal the new total: a thread
   // still taking items of the last phase (it found none left) that sees
   // the new total then fails its compare-and-swap on gcnext, and does not
   // take an item past the end of the last phase
   par->gcnext.store((long)gen << 32) ;
   par->gctotal.store(total) ;
   par->gcdone.store(0) ;
   par->gcphase = phase ;
   par->gcgen.store(gen, std::memory_order_release) ;
   w.gcseen = gen ;
   par_gc_work(w, gen) ;
   while (par->gcdone.load(std::memory_order_acquire) < total)
      cpu_relax() ;
}
void hlifealgo::par_gc(hlworker &w) {
   inGC = 1 ;
   gccount++ ;
   gcstep++ ;
   if (verbose) {
     if (gcstep > 1)
       sprintf(statusline, "GC #%d(%d)", gccount, gcstep) ;
     else
       sprintf(statusline, "GC #%d", gccount) ;
     lifestatus(statusline) ;
   }
   // the roots: as do_gc(), plus the nodes of unfinished computations and jobs
   std::vector<node *> roots ;
   int i ;
   for (i=nzeros-1; i>=0; i--)
      if (zeronodea[i] != 0)
         break ;
   if (i >= 0)
      roots.push_back(zeronodea[i]) ;
   if (root != 0)
      roots.push_back(root) ;
   for (i=0; i<gsp; i++)
      roots.push_back(stack[i]) ;
   for (i=0; i<timeline.framecount; i++)
      roots.push_back((node *)timeline.frames[i]) ;
   for (size_t c=0; c<par->jobchunks.size(); c++)
      for (int k=0; k<PARJOBCHUNK; k++) {
         hljob *j = par->jobchunks[c] + k ;
         if (j->live) {
            roots.push_back(j->n) ;
            par_gc_roots_of(roots, j->in1, 9) ;
            par_gc_roots_of(roots, j->r1, 9) ;
            par_gc_roots_of(roots, j->in2, 4) ;
            par_gc_roots_of(roots, j->r2, 4) ;
         }
      }
   for (int t=0; t<par->n; t++) {
      hlworker &v = par->w[t] ;
      for (size_t k=0; k<v.frames.size(); k++) {
         hlframe *f = v.frames[k] ;
         roots.push_back(f->n) ;
         par_gc_roots_of(roots, f->in1, 9) ;
         par_gc_roots_of(roots, f->r1, 9) ;
         par_gc_roots_of(roots, f->in2, 4) ;
         par_gc_roots_of(roots, f->r2, 4) ;
      }
   }
   // mark the top of the graph here until there are enough subtrees to share
   std::vector<node *> next ;
   size_t want = 64 * (size_t)par->n ;
   while (!roots.empty() && roots.size() < want) {
      next.clear() ;
      for (size_t k=0; k<roots.size(); k++)
         par_gc_visit(roots[k], next) ;
      roots.swap(next) ;
   }
   par->gcitems.swap(roots) ;
   par_gc_phase(w, PARGC_MARK, (long)par->gcitems.size()) ;
   // clear the index, then sweep the node blocks
   par->gcpos.clear() ;
   for (g_uintptr_t j=0; j<idx.cap; j+=PARGCRANGE)
      par->gcpos.push_back(j) ;
   par_gc_phase(w, PARGC_CLEAR, (long)par->gcpos.size()) ;
   par->gcitems.clear() ;
   for (node *b=nodeblocks; b; b=b->next)
      par->gcitems.push_back(b) ;
   for (int t=0; t<par->n; t++) {
      hlworker &v = par->w[t] ;
      v.gcfree = v.gcfreelast = 0 ;
      v.gckept = 0 ;
   }
   par_gc_phase(w, PARGC_SWEEP, (long)par->gcitems.size()) ;
   par->gcitems.clear() ;
   g_uintptr_t oldpop = hashpop ;
   hashpop = 0 ;
   freenodes = 0 ;
   for (int t=0; t<par->n; t++) {
      hlworker &v = par->w[t] ;
      hashpop += v.gckept ;
      if (v.gcfree) {
         v.gcfreelast->next = freenodes ;
         freenodes = v.gcfree ;
      }
      v.freenodes = 0 ;
   }
   idx.used = hashpop ;
   g_uintptr_t freed_nodes = oldpop > hashpop ? oldpop - hashpop : 0 ;
   inGC = 0 ;
   if (verbose) {
     double perc = (double)freed_nodes / (double)(hashpop + freed_nodes) * 100.0 ;
     sprintf(statusline+strlen(statusline), " freed %g percent (%" PRIuPTR ").",
                                                   perc, freed_nodes) ;
     lifestatus(statusline) ;
   }
   if (needPop) {
      calcPopulation() ;
      popValid = 1 ;
      needPop = 0 ;
      poller->updatePop() ;
   }
}
void hlifealgo::par_gc_roots_of(std::vector<node *> &roots, node **a, int k) {
   for (int i=0; i<k; i++)
      if (a[i])
         roots.push_back(a[i]) ;
}
void hlifealgo::par_stop() {
   if (par == 0)
      return ;
   {
      std::lock_guard<std::mutex> lk(par->sleepmx) ;
      par->quit.store(1) ;
   }
   par->sleepcv.notify_all() ;
   for (size_t i=0; i<par->threads.size(); i++)
      par->threads[i].join() ;
   for (size_t c=0; c<par->jobchunks.size(); c++)
      delete [] par->jobchunks[c] ;
   delete [] par->w ;
   delete par ;
   par = 0 ;
}
