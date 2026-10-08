// This file is part of Golly.
// See docs/License.html for the copyright notice.

#ifndef HLIFEALGO_H
#define HLIFEALGO_H
#include "lifealgo.h"
#include "liferules.h"
#include "util.h"
#include <vector>
#include <atomic>
/*
 *   Into instances of this node structure is where almost all of the
 *   memory allocated by this program goes.  Thus, it is imperative we
 *   keep it as small as possible so we can explore patterns as large
 *   and as deep as possible.
 *
 *   But first, how does this program even work?  Well, there are
 *   two major tricks that are used.
 *
 *   The first trick is to represent the 2D space `symbolically'
 *   (in the sense that a binary decision diagram is a symbolic
 *   representation of a boolean predicate).  This can be thought
 *   of as a sort of compression.  We break up space into a grid of
 *   squares, each containing 8x8 cells.  And we `canonicalize'
 *   each square; that is, all the squares with no cells set are
 *   represented by a single actual instance of an empty square;
 *   all squares with only the upper-left-most cell set are
 *   represented by yet another instance, and so on.  A single pointer
 *   to the single instance of each square takes less space than
 *   representing the actual cell bits themselves.
 *
 *   Where do we store these pointers?  At first, one might envision
 *   a large two-dimensional array of pointers, each one pointing
 *   to one of the square instances.  But instead, we group the
 *   squares (we'll call them 8-squares) into larger squares 16
 *   cells on a side; these are 16-squares.  Each 16-square contains
 *   four 8-squares, so each 16-square is represented by four
 *   pointers, each to an 8-square.  And we canonicalize these as
 *   well, so for a particular set of values for a 16 by 16 array
 *   of cells, we'll only have a single 16-square.
 *
 *   And so on up; we canonicalize 32-squares out of 16-squares, and
 *   on up to some limit.  Now the limit need not be very large;
 *   having just 20 levels of nodes gives us a universe that is
 *   4 * 2**20 or about 4M cells on a side.  Having 100 levels of
 *   nodes (easily within the limits of this program) gives us a
 *   universe that is 4 * 2**100 or about 5E30 cells on a side.
 *   I've run universes that expand well beyond 1E50 on a side with
 *   this program.
 *
 *   [A nice thing about this representation is that there are no
 *   coordinate values anywhere, which means that there are no
 *   limits to the coordinate values or complex multidimensional
 *   arithmetic needed.]
 *
 *   [Note that this structure so far is very similar to the octtrees
 *   used in 3D simulation and rendering programs.  It's different,
 *   however, in that we canonicalize the nodes, and also, of course,
 *   in that it is 2D rather than 3D.]
 *
 *   I mentioned there were two tricks, and that's only the first.
 *   The second trick is to cache the `results' of the LIFE calculation,
 *   but in a way that looks ahead farther in time as you go higher
 *   in the tree, much like the tree nodes themselves scan larger
 *   distances in space.  This trick is just a little subtle, but it
 *   is where the bulk of the power of the program comes from.
 *
 *   Consider once again the 8-squares.  We want to cache the result
 *   of executing LIFE on that area.  We could cache the result of
 *   looking ahead just one generation; that would yield a 6x6 square.
 *   (Note that we cannot calculate an 8-square, because we are
 *   using the single instance of the 8-square to represent all the
 *   different places that 8x8 arrangement occurs, and those different
 *   places might be surrounded by different border cells.  But we
 *   can say for sure that the central 6-square will evolve in a
 *   unique way in the next generation.)
 *
 *   We could also calculate the 4-square that is two generations
 *   hence, and the 3-square that is three generations hence, and
 *   the 2-square that is four generations hence.  We choose the
 *   4-square that is two generations hence; why will be clear in
 *   a moment.
 *
 *   Now let's consider the 16-square.  We would like to look farther
 *   ahead for this square (if we always only looked two generations
 *   ahead, our runtime would be at *least* linear in the number of
 *   generations, and we want to beat that.)  So let's look 4 generations
 *   ahead, and cache the resulting 8-square.  So we do.
 *
 *   Where do we cache the results?  Well, we cache the results in the
 *   same node structure we are using to store the pointers to the
 *   smaller squares themselves.  And since we're hashing them all
 *   together, we want a next pointer for the hash chain.  Put all of
 *   this together, and you get the following structure for the 16-squares
 *   and larger:
 */
struct node {
   node *next ;              /* hash link */
   node *nw, *ne, *sw, *se ; /* constant; nw != 0 means nonleaf */
   node *res ;               /* cache */
} ;
/*
 *   For the 8-squares, we do not have `children', we have actual data
 *   values.  We still break up the 8-square into 4-squares, but the
 *   4-squares only have 16 cells in them, so we represent them directly
 *   by an unsigned short (in this case, the direct value itself takes
 *   less memory than the pointer we might replace it with).
 *
 *   One minor trick about the following structure.  We did lie above
 *   somewhat; sometimes the struct node * points to an actual struct
 *   node, and sometimes it points to a struct leaf.  So we need a way
 *   to tell if the thing we are pointing at is a node or a leaf.  We
 *   could add another bit to the node structure, but this would grow
 *   it, and we want it to stay as small as possible.  Now, notice
 *   that, in all valid struct nodes, all four pointers (nw, ne, sw,
 *   and se) must contain a live non-zero value.  We simply ensure
 *   that the struct leaf contains a small value (0, or 1 for a leaf in
 *   the hash table, see SLOTLEAF in hlifealgo.cpp) where the first (nw)
 *   pointer field would be in a struct node.
 *
 *   Each short represents a 4-square in normal, left-to-right then top-down
 *   order from the most significant bit.  So bit 0x8000 is the upper
 *   left (or northwest) bit, and bit 0x1000 is the upper right bit, and
 *   so on.
 */
struct leaf {
   node *next ;              /* gc mark bit */
   node *isnode ;            /* 0, or SLOTLEAF when hashed */
   unsigned short nw, ne, sw, se ;  /* constant */
   bigint leafpop ;         /* how many set bits */
   unsigned short res1, res2 ;      /* constant */
   g_uintptr_t wnum ;       /* cell number while writing macrocell */
} ;
/*
 *   If it is a struct node, this returns a non-zero value, otherwise it
 *   returns a zero value.
 */
#define is_node(n) (((g_uintptr_t)(((node *)(n))->nw)) > 3)
/*
 *   The hash index (see "The hash table" in hlifealgo.cpp).
 */
struct hlindex {
   unsigned long long *e ;   // entries: 0 (empty), or a node pointer and tag
   g_uintptr_t cap ;         // entries
   g_uintptr_t used ;        // entries not empty
   g_uintptr_t limit ;       // used that makes the index grow
} ;
/*
 *   For explicit prefetching we retain some state on our lookup
 *   calculations.
 */
#ifdef USEPREFETCH
struct setup_t { 
   g_uintptr_t h ;
   struct node *nw, *ne, *sw, *se ;
   void prefetch(node **addr) const { PREFETCH(addr) ; }
} ;
#endif
struct hlpar ;
struct hlworker ;
struct hljob ;
struct hlwait ;
struct hlframe ;
struct hlitem ;
/**
 *   Our hlifealgo class.
 */
class hlifealgo : public lifealgo {
public:
   hlifealgo() ;
   virtual ~hlifealgo() ;
   virtual int setcell(int x, int y, int newstate) ;
   virtual int getcell(int x, int y) ;
   virtual int nextcell(int x, int y, int &state) ;
   virtual void endofpattern() ;
   virtual void setIncrement(bigint inc) ;
   virtual void setIncrement(int inc) { setIncrement(bigint(inc)) ; }
   virtual void setGeneration(bigint gen) { generation = gen ; }
   virtual const bigint &getPopulation() ;
   virtual int isEmpty() ;
   virtual int hyperCapable() { return 1 ; }
   virtual void setMaxMemory(int m) ;
   virtual int getMaxMemory() { return (int)(maxmem >> 20) ; }
   virtual const char *setrule(const char *s) ;
   virtual const char *getrule() { return hliferules.getrule() ; }
   virtual void step() ;
   virtual void* getcurrentstate() { return root ; }
   virtual void setcurrentstate(void *n) ;
   /*
    *   The contract of draw() is that it render every pixel in the
    *   viewport precisely once.  This allows us to eliminate all
    *   flashing.  Later we'll make this be damage-specific.
    */
   virtual void draw(viewport &view, liferender &renderer) ;
   virtual void fit(viewport &view, int force) ;
   virtual void lowerRightPixel(bigint &x, bigint &y, int mag) ;
   virtual void findedges(bigint *t, bigint *l, bigint *b, bigint *r) ;
   virtual const char *readmacrocell(char *line) ;
   virtual const char *writeNativeFormat(std::ostream &os, char *comments) ;
   static void doInitializeAlgoInfo(staticAlgoInfo &) ;
   static void doInitializeParAlgoInfo(staticAlgoInfo &) ;
   /*
    *   Parallel stepping.  nthreads is this universe's thread count; it
    *   starts as numthreads, or parthreads for a universe made as
    *   "HashLife Parallel".  0 uses the serial code; N >= 1 uses
    *   the parallel code with N threads.  Results of nodes at level
    *   parcutoff or above (2^parcutoff cells wide; at least 5) are computed
    *   as separate tasks; smaller ones run inside the task that needs them.
    *   parcutoff 0 chooses the level by timing steps (see hlifealgo.cpp).
    */
   int nthreads ;
   static int numthreads ;
   static int parthreads ;
   static int parcutoff ;
private:
/*
 *   Some globals representing our universe.  The root is the
 *   real root of the universe, and the depth is the depth of the
 *   tree where 2 means that root is a leaf, and 3 means that the
 *   children of root are leaves, and so on.  The center of the
 *   root is always coordinate position (0,0), so at startup the
 *   x and y coordinates range from -4..3; in general,
 *   -(2**depth)..(2**depth)-1.  The zeronodea is an
 *   array of canonical `empty-space' nodes at various depths.
 *   The ngens is an input parameter which is the second power of
 *   the number of generations to run.
 */
   node *root ;
   int depth ;
   node **zeronodea ;
   int nzeros ;
/*
 *   Finally, our gc routine.  We keep a `stack' of all the `roots'
 *   we want to preserve.  Nodes not reachable from here, we allow to
 *   be freed.  Same with leaves.
 */
   node **stack ;
   int stacksize ;
   g_uintptr_t hashpop ;
   static double maxloadfactor ;
   hlindex idx ;
   int halvesdone ;
   int gsp ;
   g_uintptr_t alloced, maxmem ;
   node *freenodes ;
   int okaytogc ;
   g_uintptr_t totalthings ;
   node *nodeblocks ;
   char *regions, *regionnext, *regionend ;  // see newblock()
   char *ruletable ;
   int leafnk ;                        // see setleafrule()
   unsigned char leafk[9] ;
   unsigned short leafbm[9], leafsm[9] ;
   bigint population ;
   bigint setincrement ;
   bigint pow2step ; // greatest power of two in increment
   int nonpow2 ; // increment / pow2step
   int ngens ; // log2(pow2step)
   int popValid, needPop, inGC ;
   /*
    *   When rendering we store the relevant bits here rather than
    *   passing them deep into recursive subroutines.
    */
   liferender *renderer ;
   viewport *view ;
   int uviewh, uvieww, viewh, vieww, mag, pmag ;
   int llbits, llsize ;
   char *llxb, *llyb ;
   int hashed ;
   int cacheinvalid ;
   g_uintptr_t cellcounter ; // used when writing
   g_uintptr_t writecells ; // how many to write
   int gccount ; // how many gcs total this pattern
   int gcstep ; // how many gcs this step
   hperf running_hperf, step_hperf, inc_hperf ;
   int softinterrupt ;
   static char statusline[] ;
//
   void leafres(leaf *n) ;
   void setleafrule() ;
   void leafstep(const leaf *n, const leaf *ne, const leaf *t, const leaf *e,
                 int gens, unsigned short *q) const ;
   void index_alloc(hlindex &x, g_uintptr_t cap) ;
   void index_put(hlindex &x, node *p) ;
   void index_full() ;
   void index_grow(g_uintptr_t cap) ;
   node *find_node_h(g_uintptr_t h, node *nw, node *ne, node *sw, node *se) ;
   node *find_node_new(node *nw, node *ne, node *sw, node *se) ;
   node *find_node(node *nw, node *ne, node *sw, node *se) ;
#ifdef USEPREFETCH
   node *find_node(setup_t &su) ;
   void setupprefetch(setup_t &su, node *nw, node *ne, node *sw, node *se) ;
#endif
   void unhash_node(node *n) ;
   void unhash_node2(node *n) ;
   void rehash_node(node *n) ;
   leaf *find_leaf(unsigned short nw, unsigned short ne,
                   unsigned short sw, unsigned short se) ;
   leaf *find_leaf_new(unsigned short nw, unsigned short ne,
                       unsigned short sw, unsigned short se) ;
   node *getres(node *n, int depth) ;
   node *dorecurs(node *n, node *ne, node *t, node *e, int depth) ;
   node *dorecurs_half(node *n, node *ne, node *t, node *e, int depth) ;
   leaf *dorecurs_leaf(leaf *n, leaf *ne, leaf *t, leaf *e) ;
   leaf *dorecurs_leaf_half(leaf *n, leaf *ne, leaf *t, leaf *e) ;
   leaf *dorecurs_leaf_quarter(leaf *n, leaf *ne, leaf *t, leaf *e) ;
   node *newnode() ;
   node *newblock() ;
   leaf *newleaf() ;
   node *newclearednode() ;
   leaf *newclearedleaf() ;
   void pushroot_1() ;
   int node_depth(node *n) ;
   node *zeronode(int depth) ;
   node *pushroot(node *n) ;
   node *make_internal_node(node *n);
   node *gsetbit(node *n, int x, int y, int newstate, int depth) ;
   int getbit(node *n, int x, int y, int depth) ;
   int nextbit(node *n, int x, int y, int depth) ;
   node *hashpattern(node *root, int depth) ;
   node *popzeros(node *n) ;
   const bigint &calcpop(node *root, int depth) ;
   void aftercalcpop2(node *root, int depth) ;
   void afterwritemc(node *root, int depth) ;
   void calcPopulation() ;
   node *save(node *n) ;
   node *savegrow(node *n) ;
   void pop(int n) ;
   void clearstack() ;
   void clearcache() ;
   void gc_mark(node *root, int invalidate) ;
   void do_gc(int invalidate) ;
   void clearcache(node *n, int depth, int clearto) ;
   void clearcache_p1(node *n, int depth, int clearto) ;
   void clearcache_p2(node *n, int depth, int clearto) ;
   void new_ngens(int newval) ;
   int log2(unsigned int n) ;
   node *runpattern() ;
   // parallel stepping (see that section in hlifealgo.cpp)
   hlpar *par ;
   node *runpattern_par(node *n, int depth) ;
   int par_cutoff(int depth) ;
   void par_cutoff_time(double t) ;
   void par_gc_mark(node **a, int k, int invalidate) ;
   void par_gc_roots(int invalidate) ;
   void par_gc(hlworker &w) ;
   void par_gc_phase(hlworker &w, int phase, long total) ;
   void par_gc_work(hlworker &w, int gen) ;
   void par_gc_roots_of(std::vector<node *> &roots, node **a, int k) ;
   void par_stop() ;
   void par_topology() ;
   void par_worker(hlworker &w) ;
   void par_attention(hlworker &w) ;
   int par_one(hlworker &w) ;
   int par_steal(hlworker &w, hlitem &it) ;
   int par_give(hlworker &w, hlitem &it, int remote) ;
   void par_answer(hlworker &w) ;
   void par_close(hlworker &w) ;
   void par_run(hlworker &w, void *p, long a) ;
   node *getres_h(hlworker &w, node *n, int depth) ;
   node *calc_h(hlworker &w, node *n, int depth) ;
   void par_convert(hlworker &w, hlframe &f) ;
   void par_stage2(hlworker &w, hljob *j, int q) ;
   void par_finish(hlworker &w, hljob *j) ;
   void par_complete(hlworker &w, node *c, node *r) ;
   void par_deliver(hlworker &w, hlwait *wt, node *r) ;
   hljob *par_job_alloc(hlworker &w) ;
   void par_count_nodes(hlworker &w) ;
   void par_flush(hlworker &w) ;
   node *par_newnode(hlworker &w) ;
   void par_index_full(hlworker &w) ;
   void par_index_grow(hlworker &w) ;
   void par_rehash(hlworker &w, g_uintptr_t ncap) ;
   node *find_node_par(hlworker &w, g_uintptr_t h,
                       node *nw, node *ne, node *sw, node *se) ;
   node *find_node_par(hlworker &w, node *nw, node *ne, node *sw, node *se) ;
   leaf *find_leaf_par(hlworker &w, unsigned short nw, unsigned short ne,
                       unsigned short sw, unsigned short se) ;
   node *getres_in(hlworker &w, node *n, int depth) ;
   node *calc_in(hlworker &w, node *n, int depth) ;
   node *dorecurs_in(hlworker &w, node *n, node *ne, node *t, node *e, int depth) ;
   node *dorecurs_half_in(hlworker &w, node *n, node *ne, node *t, node *e, int depth) ;
   leaf *dorecurs_leaf_in(hlworker &w, leaf *n, leaf *ne, leaf *t, leaf *e) ;
   leaf *dorecurs_leaf_half_in(hlworker &w, leaf *n, leaf *ne, leaf *t, leaf *e) ;
   leaf *dorecurs_leaf_quarter_in(hlworker &w, leaf *n, leaf *ne, leaf *t, leaf *e) ;
   void renderbm(int x, int y) ;
   void fill_ll(int d) ;
   void drawnode(node *n, int llx, int lly, int depth, node *z) ;
   void ensure_hashed() ;
   g_uintptr_t writecell(std::ostream &os, node *root, int depth) ;
   g_uintptr_t writecell_2p1(node *root, int depth) ;
   g_uintptr_t writecell_2p2(std::ostream &os, node *root, int depth) ;
   void unpack8x8(unsigned short nw, unsigned short ne,
                  unsigned short sw, unsigned short se,
                  unsigned int *top, unsigned int *bot) ;
   liferules hliferules ;
} ;
#endif
