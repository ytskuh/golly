// This file is part of Golly.
// See docs/License.html for the copyright notice.

/**
 *   This is the fast "conventional" life algorithm code.
 */
#ifndef QLIFEALGO_H
#define QLIFEALGO_H
#include "lifealgo.h"
#include "liferules.h"
#include <vector>
#include <atomic>
struct qlpar ;
struct qlgpu ;
struct qlgbrick ;
/*
 *   An outer-totalistic Moore rule for the vector brick kernel (see
 *   qlifealgo.cpp): with T the number of live cells in a 3x3 block
 *   (center included), a dead center is born if mb[T] and a live center
 *   survives if ms[T] (masks are 0 or all ones).
 */
struct qlvrule {
   int life ;                    // B3/S23
   unsigned int mb[10], ms[10] ;
} ;
typedef int (*qlkernel)(unsigned int *b, unsigned int r,
                        const unsigned int *u, unsigned int ur,
                        int recomp, unsigned int df, const qlvrule *ru) ;
/*
 *   The smallest unit of the universe is the `slice', which is a
 *   4 (horizontal) by 8 (vertical) chunk of the world.  Each slice
 *   is stored in a 32-bit word.  The most significant bit of the
 *   word is the upper left bit; the remaining bits go across
 *   horizontally then down vertically.
 *
 *   We always recompute in units of a slice.  This is smaller than the
 *   8x8 in some programs, and in this particular case adds to the
 *   efficiency.
 *
 *   This arrangment of bits was selected to minimize the bit
 *   twiddling needed for lookup; in general, we use a lookup table
 *   that takes a 4x4 current-generation and figures out the values
 *   of the inner 2x2 cells.
 *
 *   A `brick' is composed of eight slices for the even generation and
 *   eight slices for the odd generation.  This is a total of 16
 *   32-bit words.  The slices are concatenated horizontally so each
 *   brick contains two generations of a 32x8 section of the universe.
 *
 *   No other data is stored directly into the bricks; various flags are
 *   stored in the next level up.  We try and minimize the size of the
 *   data for a given universe so that we can minimize the number of
 *   data cache misses and TLB misses.  For almost all universes, the
 *   bulk of the allocated data is in these 64-byte bricks.
 *
 *   We use a stagger-step type algorithm to minimize the number of
 *   neighbors that need to be examined (thanks, Alan!) for each slice.
 *   Thus, a brick that holds (0,0)-(31,7) for the even generation would
 *   hold (1,1)-(32,8) in the odd generations.
 */
struct brick { /* 64 bytes */
   unsigned int d[16] ;
} ;
/*
 *   A tile contains the flags for four bricks, and pointers to the bricks
 *   themselves.  These pointers are never null; if the portion of the
 *   universe is completely empty, they point to a unique allocated
 *   `emptybrick'.  This means that we do not need to check if the pointers
 *   are null; we can simply always dereference and get valid data.  Thus,
 *   each tile corresponds to a 32x32 section of the universe.
 *
 *   The c flags indicate that, in the previous computation,
 *   the data was found to change in some slice or
 *   subslice.  The least significant bit in each c01 flag corresponds
 *   to the last (8th) slice for phase 0->1, and to the first slice for
 *   phase 1->0.
 *
 *   There are ten valid bits in c flags 1-4, one for each slice,
 *   one most significant one that indicates that the rightmost (leftmost)
 *   two columns of the first (eighth) slice changed in phase 0->1 (1->0)
 *   (and thus, the next brick to the left (right) needs to be recomputed).
 *   The tenth bit is just the ninth bit from the previous generation.
 *
 *   For c flags 0 and 5, we only have the first nine bits as above.
 *
 *   We need six flags; four for the slices themselves, and one indicating
 *   that the top two rows of a particular slice has changed, and thus
 *   the next brick up needs to be recomputed, and one indicating that the
 *   bottom two rows of a particular slice has changed.
 *
 *   In all cases, tiles only contain data about what has happened completely
 *   within their borders; indicating that the next slice over needs to be
 *   recomputed is handled by always inspecting the neighboring slices
 *   *before* a recomputation.
 *
 *   The flags int is divided into two 12-bit fields, each holding a population
 *   count, and one eight-bit field, holding dirty flags.
 *
 *   Note that tiles only point `down' to bricks, never to each other or to
 *   higher-level supertiles.
 *
 *   Tiles are numbered as level `0' of the universe tree.
 *
 *   The tiles are 32 bytes each; they can hold up to four bricks, so the
 *   memory consumption of the tiles tends to be small.
 */
// Tiles and supertiles take whole cache lines, so that threads computing
// different parts of the universe never write the same line (see
// "Parallel generations" in qlifealgo.cpp).
struct alignas(64) tile { /* 64 bytes */
   struct brick *b[4] ;
   short c[6] ;
   int flags, localdeltaforward ;
   int frozen ;         // part of a shadow that is not computed (see qlifealgo.cpp)
} ;
/*
 *   Supertiles hold pointers to eight subtiles, which can either be 
 *   tiles or supertiles themselves.  Supertiles are numbered as levels `1' on
 *   up of the universe tree.
 *
 *   Odd levels stack their subtiles horizontally; even levels stack their
 *   subtiles vertically.  Thus, each supertile at level 1 corresponds to
 *   a 256x32 section of the universe.  Each supertile at level 2 corresponds
 *   to a 256x256 section of the universe.  And so on.
 *
 *   Levels are never skipped (that is, each pointer in a supertile at
 *   level `n' points to a valid supertile or tile at level `n-1').
 *
 *   The c flag indicates that changes have happened in the subtiles.
 *   The least significant eight bits indicate that changes have occurred
 *   anywhere in the subtile; the least significant bit (bit 0) is associated
 *   with the first subtile for phase 0->1 and the eighth subtile for phase
 *   1->0, and bit 7 is associated with other.  Bit number 8 indicates that
 *   some bits on the right (left) edge of the eighth (first) subtile have
 *   changed for phase 0->1 (1->0).  Bits numbered 9 through 16 indicate that
 *   bits on the bottom (top) of the corresponding subtile have changed, and
 *   bit number 17 indicates that a bit in the lower right (upper left) corner
 *   of the eighth (first) subtile has changed.
 *
 *   Bit 18 through 27 correspond to the previous generation's bits 8 through
 *   17.  Bits 28 through 31 are the dirty bits; we currently use three of
 *   them.
 *
 *   The above description corresponds to odd levels.  For even levels,
 *   since tiles are stacked vertically instead of horizontally, change
 *   `lower' to `right' and `right' to `lower'.
 *
 *   The dirty flags again are used to indicate that this tile needed to
 *   be recomputed since the last time the dirty bit was cleared.
 *
 *   The two pop values hold a population count, if the appropriate dirty
 *   bit is clear.  These counts will never hold values greater than
 *   500M, so that a sum of the eight in unsigned arithmetic is guaranteed
 *   to never overflow.
 *
 *   This completes the universe data structures.  Note that there is no
 *   limit (other than the size of a pointer and memory constraints) on
 *   the size of the universe.  For instance, using these data structures
 *   we can easily build a universe with elements separated by 2^200
 *   pixels.
 *
 *   The supertiles are 44 bytes each; they correspond to at least a
 *   256x32 chunk of the universe, so the total memory consumption due to
 *   supertiles tends to be small.
 */
struct alignas(128) supertile { /* 128 bytes */
   struct supertile *d[8] ;
   int flags ;
   int pop[2] ;
   // used by parallel generations (see qlifealgo.cpp); amark is -1 in
   // frozen parts of shadows
   int amark, reg ;
   float cost ;
} ;
/*
 *   This is a common header for chunks of memory linked together.
 */
struct linkedmem {
   struct linkedmem *next ;
} ;
/*
 *   This structure contains all of our variables that pertain to a
 *   particular universe.  (Thus, we support multiple universes.)
 *
 *   The minx, miny, maxx, and maxy values describe the borders of
 *   the universe dictated by the level of the root supertile.  If
 *   ever during computation these bounds are found to be too tight,
 *   another supertile is added on top of the root, expanding these
 *   bounds.
 *
 *   These bounds currently limit the size of the universe to 2^32 in
 *   each direction; however, they are only used during the set call
 *   (when initially setting up the universe).  Changing them to
 *   doubles or 64-bit ints will relax this limitation.  For now we
 *   leave them as is.
 *
 *   The rootlev variable contains the supertile level of the root node.
 *
 *   The tilelist, supertilelist, and bricklist are freelists for the
 *   appropriate type of structure.
 *
 *   The memused is a linked list of all memory blocks allocated; this
 *   enables us to free the universe and all of its memory without actually
 *   walking the entire life tree.
 *
 *   The emptybrick pointer points to the unique brick that is guaranteed
 *   to always be empty.  The emptytile pointer is similar.
 *
 *   Finally, root is the top of the current life tree.  Nullroot is the
 *   topmost empty supertile allocated, and nullroots[] holds the empty
 *   supertiles at each level.  Setting this to 40 limits the number of
 *   levels to 40, which is sufficient for a 2^65x2^62 universe.
 */
const int QLMODES = 5 ;   // modes of parallel generations (see qlifealgo::parpick)
class qlifealgo : public lifealgo {
public:
   qlifealgo() ;
   virtual ~qlifealgo() ;
   virtual int setcell(int x, int y, int newstate) ;
   virtual int getcell(int x, int y) ;
   virtual int nextcell(int x, int y, int &v) ;
   // call after setcell calls
   virtual void endofpattern() {
     poller->bailIfCalculating() ;
     popValid = 0 ;
     gpuvalid = 0 ;
   }
   virtual void setIncrement(bigint inc) { increment = inc ; }
   virtual void setIncrement(int inc) { increment = inc ; }
   virtual void setGeneration(bigint gen) {
      if (gpuahead) gpusync() ;
      gpuvalid = 0 ;
      generation = gen ;
   }
   virtual const bigint &getPopulation() ;
   virtual int isEmpty() ;
   // can we do the gen count doubling? only hashlife
   virtual int hyperCapable() { return 0 ; }
   virtual void setMaxMemory(int m) ;
   virtual int getMaxMemory() { return (int)(maxmemory >> 20) ; }
   virtual const char *setrule(const char *s) ;
   virtual const char *getrule() { return qliferules.getrule() ; }
   virtual void step() ;
   virtual void* getcurrentstate() { return 0 ; }
   virtual void setcurrentstate(void *) {}
   virtual void draw(viewport &view, liferender &renderer) ;
   virtual void fit(viewport &view, int force) ;
   virtual void lowerRightPixel(bigint &x, bigint &y, int mag) ;
   virtual void findedges(bigint *t, bigint *l, bigint *b, bigint *r) ;
   virtual const char *writeNativeFormat(std::ostream &, char *) {
      return "No native format for qlifealgo yet." ;
   }
   static void doInitializeAlgoInfo(staticAlgoInfo &) ;
   static void doInitializeParAlgoInfo(staticAlgoInfo &) ;
   static void doInitializeCudaAlgoInfo(staticAlgoInfo &) ;
   // a universe made as "QuickLife CUDA": generations run on the GPU
   // when the rule and grid allow (see "GPU generations" in qlifealgo.cpp)
   int usegpu ;
   // parallel generations: nthreads > 1 uses that many threads.  It starts
   // as numthreads, or parthreads for a universe made as "QuickLife Parallel"
   int nthreads ;
   static int numthreads ;
   static int parthreads ;
   void parrun(int id) ;   // used by the worker threads
private:
   qlgpu *gpu ;
   int gpufailed ;      // no usable GPU
   int gpuvalid ;       // the GPU holds the current generation
   int gpuahead ;       // ... and the tree does not
   int gpuok() ;
   void gpustep() ;
   void gpusync() ;
   void gpudraw() ;
   void gpubricks(supertile *p, int lev, int xdel, int ydel, int odd, std::vector<qlgbrick> &v) ;
   void putbrick(const qlgbrick &k, int odd) ;
   void freetree(supertile *p, int lev) ;
   linkedmem *filllist(int size) ;
   linkedmem *grab(linkedmem *&list, int size, int which) ;
   brick *newbrick() ;
   tile *newtile() ;
   supertile *newsupertile(int lev) ;
   void uproot() ;
   template<int PAR> int doquad01(supertile *zis, supertile *edge,
                                  supertile *par, supertile *cor, int lev, int ex) ;
   template<int PAR> int doquad10(supertile *zis, supertile *edge,
                                  supertile *par, supertile *cor, int lev, int ex) ;
   template<int PAR> int p01(tile *p, tile *pr, tile *pd, tile *prd, int prex) ;
   template<int PAR> int p10(tile *plu, tile *pu, tile *pl, tile *p, int plex) ;
   void parinit() ;
   void parstop() ;
   int parok() ;
   int parmaxgens() ;
   int parpick() ;
   void partimed(int mode, double secs, double tiles) ;
   void parepoch(int e, int mode) ;
   void parbarrier(int id, long long b, int global) ;
   void parshadows(int e) ;
   supertile *shcopy(supertile *s, int lev, int x0, int y0, int band) ;
   std::vector<supertile *> frozennull ;   // frozen empty tile (0) and supertiles
   void shfree(supertile *s, int lev) ;
   void parregion(int k, int q, int odd) ;
   int isregion(supertile *s) ;
   int parmark(supertile *n, int lev) ;
   supertile *parcollect(supertile *const nb[9], int lev, unsigned long long key) ;
   void parlink(supertile *const nb[9], int lev) ;
   int parcombine(supertile *n, int lev, int q, int odd) ;
   void setruletable(int odd) ;
   qlpar *par ;
   int inpar ;
   int parlev ;         // level of the regions
   int nmodes, modethreads[QLMODES] ;   // threads of each mode (see parpick)
   int parlevm[QLMODES] ;   // best region level of each mode (see parpick)
   int parstamp ;       // marks supertiles of the current epoch (amark, reg)
   double tmode[QLMODES] ;    // recent time per tile computed of each mode (see parpick)
   double work ;        // tiles computed per generation in the last run (see parpick)
   double wlost[QLMODES] ;    // work when each mode last lost a comparison
   double tclock ;      // seconds in runs of generations
   double tnext[QLMODES] ;    // when (tclock) to try each mode again
   double tlast[QLMODES] ;    // when each mode was last tried
   // the same for the level below (0) and above (1) each mode's
   double lnext[QLMODES][2], lwork[QLMODES][2], llast[QLMODES][2] ;
   int runlev, lastlev, trylev ;   // level of this run, the last, the try
   int trylevel ;       // whether the try is of a level (else of a mode)
   int lastmode ;
   int trymode, tryleft ;   // the mode being tried, runs left
   double trywork, trysecs, trytiles, trysum, trysumtiles ;
   G_INT64 find_set_bits(supertile *p, int lev, int gm1) ;
   int isEmpty(supertile *p, int lev, int gm1) ;
   supertile *mdelete(supertile *p, int lev, int odd) ;
   void release(linkedmem *&list, int which, void *p) ;
   void parclean(int k, int odd) ;
   G_INT64 popcount() ;
   int uproot_needed() ;
   void dogen() ;
   void renderbm(int x, int y) ;
   void renderbm(int x, int y, int xsize, int ysize) ;
   void BlitCells(supertile *p, int xoff, int yoff, int wd, int ht, int lev) ;
   void ShrinkCells(supertile *p, int xoff, int yoff, int wd, int ht, int lev) ;
   int nextcell(int x, int y, supertile *n, int lev) ;
   void fill_ll(int d) ;
   int lowsub(vector<supertile*> &src, vector<supertile*> &dst, int lev) ;
   int highsub(vector<supertile*> &src, vector<supertile*> &dst, int lev) ;
   void allsub(vector<supertile*> &src, vector<supertile*> &dst, int lev) ;
   int gethbitsfromleaves(vector<supertile *> v) ;
   int getvbitsfromleaves(vector<supertile *> v) ;
   supertile *markglobalchange(supertile *, int, int &) ;
   void markglobalchange() ; // call if the rule changes
   /* data elements */
   int min, max, rootlev ;
   int minlow32 ;
   bigint bmin, bmax ;
   bigint population ;
   int popValid ;
   // the allocation state, which threads of parallel generations write
   // (under alloclock, see grab), on cache lines apart from the fields the
   // generation code reads
   friend struct qlallock ;
   alignas(64) std::atomic_flag alloclock ;
   linkedmem *tilelist, *supertilelist, *bricklist ;
   linkedmem *memused ;
   g_uintptr_t usedmemory ;
   alignas(64) brick *emptybrick ;
   tile *emptytile ;
   supertile *root, *nullroot, *nullroots[40] ;
   int cleandowncounter ;
   g_uintptr_t maxmemory ;
   // vector brick kernel: vrule[k] for qliferules.rule0/rule1 if vecok[k]
   qlvrule vrule[2] ;
   int vecok[2] ;
   qlkernel kern01, kern10 ;
   // when drawing, these are used
   liferender *renderer ;
   viewport *view ;
   int uviewh, uvieww, viewh, vieww, mag, pmag, kadd ;
   int oddgen ;
   int bmleft, bmtop, bmlev, shbmsize, logshbmsize ;
   int quickb, deltaforward ;
   int llbits, llsize ;
   char *llxb, *llyb ;
   liferules qliferules ;
} ;
#endif
