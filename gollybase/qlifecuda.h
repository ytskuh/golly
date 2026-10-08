// This file is part of Golly.
// See docs/License.html for the copyright notice.

/**
 *   Generations of QuickLife universes on a CUDA GPU (see qlifecuda.cu).
 *   Used by qlifealgo for universes made as "QuickLife CUDA".
 */
#ifndef QLIFECUDA_H
#define QLIFECUDA_H
#include <vector>
/*
 *   A brick of the QuickLife tree for one generation parity: the cells at
 *   tree coordinates (x, y) to (x+31, y+7), as 8 slices of 4x8 cells (see
 *   qlifealgo.h).  x is a multiple of 32 and y of 8.
 */
struct qlgbrick {
   int x, y ;
   unsigned int d[8] ;
} ;
struct qlgpu ;
// start the CUDA driver in the background (once per process)
void qlgpu_warmup() ;
// a GPU universe, or 0 if there is no usable CUDA device
qlgpu *qlgpu_new() ;
void qlgpu_delete(qlgpu *g) ;
// replace the cells with these bricks of a generation of parity odd
void qlgpu_load(qlgpu *g, const std::vector<qlgbrick> &b, int odd) ;
// the non-empty bricks of the current generation, of parity odd
void qlgpu_save(qlgpu *g, std::vector<qlgbrick> &b, int odd) ;
/*
 *   The cells of the current generation (of parity odd) as a bit image:
 *   element (px, py) of w x h is set if a cell at x = cx + px * 2^ls + i,
 *   y = cy + py * 2^ls + j (0 <= i, j < 2^ls) is alive, in Golly's cell
 *   coordinates (y downward).  bits gets h rows of (w + 31) / 32 words,
 *   most significant bit first.
 */
void qlgpu_draw(qlgpu *g, int odd, long long cx, long long cy, int ls, int w, int h,
                std::vector<unsigned int> &bits) ;
// the number of live cells of the current generation (of parity odd)
long long qlgpu_population(qlgpu *g, int odd) ;
// the bounding box of the live cells (Golly's coordinates); 0 if there are none
int qlgpu_edges(qlgpu *g, int odd, long long &left, long long &top,
                long long &right, long long &bottom) ;
/*
 *   Compute gens generations from the current one, of parity odd, with an
 *   outer-totalistic rule: with T the number of live cells in a 3x3 block
 *   (center included), a dead center is born if bit T of born[p] is set
 *   and a live center survives if bit T of stays[p] is set, where p is the
 *   parity of the generation the new one is computed from.  born[p] must
 *   not have bit 0 set.
 */
void qlgpu_run(qlgpu *g, int gens, int odd,
               const unsigned int born[2], const unsigned int stays[2]) ;
#endif
