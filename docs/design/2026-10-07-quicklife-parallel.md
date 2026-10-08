---
status: active
updated: 2026-10-07
composition_mode: llm_generated
---

# QuickLife: vector tile kernel and epoch-based parallel generations

Scope: `gollybase/qlifealgo.cpp`, `gollybase/qlifealgo.h` — the algorithms "QuickLife" (serial) and "QuickLife Parallel" (`nthreads > 1`). Cell results must stay identical to upstream QuickLife (commit dcbc0873) for every rule and pattern.

## Background

QuickLife stores the universe as a tree:

| name | size (cells) | content |
|---|---|---|
| slice | 4 wide x 8 tall | one 32-bit word; bit 31 is the upper-left cell, then across, then down |
| brick | 32 x 8 | 8 slices for even generations (`d[0..7]`), 8 for odd (`d[8..15]`) |
| tile (level 0) | 32 x 32 | 4 bricks, change bits `c[0..5]`, flags |
| supertile level L | 8 children of level L-1 | odd L stacks children along x, even L along y; flags with change bits |

- Stagger: the odd copy is stored shifted by (1,1) cells, so phase 0->1 (`p01`, `doquad01`) reads the right, lower and lower-right neighbors, and phase 1->0 (`p10`, `doquad10`) the left, upper and upper-left ones.
- Change bits compare a result with the result two generations earlier (same copy). A subtree is recomputed only when it or a neighbor changed in the previous phase.
- Change propagates at most one cell per generation (range-1 rules).
- A generation is little work: ql-cambrian-explosion takes about 136 us per generation serially on a 5 GHz core (25,600 generations), ql-52513m about 5 us (819,200 generations). Every cost paid per generation and per thread (a barrier, a cache line fetched from another core, an idle thread) is of the same order as the work.

## 1. Vector brick kernel

For rules whose 4x4 lookup table equals an outer-totalistic Moore rule (birth set B, survival set S, including the two tables used for B0 rules), `p01/p10` compute all 8 slices of a brick at once with bit-sliced arithmetic on 8 x 32-bit lanes (one 256-bit vector), when at least `QLVECMIN` slices of the brick need recomputing:

1. Horizontal 3-cell sums for every row (sum and carry bits), from the brick and its neighbor slices.
2. The same for the brick below (phase 0->1) or above (phase 1->0).
3. Vertical sums with 4- and 8-bit shifts within each lane: a 4-bit count `T` of each 3x3 block and its center `C`.
4. New cell = `C ? S contains T-1 : B contains T`; Life (B3/S23) has a fixed expression.
5. Store the new values of the slices to recompute, and compute the change bits that the scalar per-slice loop would produce, for all slices at once (`qlchg01`, `qlchg10`):
   - `d[j]`: change of slice `j` (old XOR new, OR `deltaforward`; 0 for slices not recomputed).
   - Phase 0->1: `m(j) = d[j+1] | (d[j] & 0x33333333)` for `j = -1..7`; bit `j+1` of "any" is `m(j) != 0`, of "low" is `(m(j) & 0xff) != 0`.
   - Phase 1->0: `m(j) = d[j-1] | (d[j] & 0xcccccccc)` for `j = 0..8`; bit `8-j` of "any" is `m(j) != 0`, of "high" is `(m(j) >> 24) != 0`.
   - These give the tile's change bits `c[]` exactly as the loop does, since the loop's running state only carries the previous slice's change.

Bricks with fewer than `QLVECMIN` (2) slices to recompute, and rules that are not outer-totalistic, use the lookup table and the scalar loop. `setrule` checks all 65,536 table entries against the formula; any mismatch keeps the lookup table. The kernel uses GCC/Clang vector extensions, compiled for the x86-64 baseline, AVX2 and AVX-512VL and chosen at run time; other compilers keep the lookup table.

## 2. Epoch-based parallel generations

`step()` runs runs of up to 63 generations ("epochs", `parepoch`), each either serially or with the first `n`, `n/2`, `n/4` or `n/8` workers (those of at least 2 threads; `n` = `nthreads`). These are the modes; workers fill L3 groups in order, so a mode uses as few groups as it can. The mode and the region level are chosen by measured time (see "Choosing the mode and the level").

### Regions

- A region is a supertile at level `parlev`; a region's position is its key, the child numbers from the root down (3 bits per level).
- Each epoch the calling thread marks the subtrees whose flags show change (`parmark`), lists as regions the marked supertiles at `parlev` and their eight neighbors (the ring), creating empty supertiles where needed (`parcollect`), and links each region to the supertiles `doquad` reads (`parlink`). Change moves at most one cell per generation and an epoch is shorter than a region's side less 8 cells, so nothing outside the regions changes during the epoch.
- `parlev` is at least 2 (regions of 256 x 256 cells or larger): level 1 regions (256 x 32) limit epochs to 24 generations, which costs more than it gains. Within that, the level is chosen per mode by time; it moves down at once when there are fewer active regions than workers.

### Work split

- Regions are split into one contiguous key range per worker (compact parts of the universe), by cost: each region's measured time per generation, smoothed over epochs and kept in its supertile.
- The range bounds are keys kept from epoch to epoch (one set per mode), so regions stay with their worker and their cells stay in its caches; they are recomputed only when a range's cost exceeds 1.2 times the mean.
- A worker takes regions from the front of its own range, 4 at a time, then single regions from the back of the ranges of the workers in its L3 group (never across groups). Taking from the back means the regions computed by another worker are the same from one generation to the next while the imbalance persists, so few regions change cache from generation to generation.

### Shadows between L3 groups

With workers in more than one L3 group, a group never reads another group's regions during an epoch:

- At the start of the epoch each group copies, of every region of another group next to one of its own, the cells within `bandwidth` = `e + 8` cells (rounded up to whole tiles) of its own regions (a shadow), plus a ring of one tile around that band.
- The group computes the band along with its own regions. The ring is frozen: copied with true cells, never computed (`doquad` skips subtiles marked frozen), and without change bits, so it neither changes nor makes its neighbors recompute; beyond the ring the shadow is frozen and empty.
- A shadow's cells near its edge go wrong because the shadow does not see what lies beyond, but that error moves at most one cell per generation and does not reach the group's own regions within the epoch.
- So groups exchange cells once per epoch (the copy) instead of every generation.

### One epoch

1. Copy the shadows (each group in parallel), global barrier.
2. Each generation: every group's workers run `doquad01/10<1>` on the regions and shadows that changed in the previous generation or whose neighbors it reads did, then wait at a barrier of their group. The rule table is per thread.
3. If the tree is due for cleaning (every 63 generations), each worker runs `mdelete` below its regions; the shadows are freed; global barrier.
4. The calling thread sets the flags of the tree above the regions as `doquad` would have after the last two generations (`parcombine`), and frees empty regions.

### Concurrency rules

- Within a generation a region writes only its own new-generation copy, change bits and new children, and reads its neighbors' old-generation copy. The parallel tile code reads both positions of a neighbor's change bits (processed or not yet processed in this generation); that only adds recomputation.
- Memory is freed only between generations of a region's owner group (cleaning pass) or between epochs; freed structures go to the freeing thread's own free list.
- Tiles and supertiles take whole cache lines (64 and 128 bytes), and the allocator's shared state has cache lines of its own, so threads working on different regions never write the same line.

### Choosing the mode and the level

Definitions:

- run: one call of `parepoch` (an epoch) or of up to 63 serial generations, timed as a whole.
- work: tiles computed per generation (`p01`/`p10` calls, counted per thread), which does not depend on the mode.
- time of a mode, `tmode[m]`: seconds per tile computed, averaged over about the last `QLTAU` = 4 ms of its runs (each run weighs `min(1, run seconds / QLTAU)`). Per tile, so times measured on a smaller or larger pattern compare.

Rules:

- Use the mode with the least time, at its level `parlevm[m]`.
- Now and then try one other mode, or a level next to the best mode's level (one at a time; among those due, the one tried longest ago). A mode with fewer threads than the best is tried at a level at least as high as the best mode's, a mode with more threads at a level at most as high.
- A try is 4 runs; only the last 2 are timed, and more runs follow until the timed ones take `QLTAU`. The first run after any change of mode or level is not timed, since it moves regions between threads.
- A loser keeps the work at the comparison and is tried again:
  - when the work has grown 1.3 times (a mode with more threads, a higher level) or shrunk 1.3 times (fewer threads, a lower level), since more threads and larger regions gain as the work grows;
  - otherwise after `QLTRYGAP (nmodes + 1)` times the try's expected cost (its extra time per tile over the tiles of a try, plus twice the measured cost of switching), with `QLTRYGAP` = 20: all tries together then cost about 5% of the time. This catches changes the work does not show, such as a sparse pattern spreading out.

Bounded grids, steps of fewer than 4 generations, generations with a global rule change, and universes more than 19 levels above the regions run serially.

### Thread placement

On Linux, workers fill the calling thread's L3 group first and are pinned to their group. Waiting threads pause, then yield after 1024 spins (there may be more threads than CPUs), and sleep between steps.

## 3. Measured limits and directions not adopted

Measured on an AMD EPYC 9575F (8 L3 groups of 8 cores, SMT off) with ql-cambrian-explosion, 25,600 generations in steps of 256, all threads forced; serial time 136 us per generation. Per worker per generation:

| threads | own regions | shadows | waiting in group | waiting for other groups | main thread between epochs |
|---|---|---|---|---|---|
| 32 | 8.0 us | 1.6 us | 1.9 us | 1.9 us | 2.1 us |
| 64 | 5.9 us | 1.9 us | 2.4 us | 3.4 us | 2.7 us |

- Waiting at the per-generation barrier: releasing a barrier takes under 1 us. The rest comes from the largest region being a large part of a worker's share of a generation, which is why the level is chosen by time and smaller regions are allowed.
- Own regions take 1.1 (8 threads), 1.5 (16), 1.9 (32) and 2.8 (64) times the serial time in total. Measured parts:
  - clock: 5.0 GHz serial, 4.8 GHz up to 32 threads, 3.7 GHz at 64;
  - cache lines at the borders between workers' areas, written by one core and read by another every generation; the share grows as each worker's area shrinks;
  - regions computed by different workers in successive generations (about 20% of own-region time before stealing from the back).
- Cross-group reads during an epoch are about 600 cache lines per generation at 32 threads, mostly at the copying of shadows.
- A worker range's cost varies by more than 1.2 times between epochs, so the ranges are recomputed in most epochs (425 of 491 at 32 threads); since they are split by cumulative cost in key order, the bounds move by a few regions each time.
- The main thread's work between epochs (marking, collecting and linking regions, shadows, combining flags) is spread over several functions and grows with the number of regions.

Directions measured and not adopted:

| direction | result |
|---|---|
| subtree tasks for all workers of a group | helped at 8 threads, slower at 32 and 64 (task bookkeeping) |
| per-region generation counters instead of the barrier | helped at 8 threads, slower at 32 and 64 (scanning for ready regions) |
| a fixed count of active regions per worker for the level | sparse patterns want large regions, dense ones small; no count suits both |
| a level rule from the regions' costs | pushed sparse patterns to small regions, whose per-region overhead dominates |
| separate tolerances for group and worker balance | no change in time |
| free lists per L3 group | no change in cross-group or DRAM fills |
| bricks and tiles padded to 128 bytes | 20% fewer cross-group fills, no change in time, serial 5% slower |

## Verification

- Results: `bench/golly_bench.py kernel --cases-group full` compares end generation, population and bounding box with the original build.
- Vector change bits: a test build that runs the kernel and the scalar loop on every brick (vector path for bricks with 1 or more slices to recompute) and compares the new slices and change bits; Life and outer-totalistic rules including a B0 rule.
- Machines without several L3 groups exercise shadows only through a test build that splits the CPUs into artificial groups.
