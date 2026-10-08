# Golly benchmarks

Measures two things on the patterns in `Patterns/`:

| target | binary | what is timed |
| ------ | ------ | ------------- |
| kernel | `bgolly` | `step()` calls only (pattern loading is reported separately) |
| GUI | `golly --bench` | wall time of the real GUI generating loop: `OnGenTimer`, `StepPattern`, viewport and status bar painting |

Result files record the end generation, population and bounding box of every run, so `compare` also checks that a new implementation computes the same pattern as the baseline.

## Build

Benchmark an optimized build. `gui-wx/build.ninja` compiles without `-O`, so use `makefile-gtk` (`-O3`), with its own object directory:

```sh
cd gui-wx
make -f makefile-gtk -j$(nproc) OBJDIR=ObjRel ENABLE_SOUND= golly bgolly
```

For the algorithm QuickLife CUDA (an NVIDIA GPU and the CUDA toolkit in `/opt/cuda`; `CUDA_HOME=...` for another place), add `ENABLE_CUDA=1` and use another object directory, since the other objects change too:

```sh
make -f makefile-gtk -j$(nproc) OBJDIR=ObjCuda ENABLE_SOUND= ENABLE_CUDA=1 golly bgolly
```

This writes `golly` and `bgolly` to the repo root, which is where the driver looks by default (`--golly`, `--bgolly` to override). Running `ninja` afterwards overwrites them with unoptimized binaries; result files record the binary's SHA-256 and modification time so this can be spotted later.

## Run

The driver uses only the Python standard library; run it with uv from the repo root:

```sh
uv run --project bench bench/golly_bench.py kernel                # quick group, every algorithm, with bgolly
uv run --project bench bench/golly_bench.py kernel --cases-group full --algorithms HashLife
uv run --project bench bench/golly_bench.py kernel --algorithms "HashLife,HashLife Parallel" --threads 8   # compare
uv run --project bench bench/golly_bench.py gui                   # quick group in the GUI
uv run --project bench bench/golly_bench.py gui --algorithms "HashLife,HashLife Parallel" --threads 8
uv run --project bench bench/golly_bench.py compare KERNEL.json GUI.json   # GUI overhead per case
uv run --project bench bench/golly_bench.py sweep --jobs 8        # every pattern, 1000 gens, step 1
uv run --project bench bench/golly_bench.py compare OLD.json NEW.json
```

Choosing what runs (`kernel` and `gui`):

| option | meaning | default |
| ------ | ------- | ------- |
| `--algorithms A,B,…` | algorithms to benchmark, as named in Golly's Algorithm menu. Each runs the cases of its algorithm; `HashLife Parallel` and `QuickLife Parallel` run the `HashLife` and `QuickLife` cases, and `QuickLife CUDA` the `QuickLife` cases. `HashLife` and `HashLife Parallel` also run the `QuickLife` cases, as a check on patterns that don't suit HashLife; QuickLife never runs the `HashLife` cases | each selected case with its own algorithm only |
| `--cases-group G` | case group in `cases.toml`: `quick` or `full` | `quick` |
| `--cases a,b,c` | run these cases instead of a group | |

When a case runs with more than one algorithm, the driver ends with a comparison table: each algorithm's gens/s, its speedup (gens/s divided by the gens/s of the first listed algorithm that ran the case), and whether its end generation, population and bounding box match that first algorithm's (`other gen` when the runs ended at different generations, as fixed-time GUI runs can). It also prints the geometric mean speedup per algorithm.

Other options: `--repeat N`, `--tag NAME` (added to the result file name), `--out FILE`, `--threads N` (threads for the Parallel algorithms; `kernel` passes it to bgolly only for them, `gui` as `threads=N`). `kernel` also takes `--parcutoff L`, passed to bgolly (see below). They are recorded in the result's `settings`, together with `algorithms` and `cases_group`; each result entry's `algo` is the algorithm that ran.
`compare` matches entries by case and algorithm.
Results go to `bench/results/KIND-DATE[-TAG].json`.

Kernel runs are single-process and timing-sensitive: avoid running other heavy work (including a GUI benchmark) at the same time. `sweep --jobs N` with N > 1 is meant for correctness baselines; its timings are noisy.

## Run on another machine

`bench/remote.sh` copies the working tree (sources, `Patterns/`, `bench/`; no build outputs or results) to a remote machine with rsync, builds `bgolly` there the same way as above, runs the driver, and copies the remote `bench/results/` into `bench/results/HOSTNAME/`:

```sh
bench/remote.sh kernel --cases-group full --algorithms "HashLife,HashLife Parallel" --threads 32 --tag t32
bench/remote.sh --sh './bgolly --help'                         # any command, run from the remote copy
bench/remote.sh --host USER@HOST --dir PATH kernel             # defaults: yutsi@hyperion, ~/golly-sync/cuda-2
bench/remote.sh --make 'ENABLE_CUDA=1 OBJDIR=ObjCuda' kernel --algorithms "QuickLife CUDA"   # make variables for the build
```

The remote machine needs g++, make, rsync and uv. No code is edited there; every run starts by syncing the local tree.

## Cases

`cases.toml` lists the cases. Each case belongs to one algorithm (`algo`) and to one or more groups (`groups`):

| group | cases |
| ----- | ----- |
| quick | 16, every algorithm |
| full | quick + 21 heavier cases (37 in all) |

Patterns come from `Patterns/`, except `bench/patterns/line-N.rle`: a horizontal line of N live cells, one cell high (cases `hl-line-N`, N = 1024 … 65536).

Each case fixes the step (`base^expo` generations per step) and the number of steps (`steps`); a run stops at generation `steps × base^expo`. bgolly and the GUI therefore make the same `step()` calls and their times can be compared directly.

Step size of the HashLife and QuickLife cases:

| key | value |
| --- | ----- |
| steps | 100 for every case |
| step | `min(2^30, largest 2^n at which the original algorithm's mean step time over the 100 steps is ≤ 100 ms)` |
| original algorithm | the case's algorithm (`HashLife` or `QuickLife`) in upstream Golly at commit `dcbc0873`, the commit this branch's changes start from; 500 MB hash memory for HashLife, unlimited for QuickLife (the GUI defaults) |
| measured on | the machine where `calibrate` was last run (recorded in `cases.toml`) |

So the original code takes at most ~10 s per case run. `calibrate` finds the steps:

```sh
git archive dcbc0873 cmdline gollybase gui-wx | tar -x -C ORIG
make -C ORIG/gui-wx -f makefile-gtk -j$(nproc) OBJDIR=ObjRel ENABLE_SOUND= bgolly
uv run --project bench bench/golly_bench.py calibrate --bgolly ORIG/bgolly [--cases a,b,c]
```

It times 100 steps of 2^n generations with `bgolly -b -T 10` (`-b` prints a timestamp before every step; `-T 10` stops the run once 10 s have passed since the first step, so a run that finishes all 100 steps has a mean step time ≤ 100 ms). It starts at the case's current `expo` and moves up while the mean is within 100 ms, or down until it is, assuming the step time grows with n. It prints the mean step time of every n it tried and writes them to `bench/results/calibrate-DATE.json`; copy the chosen `expo` into `cases.toml`. Cases of other algorithms keep their hand-picked steps, and so do cases marked `fixed_step = true`: `ql-lfod-rule110` and `ql-nontotalistic-bubbles` use bounded grids, which bgolly only steps one generation at a time, so they run at step 1 for 3000 and 5000 steps.

## Kernel metrics (`kernel`, `sweep`)

bgolly is run as `bgolly -a ALGO -M MEM -i INCREMENT -m GEN -q -q --summary [--threads N] [--parcutoff L] PATTERN` from the repo root. `MEM` is the GUI's default for the algorithm (0 = unlimited for QuickLife and Larger than Life, 500 MB otherwise). `--summary` prints one line after the run.

| name | definition |
| ---- | ---------- |
| `run_s` | seconds from just after loading to the end of the last `step()` (one value per repeat) |
| `median_run_s` | median of `run_s` over the repeats |
| `load_s` | seconds to read the pattern file (median) |
| `steps` | number of `step()` calls |
| `steps_per_s` | `steps / median_run_s` |
| `gens_per_s` | `end_gen / median_run_s` |
| `end_gen`, `pop`, `bbox` | generation, population and bounding box (left, top, right, bottom) after the run |
| `consistent` | all repeats produced the same `end_gen`, `pop`, `bbox` |

Parallel HashLife and QuickLife (prototype; `gollybase/hlifealgo.cpp` section "Parallel stepping", `gollybase/qlifealgo.cpp` section "Parallel generations"):

| bgolly option | meaning | default |
| ------------- | ------- | ------- |
| `--threads N` | HashLife: 0 runs the serial code; N ≥ 1 runs the parallel code with N threads (N = 1 measures its overhead). QuickLife: N ≥ 2 runs generations on N worker threads when that is faster than one thread (it times serial generations, generations on the workers in its L3 group and on all workers, uses the fastest, and tries the others every 256 generations); 0 and 1 run the serial code | 0 |
| `--parcutoff L` | HashLife only: nodes below level L (2^L cells wide) are computed by the plain serial recursion inside one piece of work; at L and above, sub-results can be handed to other threads. Values 1–4 are treated as 5. 0 chooses L by timing steps: it starts at 6, and every 32 to 512 steps runs 17 steps that alternate L with L+1 and L−1, moving to a neighbour when its steps are at least 7% faster than the steps at L around them (3 of 4 faster); a run of a single step uses 6 | 0 |

On Linux both group the CPUs this process may use by shared L3 cache and restrict each worker thread to one group, filling the group of the calling thread first (HashLife also pins the calling thread, during a step). Results therefore depend on how many threads fit in one L3 group.

Other algorithms ignore these options.

The algorithms `HashLife Parallel` and `QuickLife Parallel` (`-a "HashLife Parallel"`, and in the GUI's Algorithm menu) are the same code with the thread count of their universes set separately: `--threads N` (N ≥ 1) in bgolly, Preferences > Control > "Threads for Parallel algorithms" (`par_threads` in the prefs file) in the GUI. Default: the number of CPUs (`std::thread::hardware_concurrency()`).

## GPU prototype (`bench/gpu/gpulife.cu`)

A standalone CUDA program, not part of Golly, to measure what a GPU gives on patterns that suit QuickLife. The algorithm QuickLife CUDA (`gollybase/qlifecuda.cu`, built with `ENABLE_CUDA=1`) is the version inside Golly. It reads an RLE file with a two-state B/S rule on an unbounded grid, runs GENS generations as one step, and prints a line in the format of `bgolly --summary`, so its `pop` and `bbox` can be compared with bgolly's:

```sh
nvcc -O3 -arch=sm_120 -o gpulife bench/gpu/gpulife.cu      # sm_120: RTX 50 series
./gpulife Patterns/Life/Breeders/p90-rake-factory.rle 20000
bench/remote.sh --sh '/opt/cuda/bin/nvcc -O3 -arch=sm_120 -o /tmp/gpulife bench/gpu/gpulife.cu && /tmp/gpulife PATTERN GENS'
```

The universe is a dense grid of bits (64 cells per word), grown when live cells come near an edge; blocks of 32 words by 8 rows are skipped when neither they nor their neighbors changed in the last generation. `run_s` covers the generations, the grid growth and the final population and bounding box.

## GUI metrics (`gui`)

The driver starts `golly --bench ...` with `HOME` pointing at a temporary directory that is cleared before every run, so each run uses Golly's default preferences and the user's own preferences are never read or written.

Benchmark mode (`gui-wx/wxbench.cpp`) resizes the window until the viewport has the requested size, loads the pattern, sets the algorithm and step, fits the pattern in view, waits for the first paint, then calls `StartGenerating()`. It stops after the first step where the generation reaches `gen` or `secs` have passed, writes JSON and quits.

How Golly generates (`gui-wx/wxcontrol.cpp`): a timer calls `OnGenTimer`, which does one step (`StepPattern`) and updates the display. The timer interval is `SIXTY_HERTZ` (16 ms on Linux), or the delay when the step exponent is < 0, so Golly does at most ~60 steps per second; the step size (base^exponent) sets the speed. When a step and its display update take longer than the interval, the next timer event is already due and the next step starts at once. While generating, the status bar shows the measured steps per second.

To measure computation rather than the 60 steps/s limit, choose a step size at which every step takes longer than 16 ms.

| name | definition |
| ---- | ---------- |
| `wall_s` | seconds from `StartGenerating()` to the end of the last step |
| `steps` | number of `StepPattern` calls |
| `steps_per_s` | `steps / wall_s` |
| `step_only_steps_per_s` | `steps / step_s`: the rate if nothing but stepping took time |
| `gens`, `gens_per_s` | `end_gen - start_gen`, `gens / wall_s` |
| `batches` | number of `OnGenTimer` calls (one step and display update each) |
| `frames` | number of viewport paints (`PatternView::OnPaint`) |
| `batch_s` | time in `OnGenTimer`, including `step_s`, excluding time inside the poller's `Yield` |
| `step_s` | time in the step code of `StepPattern` (`step()`, plus `CreateBorderCells`/`DeleteBorderCells` on bounded grids), excluding `Yield` |
| `draw_s` | time in `DrawView` (building the frame with OpenGL calls) |
| `swap_s` | time in `SwapBuffers` (includes waiting for the GPU and for vsync) |
| `status_s` | time painting the status bar (includes `getPopulation()`) |
| `yield_s` | time inside the poller's `Yield` calls (events handled during a step) |
| `other_s` | `wall_s - batch_s - draw_s - swap_s - status_s`: time in the event loop outside Golly's handlers |
| `step_ms` | mean / p50 / p95 / max of per-step time; `step_ms_all` has every value |
| `frame_ms` | mean / p50 / p95 / max of per-paint `draw + swap` time; `frame_ms_all` has every value |
| `par_threads` | threads for the Parallel algorithms |
| `view_wd`, `view_ht` | viewport size in pixels (window units × display scale factor) |
| `stop_reason` | `gen` (target reached), `secs` (time limit) or `not_started` |

The driver prints, per case: `wall_s`, `steps_per_s`, `step_only_steps_per_s`, the share of `wall_s` spent in step, paint (draw + swap + status) and other, and frames/s (`frames / wall_s`).

`step_s / wall_s` is the fraction of time the GUI spends computing; the rest is GUI overhead. `compare KERNEL.json GUI.json` gives the same thing against bgolly (speedup < 1 = GUI is slower than the kernel alone).

On GTK3 under Wayland, `other_s` includes GTK repainting the whole window in software (pixman) once per display update, which costs more for larger windows (measured on KDE Wayland: ~1.9 ms at 0.7 Mpx, ~4.7 ms at 2.8 Mpx, ~7.1 ms at 6.4 Mpx per update). Under X11 (`GDK_BACKEND=x11`) this cost is absent.

### `golly --bench` arguments

Can also be run by hand; relative paths are resolved from the current directory.

```
golly --bench pattern=FILE out=FILE [algo=NAME] [base=N] [expo=N]
      [gen=N] [secs=S] [size=WxH] [mag=fit|N] [autofit=0|1] [threads=N]
```

| key | meaning | default |
| --- | ------- | ------- |
| `pattern` | pattern file | required |
| `out` | JSON result file | required |
| `algo` | algorithm to load the pattern with | Golly's normal choice |
| `base`, `expo` | step size `base^expo` | algorithm's default base, expo 0 |
| `gen` | stop when the generation reaches N | none |
| `secs` | stop after S seconds | none (`gen` or `secs` is required) |
| `size` | viewport size in window units | current window |
| `mag` | `fit` to fit the pattern, or a magnification (0 = 1:1, 1 = 1:2, -1 = 2:1, …) | `fit` |
| `autofit` | Golly's auto fit option during the run | 0 |
| `threads` | threads for the Parallel algorithms (recorded as `par_threads`) | the `par_threads` preference: number of CPUs |

Run it with a throwaway `HOME` (as the driver does) if you don't want your own preferences to affect the run. Benchmark mode never saves preferences.
