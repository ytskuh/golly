"""Benchmark driver for Golly.

Subcommands:
  kernel   run cases with bgolly (pure computation, no GUI)
  gui      run cases with golly --bench (the real GUI generating loop)
  calibrate  choose the step size of the HashLife and QuickLife cases with the original bgolly
  sweep    run every pattern in Patterns/ for a fixed number of generations
  compare  compare two result files (speedup and correctness)

See README.md in this directory.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"

# Same order the GUI uses when it looks for an algorithm that can load a pattern.
ALGOS = ["QuickLife", "HashLife", "Generations", "Larger than Life", "JvN", "Super", "RuleLoader"]
# GUI's default max memory (MB) per algorithm; 0 means no limit.
ALGO_MAXMEM = {"QuickLife": 0, "Larger than Life": 0}
DEFAULT_MAXMEM = 500
# Algorithms whose cases an algorithm runs (default: its own); HashLife also runs the QuickLife cases.
RUNS_CASES_OF = {"HashLife": ("HashLife", "QuickLife")}
# calibrate: the step is the largest 2^n, n <= CALIBRATE_MAX_EXPO, at which the original algorithm's
# mean step time over the case's steps is at most CALIBRATE_STEP_S
CALIBRATE_STEP_S = 0.1
CALIBRATE_MAX_EXPO = 30
# File types bgolly can read.
PATTERN_SUFFIXES = (".rle", ".mc", ".mcl", ".lif", ".rle.gz", ".mc.gz")


# ---------------------------------------------------------------------------
# environment


def run_text(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def machine_info() -> dict:
    cpu = None
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "cpu_governor": governor.read_text().strip() if governor.exists() else None,
        "gpu": run_text(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "session": os.environ.get("XDG_SESSION_TYPE"),
        "desktop": os.environ.get("XDG_CURRENT_DESKTOP"),
    }


def build_info(binary: Path) -> dict:
    return {
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "binary_mtime": datetime.datetime.fromtimestamp(binary.stat().st_mtime).isoformat(timespec="seconds"),
        "git_commit": run_text(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "git_branch": run_text(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(run_text(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])),
    }


# ---------------------------------------------------------------------------
# cases


def family(algo: str) -> str:
    """The algorithm whose cases `algo` runs: "HashLife Parallel" runs the HashLife cases, "QuickLife CUDA" the QuickLife ones."""
    return algo.removesuffix(" Parallel").removesuffix(" CUDA")


def runs_case(algo: str, case: dict) -> bool:
    return case["algo"] in RUNS_CASES_OF.get(family(algo), (family(algo),))


def load_runs(args) -> tuple[list[tuple[dict, str]], list[str]]:
    """(case, algorithm) pairs to run, case by case, and the algorithm list.
    Without --algorithms each case runs with its own algorithm only."""
    cases = tomllib.loads(args.cases_file.read_text())["case"]
    if args.cases:
        byname = {c["name"]: c for c in cases}
        missing = [n for n in args.cases if n not in byname]
        if missing:
            sys.exit(f"unknown case(s): {', '.join(missing)}")
        cases = [byname[n] for n in args.cases]
    else:
        cases = [c for c in cases if args.cases_group in c["groups"]]
    if args.algorithms:
        algos = args.algorithms
        runs = [(c, a) for c in cases for a in algos if runs_case(a, c)]
    else:
        algos = list(dict.fromkeys(c["algo"] for c in cases))
        runs = [(c, c["algo"]) for c in cases]
    if not runs:
        sys.exit("no selected case belongs to these algorithms")
    return runs, algos


def increment(case: dict) -> int:
    return case["base"] ** case["expo"]


def gen_target(case: dict) -> int:
    return case["steps"] * increment(case)


def label(case: dict, algo: str) -> str:
    return f"{case['name']:<30} {algo:<18}"


def compare_algorithms(results: list[dict]) -> None:
    """For cases run with more than one algorithm, print each algorithm's gens/s against the first one's."""
    bycase = {}
    for e in results:
        bycase.setdefault(e["case"], []).append(e)
    groups = [es for es in bycase.values() if len(es) > 1]
    if not groups:
        return
    print("\nspeedup = gens/s / gens/s of the case's first algorithm; "
          "result = end gen, population, bounding box against the first algorithm's\n")
    print(f"{'case':<30} {'algorithm':<18} {'gens/s':>9} {'speedup':>8}  result")
    speedups = {}
    for es in groups:
        ref = es[0]
        for e in es:
            if e["status"] != "ok" or ref["status"] != "ok":
                print(f"{e['case']:<30} {e['algo']:<18} {e['status']}")
                continue
            sp = e["gens_per_s"] / ref["gens_per_s"]
            if e is ref:
                result = ""
            elif e["end_gen"] != ref["end_gen"]:
                result = "other gen"
            else:
                result = "same" if (e["pop"], e["bbox"]) == (ref["pop"], ref["bbox"]) else "DIFFERENT"
            if e is not ref:
                speedups.setdefault((ref["algo"], e["algo"]), []).append(sp)
            print(f"{e['case']:<30} {e['algo']:<18} {fmt_rate(e['gens_per_s'])} {sp:7.2f}x  {result}")
    print()
    for (refalgo, algo), sps in speedups.items():
        print(f"{algo} against {refalgo}: geometric mean speedup {statistics.geometric_mean(sps):.2f}x "
              f"over {len(sps)} cases")


# ---------------------------------------------------------------------------
# bgolly


def bgolly_run(bgolly: Path, pattern: str, algo: str, inc: int, gen: int, timeout: float,
               extra: list[str] = []) -> dict:
    """Run bgolly once and return the parsed --summary line (or an error)."""
    cmd = [str(bgolly), "-a", algo, "-M", str(ALGO_MAXMEM.get(algo, DEFAULT_MAXMEM)),
           "-i", str(inc), "-m", str(gen), "-q", "-q", "--summary", *extra, pattern]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "process_s": timeout}
    process_s = time.perf_counter() - start
    for line in proc.stdout.splitlines():
        if line.startswith("summary "):
            fields = dict(kv.split("=", 1) for kv in line.split()[1:])
            bbox = None if fields["bbox"] == "empty" else fields["bbox"].split(",")
            return {
                "status": "ok",
                "load_s": float(fields["load_s"]),
                "run_s": float(fields["run_s"]),
                "steps": int(fields["steps"]),
                "process_s": process_s,
                "end_gen": fields["gen"],
                "pop": fields["pop"],
                "bbox": bbox,
            }
    lines = (proc.stdout + proc.stderr).strip().splitlines()
    return {"status": "error", "error": " | ".join(lines[-3:])}


def kernel_extra(algo: str, args) -> list[str]:
    extra = []
    # bgolly --threads also makes plain HashLife and QuickLife parallel, so pass it only to the Parallel algorithms
    if args.threads is not None and algo.endswith(" Parallel"):
        extra += ["--threads", str(args.threads)]
    if args.parcutoff is not None:
        extra += ["--parcutoff", str(args.parcutoff)]
    return extra


def cmd_kernel(args) -> dict:
    todo, algos = load_runs(args)
    results = []
    for case, algo in todo:
        inc = increment(case)
        runs = []
        for _ in range(args.repeat):
            r = bgolly_run(args.bgolly, case["pattern"], algo, inc, gen_target(case), args.timeout,
                           kernel_extra(algo, args))
            runs.append(r)
            if r["status"] != "ok":
                break
        entry = {
            "case": case["name"],
            "pattern": case["pattern"],
            "algo": algo,
            "base": case["base"],
            "expo": case["expo"],
            "increment": str(inc),
            "gen_target": gen_target(case),
        }
        if all(r["status"] == "ok" for r in runs):
            run_s = [r["run_s"] for r in runs]
            med = statistics.median(run_s)
            last = runs[-1]
            gens = int(last["end_gen"])
            entry.update({
                "status": "ok",
                "run_s": run_s,
                "median_run_s": med,
                "load_s": statistics.median(r["load_s"] for r in runs),
                "steps": last["steps"],
                "steps_per_s": last["steps"] / med if med > 0 else None,
                "gens_per_s": gens / med if med > 0 else None,
                "end_gen": last["end_gen"],
                "pop": last["pop"],
                "bbox": last["bbox"],
                "consistent": all((r["end_gen"], r["pop"], r["bbox"]) == (last["end_gen"], last["pop"], last["bbox"])
                                  for r in runs),
            })
            print(f"{label(case, algo)} {med:>9.4f} s  {fmt_rate(entry['steps_per_s'])} steps/s  "
                  f"{fmt_rate(entry['gens_per_s'])} gens/s   pop {last['pop']}")
        else:
            bad = runs[-1]
            entry.update({"status": bad["status"], "error": bad.get("error")})
            print(f"{label(case, algo)} {bad['status']}: {bad.get('error', '')}")
        results.append(entry)
    compare_algorithms(results)
    return {
        "kind": "kernel",
        "settings": {"repeat": args.repeat, "algorithms": algos, "cases_group": args.cases_group,
                     "cases": args.cases, "cases_file": str(args.cases_file),
                     "threads": args.threads, "parcutoff": args.parcutoff},
        "build": build_info(args.bgolly),
        "results": results,
    }


# ---------------------------------------------------------------------------
# GUI


def gui_run(golly: Path, home: Path, case: dict, algo: str, args) -> dict:
    # start every run from Golly's default preferences
    shutil.rmtree(home / ".golly", ignore_errors=True)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out = Path(tmp.name)
    secs = case.get("gui_secs", args.secs)
    cmd = [str(golly), "--bench",
           f"pattern={case['pattern']}", f"algo={algo}",
           f"base={case['base']}", f"expo={case['expo']}",
           f"gen={gen_target(case)}", f"secs={secs}",
           f"size={args.size}", f"mag={case.get('mag', 'fit')}",
           f"autofit={1 if case.get('autofit') else 0}", f"out={out}"]
    if args.threads is not None:
        cmd.append(f"threads={args.threads}")
    env = dict(os.environ, HOME=str(home))
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=secs + 120)
    except subprocess.TimeoutExpired:
        out.unlink(missing_ok=True)
        return {"status": "timeout"}
    try:
        data = json.loads(out.read_text())
    except (OSError, ValueError):
        lines = (proc.stdout + proc.stderr).strip().splitlines()
        return {"status": "error", "error": f"exit {proc.returncode}: " + " | ".join(lines[-3:])}
    finally:
        out.unlink(missing_ok=True)
    data["status"] = "ok"
    return data


def cmd_gui(args) -> dict:
    todo, algos = load_runs(args)
    results = []
    with tempfile.TemporaryDirectory(prefix="golly-bench-home-") as tmphome:
        home = Path(tmphome)
        for case, algo in todo:
            runs = [gui_run(args.golly, home, case, algo, args) for _ in range(args.repeat)]
            entry = {
                "case": case["name"],
                "pattern": case["pattern"],
                "algo": algo,
                "base": case["base"],
                "expo": case["expo"],
                "increment": str(increment(case)),
                "gen_target": gen_target(case),
            }
            if all(r["status"] == "ok" for r in runs):
                # report the run with the median wall time
                runs.sort(key=lambda r: r["wall_s"])
                med = runs[len(runs) // 2]
                entry.update({"status": "ok", "wall_s_all": [r["wall_s"] for r in runs]})
                entry.update({k: v for k, v in med.items() if k not in ("pattern", "algo", "status")})
                fps = med["frames"] / med["wall_s"] if med["wall_s"] > 0 else 0.0
                print(f"{label(case, algo)} {med['wall_s']:>8.3f} s  {fmt_rate(med['steps_per_s'])} steps/s  "
                      f"step-only {fmt_rate(med['step_only_steps_per_s'])}  "
                      f"step {share(med, 'step_s')} paint {share(med, 'draw_s', 'swap_s', 'status_s')} "
                      f"other {share(med, 'other_s')}  {fps:5.1f} fps  [{med['stop_reason']}]")
            else:
                bad = next(r for r in runs if r["status"] != "ok")
                entry.update({"status": bad["status"], "error": bad.get("error")})
                print(f"{label(case, algo)} {bad['status']}: {bad.get('error', '')}")
            results.append(entry)
    compare_algorithms(results)
    return {
        "kind": "gui",
        "settings": {"repeat": args.repeat, "algorithms": algos, "cases_group": args.cases_group,
                     "cases": args.cases, "cases_file": str(args.cases_file),
                     "size": args.size, "secs": args.secs, "threads": args.threads},
        "build": build_info(args.golly),
        "results": results,
    }


def share(r: dict, *keys: str) -> str:
    return f"{100 * sum(r[k] for k in keys) / r['wall_s']:4.0f}%" if r["wall_s"] > 0 else "   -"


# ---------------------------------------------------------------------------
# calibrate


def mean_step_s(bgolly: Path, case: dict, expo: int) -> float | None:
    """Mean step time of the case's steps at step 2^expo, or None if it is over CALIBRATE_STEP_S.

    Uses only options of the original bgolly: -b prints a timestamp before every step and after the
    last one (seconds from the first), and -T stops the run once the time limit has passed."""
    inc, steps, algo = 2 ** expo, case["steps"], case["algo"]
    limit = steps * CALIBRATE_STEP_S
    cmd = [str(bgolly), "-a", algo, "-M", str(ALGO_MAXMEM.get(algo, DEFAULT_MAXMEM)),
           "-i", str(inc), "-m", str(steps * inc), "-q", "-b", "-T", str(math.ceil(limit)), case["pattern"]]
    try:
        # -T is only checked between steps; the timeout stops a single step that takes far too long
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=limit + 60)
    except subprocess.TimeoutExpired:
        return None
    # each line is "SECONDS GEN"; the first one is before the first step
    stamps = [line.split() for line in proc.stdout.splitlines()]
    stamps = [(float(t[0]), int(t[1].replace(",", ""))) for t in stamps
              if len(t) == 2 and t[1].replace(",", "").isdigit()]
    # bgolly uses step 1 on a bounded grid; then there are more lines and the step isn't 2^expo
    if len(stamps) != steps + 1:
        if len(stamps) > steps + 1:
            sys.exit(f"{case['name']}: bgolly did not use step 2^{expo} (bounded grid?)")
        return None
    return stamps[-1][0] / steps


def cmd_calibrate(args) -> dict:
    cases = tomllib.loads(args.cases_file.read_text())["case"]
    if args.cases:
        cases = [c for c in cases if c["name"] in args.cases]
    cases = [c for c in cases if c["algo"] in ("HashLife", "QuickLife") and not c.get("fixed_step")]
    print(f"step = largest 2^n (n <= {CALIBRATE_MAX_EXPO}) with mean step time <= {CALIBRATE_STEP_S * 1000:g} ms "
          f"over the case's steps, run with {args.bgolly}\n")
    results = []
    for case in cases:
        # start from the case's current step and move up or down one exponent at a time
        expo = min(round(math.log2(increment(case))), CALIBRATE_MAX_EXPO)
        probes = {expo: mean_step_s(args.bgolly, case, expo)}
        if probes[expo] is not None:
            while expo < CALIBRATE_MAX_EXPO:
                probes[expo + 1] = mean_step_s(args.bgolly, case, expo + 1)
                if probes[expo + 1] is None:
                    break
                expo += 1
        else:
            while expo > 0 and probes[expo] is None:
                expo -= 1
                probes[expo] = mean_step_s(args.bgolly, case, expo)
        ms = {str(n): None if t is None else round(t * 1000, 3) for n, t in sorted(probes.items())}
        results.append({"case": case["name"], "algo": case["algo"], "steps": case["steps"],
                        "expo": expo, "mean_step_ms": ms})
        print(f"{case['name']:<30} {case['algo']:<10} 2^{expo:<3} mean step ms by n: "
              + "  ".join(f"{n}:{'over' if t is None else t}" for n, t in ms.items()))
    return {
        "kind": "calibrate",
        "settings": {"step_s": CALIBRATE_STEP_S, "max_expo": CALIBRATE_MAX_EXPO,
                     "cases_file": str(args.cases_file), "cases": args.cases},
        "build": build_info(args.bgolly),
        "results": results,
    }


# ---------------------------------------------------------------------------
# sweep


def pattern_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name.lower().endswith(PATTERN_SUFFIXES))


def sweep_one(bgolly: Path, path: Path, gen: int, timeout: float) -> dict:
    rel = str(path.relative_to(REPO))
    entry = {"case": rel, "pattern": rel, "base": 1, "expo": 0, "increment": "1", "gen_target": gen}
    # find the first algorithm that can load the pattern (as the GUI does)
    for algo in ALGOS:
        r = bgolly_run(bgolly, rel, algo, 1, 0, timeout)
        if r["status"] == "ok":
            break
    else:
        entry.update({"status": "unsupported", "error": r.get("error")})
        return entry
    entry["algo"] = algo
    r = bgolly_run(bgolly, rel, algo, 1, gen, timeout)
    if r["status"] != "ok":
        entry.update({"status": r["status"], "error": r.get("error")})
        return entry
    entry.update({
        "status": "ok",
        "median_run_s": r["run_s"],
        "load_s": r["load_s"],
        "gens_per_s": int(r["end_gen"]) / r["run_s"] if r["run_s"] > 0 else None,
        "end_gen": r["end_gen"],
        "pop": r["pop"],
        "bbox": r["bbox"],
    })
    return entry


def cmd_sweep(args) -> dict:
    files = pattern_files(REPO / "Patterns")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda p: sweep_one(args.bgolly, p, args.gen, args.timeout), files))
    for e in results:
        if e["status"] == "ok":
            print(f"{e['median_run_s']:9.4f} s  {e['algo']:<16} pop {e['pop']:>12}  {e['case']}")
        else:
            print(f"{e['status']:>11}  {'':<16} {'':>16}  {e['case']}  {e.get('error') or ''}")
    counts = {}
    for e in results:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    print("status counts:", counts)
    return {
        "kind": "sweep",
        "settings": {"gen": args.gen, "timeout": args.timeout, "jobs": args.jobs},
        "build": build_info(args.bgolly),
        "results": results,
    }


# ---------------------------------------------------------------------------
# compare


def fmt_rate(x: float | None) -> str:
    if x is None:
        return f"{'-':>9}"
    return f"{x:9.3g}"


def cmd_compare(args) -> None:
    base = json.loads(Path(args.base).read_text())
    new = json.loads(Path(args.new).read_text())
    if "sweep" in (base["kind"], new["kind"]) and base["kind"] != new["kind"]:
        sys.exit(f"cannot compare {base['kind']} results with {new['kind']} results")
    # older GUI result files kept the algorithm that ran in run_algo
    key = lambda r: (r["case"], r.get("run_algo", r.get("algo")))
    newby = {key(r): r for r in new["results"]}
    basekey = "wall_s" if base["kind"] == "gui" else "median_run_s"
    newkey = "wall_s" if new["kind"] == "gui" else "median_run_s"
    print(f"base: {args.base}  ({base['kind']}, {base['build'].get('git_commit', '')[:10]})")
    print(f"new:  {args.new}  ({new['kind']}, {new['build'].get('git_commit', '')[:10]})")
    print(f"time = {basekey} (base), {newkey} (new); speedup = base time / new time; "
          f"result = end gen, population, bounding box\n")
    print(f"{'case':<30} {'algorithm':<18} {'base s':>10} {'new s':>10} {'speedup':>8}  result")
    speedups, mismatches = [], 0
    for b in base["results"]:
        n = newby.get(key(b))
        if n is None:
            continue
        name = f"{b['case'][:30]:<30} {str(key(b)[1]):<18}"
        if b["status"] != "ok" or n["status"] != "ok":
            print(f"{name} {b['status']:>10} {n['status']:>10}")
            continue
        if b["gen_target"] != n["gen_target"]:
            print(f"{name} different gen targets ({b['gen_target']}, {n['gen_target']})")
            continue
        same = (b["end_gen"], b["pop"], b["bbox"]) == (n["end_gen"], n["pop"], n["bbox"])
        mismatches += not same
        tb, tn = b[basekey], n[newkey]
        sp = tb / tn if tn > 0 else float("inf")
        if tb > 0 and tn > 0:
            speedups.append(sp)
        print(f"{name} {tb:10.4f} {tn:10.4f} {sp:7.2f}x  {'same' if same else 'DIFFERENT'}")
    if speedups:
        print(f"\ngeometric mean speedup over {len(speedups)} cases: {statistics.geometric_mean(speedups):.2f}x")
    print(f"result mismatches: {mismatches}")


# ---------------------------------------------------------------------------


def write_result(data: dict, args) -> None:
    data["machine"] = machine_info()
    data["date"] = datetime.datetime.now().isoformat(timespec="seconds")
    out = args.out
    if out is None:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = f"-{args.tag}" if args.tag else ""
        out = RESULTS_DIR / f"{data['kind']}-{stamp}{tag}.json"
    Path(out).write_text(json.dumps(data, indent=1) + "\n")
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, binary: str):
        p.add_argument(f"--{binary}", type=Path, default=REPO / binary, help=f"path to {binary} (default: repo root)")
        p.add_argument("--out", type=Path, help="result file (default: bench/results/KIND-DATE[-TAG].json)")
        p.add_argument("--tag", help="label appended to the default result file name")

    def case_args(p):
        p.add_argument("--algorithms", type=lambda s: s.split(","),
                       help="comma-separated algorithms; each runs the cases of its algorithm "
                            "(\"HashLife Parallel\" runs the HashLife cases, \"QuickLife CUDA\" the QuickLife ones); a case run with several "
                            "algorithms is compared across them (default: the algorithm of each case)")
        p.add_argument("--cases-group", default="quick", help="group in cases.toml: quick or full (default: quick)")
        p.add_argument("--cases", type=lambda s: s.split(","),
                       help="comma-separated case names (instead of --cases-group)")
        p.add_argument("--cases-file", type=Path, default=BENCH_DIR / "cases.toml")

    p = sub.add_parser("kernel", help="run cases with bgolly")
    common(p, "bgolly")
    case_args(p)
    p.add_argument("--repeat", type=int, default=3, help="runs per case; the median is reported (default: 3)")
    p.add_argument("--timeout", type=float, default=600, help="seconds per run (default: 600)")
    p.add_argument("--threads", type=int,
                   help="threads for the Parallel algorithms (bgolly --threads; default: all CPUs)")
    p.add_argument("--parcutoff", type=int,
                   help="bgolly --parcutoff: HashLife parallel task cutoff level (default: bgolly's, 0 = chosen by timing)")

    p = sub.add_parser("gui", help="run cases with golly --bench")
    common(p, "golly")
    case_args(p)
    p.add_argument("--repeat", type=int, default=1, help="runs per case; the median is reported (default: 1)")
    p.add_argument("--size", default="1000x700", help="viewport size in window units (default: 1000x700)")
    p.add_argument("--secs", type=float, default=120, help="time limit per run unless the case sets gui_secs")
    p.add_argument("--threads", type=int,
                   help="threads for the Parallel algorithms (golly --bench threads=N; default: all CPUs)")

    p = sub.add_parser("calibrate", help="choose the step of the HashLife and QuickLife cases")
    p.add_argument("--bgolly", type=Path, required=True, help="bgolly built from the original code")
    p.add_argument("--out", type=Path, help="result file (default: bench/results/calibrate-DATE[-TAG].json)")
    p.add_argument("--tag", help="label appended to the default result file name")
    p.add_argument("--cases", type=lambda s: s.split(","), help="comma-separated case names (default: all)")
    p.add_argument("--cases-file", type=Path, default=BENCH_DIR / "cases.toml")

    p = sub.add_parser("sweep", help="run every pattern in Patterns/ with bgolly")
    common(p, "bgolly")
    p.add_argument("--gen", type=int, default=1000, help="generations to run, step 1 (default: 1000)")
    p.add_argument("--timeout", type=float, default=60, help="seconds per pattern (default: 60)")
    p.add_argument("--jobs", type=int, default=1, help="parallel bgolly processes (default: 1)")

    p = sub.add_parser("compare", help="compare two result files of the same kind")
    p.add_argument("base")
    p.add_argument("new")

    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # show progress when piped
    if args.command == "compare":
        cmd_compare(args)
        return
    binary = args.bgolly if args.command in ("kernel", "calibrate", "sweep") else args.golly
    if not binary.exists():
        sys.exit(f"{binary} not found; build it first (see bench/README.md)")
    data = {"kernel": cmd_kernel, "gui": cmd_gui, "calibrate": cmd_calibrate, "sweep": cmd_sweep}[args.command](args)
    write_result(data, args)


if __name__ == "__main__":
    main()
