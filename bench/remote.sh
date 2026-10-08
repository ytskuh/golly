#!/usr/bin/env bash
# Run the kernel benchmark on another machine with the local working tree.
#
#   bench/remote.sh [--host USER@HOST] [--dir DIR] [--make 'VAR=VALUE...'] [ARGS...]
#   bench/remote.sh [--host USER@HOST] [--dir DIR] [--make 'VAR=VALUE...'] --sh 'COMMAND'
#
# 1. rsync the working tree (sources, Patterns, bench; no build outputs, no
#    results) to DIR on HOST,
# 2. build bgolly there with makefile-gtk (-O3, objects in ObjRel; --make adds
#    make variables, e.g. --make 'ENABLE_CUDA=1 OBJDIR=ObjCuda' for QuickLife CUDA),
# 3. run `uv run --project bench bench/golly_bench.py ARGS...` from DIR
#    (or, with --sh, run COMMAND from DIR); no ARGS only builds,
# 4. copy the remote bench/results/ into local bench/results/HOSTNAME/.
#
# Defaults: --host yutsi@hyperion, --dir golly-sync/cuda-2 (relative to the
# remote home directory).
set -euo pipefail

host=yutsi@hyperion
dir=golly-sync/cuda-2
shcmd=
makeargs=
while [[ $# -gt 0 ]]; do
   case "$1" in
      --host) host=$2; shift 2 ;;
      --dir) dir=$2; shift 2 ;;
      --sh) shcmd=$2; shift 2 ;;
      --make) makeargs=$2; shift 2 ;;
      --) shift; break ;;
      *) break ;;
   esac
done

repo=$(cd "$(dirname "$0")/.." && pwd)

ssh "$host" "mkdir -p '$dir/bench/results'"
rsync -a --delete \
   --exclude=.git --exclude='/golly' --exclude='/bgolly' \
   --exclude='gui-wx/Obj*/' --exclude='gui-wx/.ninja_*' \
   --exclude='*.o' --exclude='*.a' \
   --exclude='/bench/results/' --exclude='bench/.venv/' --exclude='__pycache__/' \
   "$repo/" "$host:$dir/"

ssh "$host" "cd '$dir/gui-wx' && make -f makefile-gtk -j\$(nproc) OBJDIR=ObjRel ENABLE_SOUND= $makeargs bgolly >/dev/null"

if [[ -n "$shcmd" ]]; then
   ssh "$host" "cd '$dir' && $shcmd"
elif [[ $# -gt 0 ]]; then
   args=$(printf '%q ' "$@")
   ssh "$host" "cd '$dir' && uv run --project bench bench/golly_bench.py $args"
fi

rhost=$(ssh "$host" hostname)
mkdir -p "$repo/bench/results/$rhost"
rsync -a "$host:$dir/bench/results/" "$repo/bench/results/$rhost/"
