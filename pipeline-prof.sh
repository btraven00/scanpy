#!/usr/bin/env bash
# pipeline.py under denet. A plain wrapper because ob 0.7.0 resolves an entrypoint
# value like "denet pipeline.py" as a file path and ends up running
# `python3 denet pipeline.py` (resolver._check_shebang).
# The trace lands next to pipeline.py's obkit-events.jsonl, and the two are joined on wall-clock time.
set -euo pipefail
out=""
for ((i = 1; i < $#; i++)); do [[ ${!i} == --output_dir ]] && { j=$((i + 1)); out=${!j}; }; done
[[ -n $out ]] || { echo "error: --output_dir is required" >&2; exit 2; }
mkdir -p "$out"
# --gpu: NVML sampling. Without a usable GPU, denet prints a warning and keeps going (CPU methods).
exec denet --json --quiet --gpu --write-env --out "$out/denet.jsonl" \
  run python3 "$(dirname "$0")/pipeline.py" "$@"
