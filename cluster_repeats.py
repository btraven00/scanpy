#!/usr/bin/env python3
"""Leiden clustering repeated over seeds, for the CLUSTREP stage.

Same clustering as cluster.py, run --repeats times with random_seed + i, all in
one process. The point is entirely cost: on pancreas one CLUST job is ~3.8s of
which ~2.2s is interpreter and import, and scoring each one through CLUST-M
costs a further ~10.6s of which 7.4s is loading R's poem. Spawning 100 jobs to
do that pays both fixed costs 100 times -- ~1450s to perform, in the metric
stage, 2.8s of arithmetic. Internalising the loop pays them once.

The alternative was a gather stage over a seed sweep, which is the native shape
for this and would keep snakemake's parallelism. It does not work yet: gather
groups by ANCESTOR MODULE id, so every parameter expansion of CLUST lands in
one node and a seed sweep cannot be separated from a resolution sweep
(omnibenchmark docs/design/010-gather.md 3.2, label-based group_by is Phase 2).
Revisit when that lands -- N jobs parallelise and one job does not.

Writes {output_dir}/{name}_clusters_repeats.tsv:
  cell_id, repeat_0 .. repeat_{N-1}

Full labelings rather than just the per-repeat cluster COUNT, which is all the
modal-k metric needs. They cost nothing extra to write and they are the part
that cannot be recovered without re-running every clustering -- pairwise
ARI/NVI stability reads the same file. ponytail: plain TSV, ~5MB at 12k cells
x 100 repeats; gzip it if a dataset makes that hurt.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` package (src/common)
from common import cli  # noqa: E402
from readers import read_neighbors  # noqa: E402

# cluster.py owns the single-run path; import rather than restate it, so the
# repeats are the same clustering and not a second implementation of it.
sys.path.insert(0, str(Path(__file__).parent))
from cluster import build_adata, cluster_leiden  # noqa: E402


def spawn_seeds(base: int, n: int) -> list[int]:
    """n decorrelated seeds from one base, deterministically.

    NOT base + i. Those ranges overlap between base seeds -- base 42 and base
    50 with repeats=100 share 92 of their 100 seeds -- so sweeping the base to
    get a distribution OF stability would compare runs that are mostly the same
    runs. SeedSequence spawns non-overlapping streams instead, and is still a
    pure function of the base, so a repeat is reproducible from its plan line.

    Bounded to int32: the seed goes to igraph via leidenalg, not to numpy.
    """
    return [int(s.generate_state(1)[0] % (2**31 - 1))
            for s in np.random.SeedSequence(base).spawn(n)]


def parse_args():
    p = argparse.ArgumentParser(description="Repeated Leiden clustering (scanpy-backed)")
    cli.add_base_args(p)                # --output_dir, --name
    cli.add_stage_args(p, "CLUSTREP")   # --neighbors_h5
    p.add_argument("--resolution", type=float, required=True,
                   help="Resolution controlling cluster granularity")
    p.add_argument("--random_seed", type=int, required=True,
                   help="Base seed; the per-repeat seeds are spawned from it")
    # Mirror cluster.py's own method params exactly -- this entrypoint is that
    # clustering repeated, not a second configuration of it.
    p.add_argument("--flavor", choices=["igraph", "leidenalg"], required=True,
                   help="Leiden backend flavor")
    p.add_argument("--partition_type", default="RBConfiguration",
                   help="Partition type (only used when --flavor leidenalg)")
    p.add_argument("--repeats", type=int, default=1,
                   help="How many seeds to run (default 1: same result as the "
                        "plain cluster entrypoint, just in repeats form)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "neighbors_h5", "flavor", "partition_type",
              "resolution", "random_seed", "repeats"):
        print(f"  {k}: {getattr(args, k)}")
    if args.repeats < 1:
        sys.exit("--repeats must be >= 1")
    if args.repeats == 1:
        # Degenerate but legal: downstream stability is then trivially 1.0.
        print("LOG: repeats=1 -- stability will be vacuous, sweep it for a real number")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    adata, cell_ids = build_adata(args.neighbors_h5)

    cols = {"cell_id": cell_ids}
    for i, seed in enumerate(spawn_seeds(args.random_seed, args.repeats)):
        labels = cluster_leiden(adata, args.flavor, args.partition_type,
                                args.resolution, seed)
        cols[f"repeat_{i}"] = labels
        print(f"LOG: repeat {i} seed={seed} k={len(set(labels))}")

    ks = [len(set(v)) for key, v in cols.items() if key != "cell_id"]
    print(f"LOG: {args.repeats} repeats, k ranged {min(ks)}..{max(ks)}")

    out = Path(args.output_dir) / f"{args.name}_clusters_repeats.tsv"
    pl.DataFrame(cols).write_csv(out, separator="\t")
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
