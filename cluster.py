#!/usr/bin/env python3
"""Leiden clustering module (scanpy-backed) for omnibenchmark.

Reads {name}_neighbors.h5 (the kNN distances + connectivities from the knn
entrypoint), runs Leiden on the connectivities, and writes:
  {output_dir}/{name}_clusters.tsv  — cell_id<TAB>cluster

Flavor / partition type
------------------------
--flavor igraph     sc.tl.leiden(flavor="igraph")         — igraph C-core
--flavor leidenalg  sc.tl.leiden(flavor="leidenalg", ...) — leidenalg backend

When --flavor leidenalg, --partition_type selects the objective:
  RBConfiguration   resolution-aware (Reichardt-Bornholdt)
  CPM               resolution-aware (Constant Potts Model)
  Modularity        no resolution (classic modularity)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import polars as pl
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import cli  # noqa: E402
from readers import read_neighbors  # noqa: E402

log = logging.getLogger(__name__)

_PARTITION_CLASSES = {
    "RBConfiguration": "RBConfigurationVertexPartition",
    "CPM": "CPMVertexPartition",
    "Modularity": "ModularityVertexPartition",
}
_RESOLUTION_AWARE = {"RBConfiguration", "CPM"}


def parse_args():
    # We own the parser; src/common/cli injects the shared contract (base args + the
    # `CLUST` stage I/O from common/schema). This module's method params are
    # hand-rolled below, so the whole CLI stays visible here.
    p = argparse.ArgumentParser(description="Leiden clustering module (scanpy-backed)")
    cli.add_base_args(p)            # --output_dir, --name
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--neighbors_h5", dest="neighbors_h5", type=Path,
                   help="H5 file with neighbors")
    g.add_argument("--neighbors_corrected_h5", dest="neighbors_h5", type=Path,
                   help="H5 file with neighbors (corrected embedding)")
    p.add_argument("--resolution", type=float, default=None,
                   help="Resolution controlling cluster granularity")
    # One job, many resolutions: the graph is loaded once and Leiden runs at every
    # point of a linear grid. Writes {name}_clusters_sweep.tsv (cell_id + one column
    # per resolution) and {name}_sweep.json instead of {name}_clusters.tsv.
    p.add_argument("--sweep", type=str, default=None,
                   help="MIN:MAX:STEP resolution grid (inclusive); replaces --resolution")
    p.add_argument("--random_seed", type=int, required=True, help="Random seed")
    p.add_argument("--flavor", choices=["igraph", "leidenalg"], required=True,
                   help="Leiden backend flavor")
    # Leiden iterations. Negative = until an iteration no longer improves quality
    # (true until-convergence; supported by both igraph and leidenalg). Default 2
    # keeps this module's historical setting. Not the same unit as rapids'
    # max_iter (cuGraph levels, early stop) or Seurat's n.iter (leidenbase, >= 1,
    # no early stop) -- see the module README.
    p.add_argument("--n_iterations", type=int, default=2,
                   help="Leiden iterations; negative = until no improvement")
    p.add_argument("--partition_type", choices=list(_PARTITION_CLASSES),
                   default="RBConfiguration",
                   help="Partition type (only used when --flavor leidenalg)")
    a = p.parse_args()
    if (a.resolution is None) == (a.sweep is None):
        p.error("give exactly one of --resolution or --sweep")
    return a


def sweep_grid(spec):
    """'MIN:MAX:STEP' -> resolutions MIN, MIN+STEP, ... up to MAX inclusive."""
    lo, hi, step = (float(x) for x in spec.split(":"))
    if not (0 < lo <= hi and step > 0):
        raise ValueError(f"bad --sweep {spec!r}: need 0 < MIN <= MAX and STEP > 0")
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    if n > 500:
        raise ValueError(f"--sweep {spec!r} gives {n} resolutions; cap is 500")
    return [round(lo + i * step, 10) for i in range(n)]


def build_adata(neighbors_h5):
    """AnnData with the stored distances/connectivities graph wired up for scanpy."""
    distances, connectivities, cell_ids = read_neighbors(neighbors_h5)
    adata = ad.AnnData(X=np.zeros((len(cell_ids), 1)))
    adata.obs_names = cell_ids
    adata.obsp["distances"] = distances
    adata.obsp["connectivities"] = connectivities
    adata.uns["neighbors"] = {
        "distances_key": "distances",
        "connectivities_key": "connectivities",
    }
    return adata, cell_ids


def cluster_leiden(adata, flavor, partition_type, resolution, random_seed, n_iterations=2):
    if flavor == "igraph": # TODO(Noemi): we want it as a constant not hard coded
        sc.tl.leiden(adata, resolution=resolution, random_state=random_seed,
                     flavor="igraph", n_iterations=n_iterations, directed=False)
    else:
        import leidenalg #TODO(Noemi): possibly change and put into env 
        pt = getattr(leidenalg, _PARTITION_CLASSES[partition_type])
        kw = dict(flavor="leidenalg", partition_type=pt, #TODO(Noemi): iteration strategy? -1 is to convergence, 2 is faster
                  random_state=random_seed, n_iterations=n_iterations, directed=False)
        if partition_type in _RESOLUTION_AWARE:
            kw["resolution"] = resolution
        elif resolution != 1.0:
            log.warning("%s does not support resolution — value %.4f ignored.",
                        partition_type, resolution)
        sc.tl.leiden(adata, **kw)
    return adata.obs["leiden"].to_list()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "neighbors_h5", "flavor", "partition_type",
              "resolution", "sweep", "random_seed", "n_iterations"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    adata, cell_ids = build_adata(args.neighbors_h5)
    if args.sweep:
        import json, time
        cols, info = {"cell_id": cell_ids}, []
        for r in sweep_grid(args.sweep):
            t0 = time.perf_counter()
            lab = cluster_leiden(adata, args.flavor, args.partition_type, r, args.random_seed, args.n_iterations)
            info.append(dict(resolution=r, n_clusters=len(set(lab)), seconds=round(time.perf_counter() - t0, 3)))
            cols[f"{r:g}"] = lab
            print(f"  resolution {r:g}: {info[-1]['n_clusters']} clusters ({info[-1]['seconds']} s)", flush=True)
        out = Path(args.output_dir) / f"{args.name}_clusters_sweep.tsv"
        pl.DataFrame(cols).write_csv(out, separator="\t")
        (Path(args.output_dir) / f"{args.name}_sweep.json").write_text(json.dumps(
            dict(sweep=args.sweep, flavor=args.flavor, random_seed=args.random_seed,
                 n_iterations=args.n_iterations, resolutions=info), indent=1))
        print(f"  wrote: {out}")
        return
    labels = cluster_leiden(
        adata, args.flavor, args.partition_type, args.resolution, args.random_seed, args.n_iterations
    )

    out = Path(args.output_dir) / f"{args.name}_clusters.tsv"
    pl.DataFrame({"cell_id": cell_ids, "cluster": labels}).write_csv(out, separator="\t")
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
