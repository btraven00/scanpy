#!/usr/bin/env python3
"""Pick a Leiden resolution from a CLUST-SWEEP table with scSHC (proof of concept).

For each candidate resolution (a column of {name}_clusters_sweep.tsv), scSHC's
testClusters tests the partition's splits against a single-population null and
merges the unsupported ones. The pick is the HIGHEST resolution at which every
cluster is supported (supported k == input k); if none is, the lowest candidate.
No ground-truth labels are used.

Scored on a fixed random subsample of cells (--n_cells) for cost; clusters with
fewer than --min_cluster cells in the subsample are dropped first (testClusters
returns K = 1 when the root split isolates a tiny cluster).

POC: scSHC is GitHub-only (not on conda), so the R side runs in an external
environment given by --rscript (default: an existing pixi env). Not portable.

Writes {output_dir}/{name}_pick.json: {"resolution": r, "rule": ..., "scores": [...]}.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--clusters_sweep_tsv", type=Path, required=True)
    p.add_argument("--rawdata_h5ad", type=Path, required=True)
    p.add_argument("--candidates", type=str, default="0.05:0.65:0.15",
                   help="MIN:MAX:STEP of sweep resolutions to score (nearest sweep column each)")
    p.add_argument("--n_cells", type=int, default=5000)
    p.add_argument("--min_cluster", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--cores", type=int, default=8)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--rscript", type=str, default="/home/b/phd/HBCA_denoise/.pixi/envs/default/bin/Rscript")
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    sw = pd.read_csv(a.clusters_sweep_tsv, sep="\t", index_col=0)
    sw.columns = sw.columns.astype(float)
    lo, hi, step = (float(x) for x in a.candidates.split(":"))
    wanted = np.arange(lo, hi + 1e-9, step)
    cols = sorted({sw.columns[np.abs(sw.columns - w).argmin()] for w in wanted})

    rng = np.random.default_rng(a.random_seed)
    cells = np.sort(rng.choice(sw.index.to_numpy(), size=min(a.n_cells, len(sw)), replace=False))
    raw = ad.read_h5ad(a.rawdata_h5ad, backed="r")
    idx = raw.obs_names.get_indexer(cells)
    if (idx < 0).any():
        sys.exit(f"{(idx < 0).sum()} sweep cells missing from {a.rawdata_h5ad}")
    order = np.argsort(idx)
    X = sp.csr_matrix(raw.layers["counts"][idx[order]]); cells = cells[order]
    print(f"  scoring {len(cols)} resolutions on {X.shape[0]} cells x {X.shape[1]} genes", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        scipy.io.mmwrite(tmp / "counts.mtx", X.T.tocsc().astype(np.float64))
        pd.Series(raw.var_names).to_csv(tmp / "genes.txt", index=False, header=False)
        part = sw.loc[cells, cols].astype(str); part.columns = [f"{c:g}" for c in cols]
        part.to_csv(tmp / "partitions.csv")
        subprocess.run([a.rscript, str(HERE / "pick_scshc.R"), str(tmp), str(a.alpha), str(a.min_cluster),
                        str(a.cores)], check=True)
        scores = pd.read_csv(tmp / "scores.csv")

    ok = scores[scores.k_supported == scores.k_in]
    pick = float(ok.resolution.max()) if len(ok) else float(scores.resolution.min())
    res = dict(resolution=pick, rule="highest resolution with every cluster supported by scSHC (alpha "
               f"{a.alpha}); lowest candidate if none", n_cells=int(X.shape[0]),
               scores=scores.to_dict(orient="records"))
    (out / f"{a.name}_pick.json").write_text(json.dumps(res, indent=1))
    print(scores.to_string(index=False)); print(f"  pick: resolution {pick}")


if __name__ == "__main__":
    main()
