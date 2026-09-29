#!/usr/bin/env python3
"""RDIMR module: log-CP10k + centred PCA on raw counts (rd-logpca).

The counterfactual for rd-bipca: same raw matrix, same cells and genes, same
filter, ordinary normalization instead of biwhitening. So the count reader and
the --min_gene_cells / empty-cell filter below are copied from
omni-bipca/pca.py and must stay identical to it; set --min_gene_cells to the
same value on both arms in the plan, or the gene set is no longer held fixed.

  counts[filtered cells, filtered genes] -> drop genes in < min_gene_cells cells
  -> drop cells left empty -> normalize_total(1e4) -> log1p -> [scale]
  -> sc.pp.pca(arpack, zero_center)

Outputs are pca.py's: {name}_embedding.tsv, {name}_loadings.tsv.
"""
import argparse
import gzip
import sys

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from pathlib import Path

import pca  # also puts src/ on sys.path
from pca import cli, phase, init_logger, run_pca, scale, write_outputs


def parse_args():
    p = argparse.ArgumentParser(description="RDIMR module: log-CP10k + PCA (scanpy)")
    cli.add_base_args(p)             # --output_dir, --name
    cli.add_stage_args(p, "RDIMR")   # --rawdata_h5ad, --filtered_cellids, --filtered_featureids
    p.add_argument("--n_components", type=int, required=True)
    p.add_argument("--random_seed", type=int, required=True)
    p.add_argument("--scale", type=str, default="false", choices=["true", "false"],
                   help="z-score each gene before PCA (sc.pp.scale, no clipping)")
    p.add_argument("--min_gene_cells", type=int, default=0,
                   help="drop genes expressed in fewer cells than this. MUST equal "
                        "rd-bipca's value, or the two arms see different gene sets")
    p.add_argument("--blas_threads", type=int, default=0,
                   help="BLAS threads (0 = inherit OMP_NUM_THREADS)")
    # ponytail: arpack only; on sparse input sklearn has nothing else (see pca.py).
    p.set_defaults(solver="arpack")
    return p.parse_args()


def read_ids(path):
    with gzip.open(path, "rt") as f:
        return [ln.strip() for ln in f if ln.strip()]


def read_counts(h5ad_path, cell_ids, gene_ids):
    """Raw counts for the given cells x genes, in the order given.
    Verbatim omni-bipca/pca.py:read_counts, except float64 out."""
    a = ad.read_h5ad(h5ad_path)
    X = a.layers["counts"] if "counts" in a.layers else a.X
    if X is None:
        sys.exit(f"error: {h5ad_path} has neither layers['counts'] nor X")
    missing = set(cell_ids) - set(a.obs_names)
    if missing:
        sys.exit(f"error: {len(missing)} filtered cell ids absent from the h5ad, "
                 f"e.g. {sorted(missing)[:3]}")
    missing = set(gene_ids) - set(a.var_names)
    if missing:
        sys.exit(f"error: {len(missing)} kept gene ids absent from the h5ad, "
                 f"e.g. {sorted(missing)[:3]}")
    ci = a.obs_names.get_indexer(cell_ids)
    gi = a.var_names.get_indexer(gene_ids)
    X = sp.csr_matrix(X)[ci][:, gi]
    d = X.data
    if d.size and not np.array_equal(d, np.rint(d)):
        sys.exit(f"error: {h5ad_path} counts hold non-integer values (min {d.min():.4g}, "
                 f"max {d.max():.4g}): log-CP10k is defined on raw counts")
    return X.astype(np.float64)


def filter_counts(X, cell_ids, gene_ids, min_gene_cells):
    """omni-bipca's filter: genes by --min_gene_cells, then cells left empty."""
    keep_g = np.asarray((X > 0).sum(axis=0)).ravel() >= min_gene_cells
    X, gene_ids = X[:, keep_g], [g for g, k in zip(gene_ids, keep_g) if k]
    keep_c = np.asarray(X.sum(axis=1)).ravel() > 0
    if not keep_c.all():
        print(f"  WARNING: dropping {int((~keep_c).sum())} cells with zero counts over "
              f"the surviving {X.shape[1]} genes; they will be absent downstream")
    X, cell_ids = X[keep_c], [c for c, k in zip(cell_ids, keep_c) if k]
    print(f"  after filtering: {X.shape}  (genes dropped: {int((~keep_g).sum())})")
    if min(X.shape) < 2:
        sys.exit("error: nothing left to factorize after filtering")
    return X, cell_ids, gene_ids


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(str(args.output_dir))

    with phase("load") as attrs:
        cell_ids = read_ids(args.filtered_cellids)
        gene_ids = read_ids(args.filtered_featureids)
        X = read_counts(args.rawdata_h5ad, cell_ids, gene_ids)
        print(f"  counts (cells x genes): {X.shape}")
        X, cell_ids, gene_ids = filter_counts(X, cell_ids, gene_ids, args.min_gene_cells)
        attrs["n_cells"], attrs["n_genes"] = X.shape

    with phase("normalize") as attrs:
        adata = ad.AnnData(X=X)
        adata.obs_names, adata.var_names = cell_ids, gene_ids
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if args.scale == "true":
            scale(adata)
        attrs["scale"] = args.scale == "true"
        attrs["min_gene_cells"] = args.min_gene_cells

    with phase("pca") as attrs:
        embedding, loadings, _, _ = run_pca(adata, args)
        attrs["solver"] = args.solver
        attrs["n_components"] = args.n_components
    print(f"  embedding: {embedding.shape}, loadings: {loadings.shape}")

    with phase("write"):
        write_outputs(args, embedding, loadings, cell_ids, gene_ids)


if __name__ == "__main__":
    main()
