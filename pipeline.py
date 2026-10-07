#!/usr/bin/env python3
"""End-to-end PCA -> kNN -> Leiden in one process (sc-brrr `pipeline` stage).

Chains this module's own pca / knn / cluster functions instead of re-implementing
them, but keeps the AnnData in memory between steps: the split-stage plan pays a
disk round-trip per step, which is exactly what sc-brrr times.

Input: --data_h5ad, log-normalised, HVG-selected (the published prep output;
cells x genes, obs index = cell ids).

Outputs ({output_dir}/{name}_*), same formats as the split-stage entrypoints:
  _embedding.tsv   pca.py   (cell_id, PC1..PCn)
  _neighbors.h5    knn.py   (distances CSR + /connectivities)
  _clusters.tsv    cluster.py (cell_id, cluster)
Per-step timings are obkit phase events (load, pca, knn, cluster, write).
"""

import argparse
import sys
from pathlib import Path

import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import cli  # noqa: E402
from writers import Embedding, write_embeddings  # noqa: E402
from phases import phase  # noqa: E402
from obkit.logger import init_logger  # noqa: E402

from pca import run_pca  # noqa: E402
from knn import write_neighbors_graph  # noqa: E402
from cluster import cluster_leiden  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="PCA -> kNN -> Leiden (scanpy, in-process)")
    cli.add_base_args(p)  # --output_dir, --name
    p.add_argument("--data_h5ad", type=Path, required=True)
    # pca.run_pca reads these off args
    p.add_argument("--solver", choices=["arpack", "full"], required=True)
    p.add_argument("--n_components", type=int, required=True)
    p.add_argument("--blas_threads", type=int, default=0)
    p.add_argument("--n_neighbors", type=int, required=True)
    # scanpy's default (None) is pynndescent above 4096 cells, i.e. approximate;
    # the sc-brrr reference method needs sklearn (exact).
    p.add_argument("--knn_transformer", choices=["sklearn", "pynndescent"], required=True)
    p.add_argument("--resolution", type=float, required=True)
    p.add_argument("--leiden_flavor", choices=["igraph", "leidenalg"], required=True)
    p.add_argument("--random_seed", type=int, required=True)
    # Not used by the code. It gives each same-seed replicate its own parameter hash, and hence its own output dir.
    p.add_argument("--replicate", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    init_logger(str(out))

    with phase("load") as attrs:
        adata = sc.read_h5ad(args.data_h5ad)
        attrs["n_cells"], attrs["n_genes"] = adata.shape

    with phase("pca") as attrs:
        embedding, _, _, _ = run_pca(adata, args)
        attrs["solver"] = args.solver

    with phase("knn") as attrs:
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep="X_pca",
                        transformer=args.knn_transformer, random_state=args.random_seed)
        attrs["transformer"] = args.knn_transformer

    with phase("cluster"):
        labels = cluster_leiden(adata, args.leiden_flavor, "RBConfiguration",
                                args.resolution, args.random_seed)

    with phase("write"):
        cells = list(adata.obs_names)
        cols = [f"PC{i + 1}" for i in range(embedding.shape[1])]
        write_embeddings(Embedding(embedding, cells, cols), out / f"{args.name}_embedding.tsv")
        write_neighbors_graph(adata, out, args.name)
        with open(out / f"{args.name}_clusters.tsv", "w") as f:
            f.write("cell_id\tcluster\n")
            f.writelines(f"{c}\t{k}\n" for c, k in zip(cells, labels))

    assert embedding.shape == (adata.n_obs, args.n_components), embedding.shape
    assert len(labels) == adata.n_obs


if __name__ == "__main__":
    main()
