"""Typed, pure steps for fusing stage entrypoints (prototype).

Every artifact type knows how to load and save itself, and every step is
`run(inputs, params) -> outputs` over those types. It touches no files. A runner
(fuse.py) can then chain steps in memory whenever one step's output type is the
next step's input type, and call load/save only at the edges. A single step run
alone is the split-stage behaviour: load -> run -> save.

Output ids and file layouts are the omni-scrna stage contracts, so a fused run
and a split run write the same files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import polars as pl
import scanpy as sc
import scipy.sparse as sp

from writers import Embedding, write_embeddings
from readers import read_neighbors


# --- types -------------------------------------------------------------------

class Matrix:
    """Cells x genes, log-normalised (an AnnData). Input only: the published data."""

    @staticmethod
    def load(path):
        path = Path(path)
        if path.suffix == ".h5ad":
            return sc.read_h5ad(path)
        from pca import load_matrix  # omni-scrna TENx layout (normalized_selected_h5)
        return load_matrix(path)


class EmbeddingT:
    """{name}_embedding.tsv: cell_id + PC1..PCn (writers.Embedding)."""
    suffix = "_embedding.tsv"

    @staticmethod
    def load(path):
        # N header names, N+1 data columns (first = unnamed row ids); see knn.py
        df = pl.read_csv(path, separator="\t", skip_rows=1, has_header=False)
        cols = pl.read_csv(path, separator="\t", n_rows=0).columns[1:]
        return Embedding(df[:, 1:].to_numpy().astype(np.float64), df[:, 0].to_list(), list(cols))

    @staticmethod
    def save(obj, path):
        write_embeddings(obj, path)


@dataclass
class Graph:
    distances: sp.csr_matrix
    connectivities: sp.csr_matrix
    cell_ids: list = field(default_factory=list)


class GraphT:
    """{name}_neighbors.h5: distances CSR at the root + /connectivities (knn.py layout)."""
    suffix = "_neighbors.h5"

    @staticmethod
    def load(path):
        d, c, ids = read_neighbors(path)
        return Graph(d, c, ids)

    @staticmethod
    def save(obj, path):
        import h5py
        with h5py.File(path, "w") as h5:
            h5.create_dataset("cell_ids", data=np.array(obj.cell_ids, dtype="S"))
            for grp, m in ((h5, obj.distances), (h5.create_group("connectivities"), obj.connectivities)):
                m = m.tocsr()
                grp.create_dataset("data", data=m.data)
                grp.create_dataset("indices", data=m.indices)
                grp.create_dataset("indptr", data=m.indptr)


@dataclass
class Clusters:
    cell_ids: list
    labels: list


class ClustersT:
    """{name}_clusters.tsv: cell_id<TAB>cluster."""
    suffix = "_clusters.tsv"

    @staticmethod
    def load(path):
        df = pl.read_csv(path, separator="\t", schema_overrides={"cluster": pl.String})
        return Clusters(df["cell_id"].to_list(), df["cluster"].to_list())

    @staticmethod
    def save(obj, path):
        pl.DataFrame({"cell_id": obj.cell_ids, "cluster": obj.labels}).write_csv(path, separator="\t")


# --- steps -------------------------------------------------------------------

@dataclass
class Step:
    stage: str
    inputs: dict   # input id -> type
    outputs: dict  # output id -> type
    params: dict   # param name -> python type
    run: callable


def _pca(ins, p):
    from pca import run_pca
    adata = ins["data_h5ad"]
    emb, *_ = run_pca(adata, SimpleNamespace(**p))
    return {"embedding_tsv": Embedding(emb, list(adata.obs_names), [f"PC{i + 1}" for i in range(emb.shape[1])])}


def _nng(ins, p):
    e = ins["embedding_tsv"]
    a = ad.AnnData(X=np.zeros((len(e.row_ids), 1)))
    a.obs_names = e.row_ids
    a.obsm["X_pca"] = e.matrix
    sc.pp.neighbors(a, n_neighbors=p["n_neighbors"], use_rep="X_pca",
                    transformer=p["knn_transformer"], random_state=p["random_seed"])
    return {"neighbors_h5": Graph(a.obsp["distances"], a.obsp["connectivities"], list(a.obs_names))}


def _clust(ins, p):
    from cluster import cluster_leiden
    g = ins["neighbors_h5"]
    a = ad.AnnData(X=np.zeros((len(g.cell_ids), 1)))
    a.obs_names = g.cell_ids
    a.obsp["distances"], a.obsp["connectivities"] = g.distances, g.connectivities
    a.uns["neighbors"] = {"distances_key": "distances", "connectivities_key": "connectivities"}
    labels = cluster_leiden(a, p["leiden_flavor"], "RBConfiguration", p["resolution"], p["random_seed"])
    return {"clusters_tsv": Clusters(g.cell_ids, labels)}


STEPS = {s.stage: s for s in [
    Step("PCA", {"data_h5ad": Matrix}, {"embedding_tsv": EmbeddingT},
         {"solver": str, "n_components": int, "blas_threads": int, "random_seed": int}, _pca),
    # scanpy's default (None) is pynndescent above 4096 cells, i.e. approximate; sklearn is exact.
    Step("NNG", {"embedding_tsv": EmbeddingT}, {"neighbors_h5": GraphT},
         {"n_neighbors": int, "knn_transformer": str, "random_seed": int}, _nng),
    Step("CLUST", {"neighbors_h5": GraphT}, {"clusters_tsv": ClustersT},
         {"resolution": float, "leiden_flavor": str, "random_seed": int}, _clust),
]}


def plan(stages):
    """Typecheck a chain. Returns the external inputs: ids that no earlier step produces."""
    produced, external = {}, {}
    for st in (STEPS[s] for s in stages):
        for k, t in st.inputs.items():
            if k in produced:
                assert produced[k] is t, f"{st.stage}: {k} is {produced[k].__name__}, wants {t.__name__}"
            else:
                external[k] = t
        produced.update(st.outputs)
    return external
