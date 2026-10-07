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
from typing import Any, Callable, TypedDict, get_type_hints

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
# A step's contract is its typed signature: run(ins: <In>, p: <Params>) -> <Out>.
# mypy checks each implementation against it (`pixi run -e test typecheck`), and the
# runner reads the same TypedDicts at runtime, so the contract is written down once.

IO = {ad.AnnData: Matrix, Embedding: EmbeddingT, Graph: GraphT, Clusters: ClustersT}


class PCAIn(TypedDict):
    data_h5ad: ad.AnnData

class PCAOut(TypedDict):
    embedding_tsv: Embedding

class PCAParams(TypedDict):
    solver: str
    n_components: int
    blas_threads: int
    random_seed: int

def pca_step(ins: PCAIn, p: PCAParams) -> PCAOut:
    from pca import run_pca
    src = ins["data_h5ad"]
    # Fresh AnnData that shares X: sc.pp.pca writes obsm/varm/uns into the object
    # it gets, and the caller's input must not change. X is not copied.
    adata = ad.AnnData(X=src.X, obs=src.obs[[]], var=src.var[[]])
    emb, *_ = run_pca(adata, SimpleNamespace(**p))
    return {"embedding_tsv": Embedding(emb, list(adata.obs_names), [f"PC{i + 1}" for i in range(emb.shape[1])])}


class NNGIn(TypedDict):
    embedding_tsv: Embedding

class NNGOut(TypedDict):
    neighbors_h5: Graph

class NNGParams(TypedDict):
    n_neighbors: int
    knn_transformer: str  # sklearn = exact; scanpy's default is pynndescent (approximate) above 4096 cells
    random_seed: int

def nng_step(ins: NNGIn, p: NNGParams) -> NNGOut:
    e = ins["embedding_tsv"]
    a = ad.AnnData(X=np.zeros((len(e.row_ids), 1)))
    a.obs_names = e.row_ids
    a.obsm["X_pca"] = e.matrix
    sc.pp.neighbors(a, n_neighbors=p["n_neighbors"], use_rep="X_pca",
                    transformer=p["knn_transformer"], random_state=p["random_seed"])
    return {"neighbors_h5": Graph(a.obsp["distances"], a.obsp["connectivities"], list(a.obs_names))}


class CLUSTIn(TypedDict):
    neighbors_h5: Graph

class CLUSTOut(TypedDict):
    clusters_tsv: Clusters

class CLUSTParams(TypedDict):
    resolution: float
    leiden_flavor: str
    random_seed: int

def clust_step(ins: CLUSTIn, p: CLUSTParams) -> CLUSTOut:
    from cluster import cluster_leiden
    g = ins["neighbors_h5"]
    a = ad.AnnData(X=np.zeros((len(g.cell_ids), 1)))
    a.obs_names = g.cell_ids
    a.obsp["distances"], a.obsp["connectivities"] = g.distances, g.connectivities
    a.uns["neighbors"] = {"distances_key": "distances", "connectivities_key": "connectivities"}
    labels = cluster_leiden(a, p["leiden_flavor"], "RBConfiguration", p["resolution"], p["random_seed"])
    return {"clusters_tsv": Clusters(g.cell_ids, labels)}


@dataclass
class Step:
    stage: str
    run: Callable[..., Any]

    def _hints(self, arg):
        return get_type_hints(get_type_hints(self.run)[arg])

    @property
    def inputs(self):   # input id -> IO class
        return {k: IO[t] for k, t in self._hints("ins").items()}

    @property
    def outputs(self):  # output id -> IO class
        return {k: IO[t] for k, t in self._hints("return").items()}

    @property
    def params(self):   # param name -> python type
        return self._hints("p")


STEPS = {s.stage: s for s in [Step("PCA", pca_step), Step("NNG", nng_step), Step("CLUST", clust_step)]}

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
