"""Fusion must be invisible: a fused PCA,NNG,CLUST run writes the same artifacts
as three split runs chained through files. A mismatch means either a lossy
writer/reader round-trip or a step that is not pure."""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import fuse
from steps import STEPS, GraphT, plan

P = ["--solver", "arpack", "--n_components", "10", "--n_neighbors", "10",
     "--knn_transformer", "sklearn", "--resolution", "1.0", "--leiden_flavor", "igraph",
     "--random_seed", "0"]


def _params(stages):
    keep = {k for s in stages for k in STEPS[s].params}
    return [x for i in range(0, len(P), 2) if P[i][2:] in keep for x in P[i:i + 2]]


@pytest.fixture
def h5ad(tmp_path):
    rng = np.random.default_rng(0)
    X = sp.random(300, 60, density=0.2, format="csr", random_state=rng, dtype=np.float32)
    a = ad.AnnData(X=X)
    a.obs_names = [f"c{i}" for i in range(300)]
    a.write_h5ad(tmp_path / "in.h5ad")
    return tmp_path / "in.h5ad"


def test_fused_equals_split(tmp_path, h5ad):
    f, s = tmp_path / "fused", tmp_path / "split"
    fuse.main(["--output_dir", str(f), "--name", "x", "--steps", "PCA,NNG,CLUST", "--data_h5ad", str(h5ad)] + P)
    fuse.main(["--output_dir", str(s), "--name", "x", "--steps", "PCA", "--data_h5ad", str(h5ad)] + _params(["PCA"]))
    fuse.main(["--output_dir", str(s), "--name", "x", "--steps", "NNG",
               "--embedding_tsv", str(s / "x_embedding.tsv")] + _params(["NNG"]))
    fuse.main(["--output_dir", str(s), "--name", "x", "--steps", "CLUST",
               "--neighbors_h5", str(s / "x_neighbors.h5")] + _params(["CLUST"]))

    for name in ("x_embedding.tsv", "x_clusters.tsv"):
        assert (f / name).read_bytes() == (s / name).read_bytes(), name
    gf, gs = GraphT.load(f / "x_neighbors.h5"), GraphT.load(s / "x_neighbors.h5")
    assert gf.cell_ids == gs.cell_ids
    for m in ("distances", "connectivities"):
        assert (getattr(gf, m) != getattr(gs, m)).nnz == 0, m


def test_chain_typecheck():
    assert set(plan(["PCA", "NNG", "CLUST"])) == {"data_h5ad"}
    assert set(plan(["NNG"])) == {"embedding_tsv"}
