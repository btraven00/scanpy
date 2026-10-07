"""Fusion must be invisible: a fused PCA,NNG,CLUST run writes the same artifacts
as three split runs chained through files. A mismatch means either a lossy
writer/reader round-trip or a step that is not pure."""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import fuse
from steps import STEPS, GraphT, plan

PARAMS = {"solver": "arpack", "n_components": "10", "dtype": "float64", "n_neighbors": "10",
          "knn_transformer": "sklearn", "resolution": "1.0", "leiden_flavor": "igraph", "random_seed": "0"}


def _params(stages):
    return [x for s in stages for k in STEPS[s].params for x in (f"--{s.lower()}_{k}", PARAMS[k])]


P = _params(["PCA", "NNG", "CLUST"])


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


# --- purity guards (tests only; never in timed runs) --------------------------

import sys
from contextlib import contextmanager, nullcontext

import scipy.sparse as sp

from steps import Clusters, Embedding, Graph, Matrix

_guard = {"dirs": None}


def _hook(event, args):
    dirs = _guard["dirs"]
    if dirs is None:
        return
    if event in ("socket.connect", "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"):
        raise RuntimeError(f"impure step: {event}")
    if event == "open":
        path, mode = str(args[0]), args[1] or "r"
        if any(c in str(mode) for c in "wax+") or any(path.startswith(d) for d in dirs):
            raise RuntimeError(f"impure step: open({path!r}, {mode!r})")


sys.addaudithook(_hook)  # can't be removed; inert while _guard["dirs"] is None


@contextmanager
def no_io(*dirs):
    """Inside: no writes anywhere, no reads under `dirs` (the data), no network or subprocesses.
    Blind spot: C libraries that open files themselves (e.g. HDF5) bypass audit hooks."""
    _guard["dirs"] = [str(d) for d in dirs]
    try:
        yield
    finally:
        _guard["dirs"] = None


def freeze(x):
    """Make every array reachable from a step input read-only, so in-place edits raise."""
    if isinstance(x, np.ndarray):
        x.flags.writeable = False
    elif sp.issparse(x):
        for a in (x.data, x.indices, x.indptr):
            a.flags.writeable = False
    elif isinstance(x, ad.AnnData):
        freeze(x.X)
    elif isinstance(x, Embedding):
        freeze(x.matrix)
    elif isinstance(x, Graph):
        freeze(x.distances)
        freeze(x.connectivities)
    return x


def _step_params(stage):
    return {k: t(PARAMS[k]) for k, t in STEPS[stage].params.items()}


def _chain(adata, guard):
    env = {"data_h5ad": freeze(adata)}
    for stage in ("PCA", "NNG", "CLUST"):
        st = STEPS[stage]
        with guard():
            out = st.run({k: env[k] for k in st.inputs}, _step_params(stage))
        assert set(out) == set(st.outputs)
        env.update({k: freeze(v) for k, v in out.items()})
    return env


def test_steps_are_pure(tmp_path, h5ad):
    adata = Matrix.load(h5ad)
    _chain(adata, nullcontext)  # unguarded warm-up: lazy imports and JIT caches touch the filesystem
    env = _chain(adata, lambda: no_io(tmp_path))
    assert isinstance(env["clusters_tsv"], Clusters)
    assert "X_pca" not in adata.obsm, "PCA wrote into its input"


def test_guards_fire(tmp_path):
    (tmp_path / "side.txt").write_text("x")
    with pytest.raises(RuntimeError, match="impure"), no_io(tmp_path):
        open(tmp_path / "side.txt").read()
    with pytest.raises(ValueError, match="read-only"):
        freeze(np.zeros(3))[0] = 1
