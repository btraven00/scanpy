"""W1 acceptance: ndimr.py (all-gene PCA, --pca_type), run as the benchmark
runs it, on a small synthetic matrix."""

import subprocess
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from conftest import ROOT
from pca import scale

rng = np.random.default_rng(0)
COUNTS = rng.poisson(rng.gamma(1.0, 2.0, size=(60, 40))).astype(np.float64)  # cells x genes
COUNTS[:, 5] = 0            # constant gene: scaling leaves it at 0, not unit variance
CELLS = [f"c{i}" for i in range(60)]
GENES = [f"g{j}" for j in range(40)]


def write_tenx(path, X):
    """TENx layout: genes x cells CSC, as NORM/FEAT write it."""
    m = sp.csc_matrix(X.T)
    with h5py.File(path, "w") as h5:
        g = h5.create_group("matrix")
        for k in ("data", "indices", "indptr"):
            g[k] = getattr(m, k)
        g["shape"] = np.array(m.shape)
        g["genes"] = np.array(GENES, dtype="S")
        g["barcodes"] = np.array(CELLS, dtype="S")


def run(script, out, name, *args):
    subprocess.run([sys.executable, str(ROOT / script), "--output_dir", str(out),
                    "--name", name, *map(str, args)], check=True, cwd=ROOT)
    read = lambda s: pd.read_csv(out / f"{name}_{s}.tsv", sep="\t", index_col=0)
    return read("embedding"), read("loadings")


@pytest.fixture(scope="module")
def norm_h5(tmp_path_factory):
    p = tmp_path_factory.mktemp("in") / "norm.h5"
    write_tenx(p, np.log1p(COUNTS))
    return p


def test_ndimr_reproduces_pca(norm_h5, tmp_path):
    common = ["--solver", "arpack", "--n_components", 5, "--random_seed", 42]
    pe, pl = run("pca.py", tmp_path, "pc", "--normalized_selected_h5", norm_h5, *common)
    ne, nl = run("ndimr.py", tmp_path, "nd", "--normalized_h5", norm_h5, *common)
    assert list(ne.index) == list(pe.index) == CELLS
    np.testing.assert_allclose(ne.to_numpy(), pe.to_numpy(), atol=1e-10)
    np.testing.assert_allclose(nl.to_numpy(), pl.to_numpy(), atol=1e-10)


def test_ndimr_standardized(norm_h5, tmp_path):
    common = ["--normalized_h5", norm_h5, "--solver", "arpack", "--n_components", 5,
              "--random_seed", 42]
    e0, _ = run("ndimr.py", tmp_path, "s0", *common, "--pca_type", "centered")
    e1, _ = run("ndimr.py", tmp_path, "s1", *common, "--pca_type", "standardized")
    assert list(e1.index) == CELLS
    assert not np.allclose(np.abs(e0.to_numpy()), np.abs(e1.to_numpy()))

    a = ad.AnnData(X=sp.csr_matrix(np.log1p(COUNTS)))
    scale(a)
    v = np.var(np.asarray(a.X), axis=0, ddof=1)
    expressed = COUNTS.std(axis=0) > 0
    np.testing.assert_allclose(v[expressed], 1.0, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(a.X).mean(axis=0), 0.0, atol=1e-12)
