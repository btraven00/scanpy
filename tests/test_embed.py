"""W1/W2 acceptance: ndimr.py (all-gene PCA, --scale) and rdimr.py (log-CP10k
PCA on raw counts), run as the benchmark runs them, on a small synthetic matrix."""

import gzip
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
COUNTS[:, 5] = 0            # never expressed: min_gene_cells drops it
COUNTS[:, 6] = 0
COUNTS[0, 6] = 3            # expressed in exactly one cell
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


def test_ndimr_scale(norm_h5, tmp_path):
    common = ["--normalized_h5", norm_h5, "--solver", "arpack", "--n_components", 5,
              "--random_seed", 42]
    e0, _ = run("ndimr.py", tmp_path, "s0", *common, "--scale", "false")
    e1, _ = run("ndimr.py", tmp_path, "s1", *common, "--scale", "true")
    assert list(e1.index) == CELLS
    assert not np.allclose(np.abs(e0.to_numpy()), np.abs(e1.to_numpy()))

    a = ad.AnnData(X=sp.csr_matrix(np.log1p(COUNTS)))
    scale(a)
    v = np.var(np.asarray(a.X), axis=0, ddof=1)
    expressed = COUNTS.std(axis=0) > 0
    np.testing.assert_allclose(v[expressed], 1.0, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(a.X).mean(axis=0), 0.0, atol=1e-12)


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    a = ad.AnnData(X=None, obs=pd.DataFrame(index=CELLS), var=pd.DataFrame(index=GENES))
    a.layers["counts"] = sp.csr_matrix(COUNTS)
    a.write_h5ad(d / "raw.h5ad")
    keep_c, keep_g = CELLS[::-1][:50], GENES[:30]   # reordered and subset, as FILT does
    for f, ids in (("cells.txt.gz", keep_c), ("genes.txt.gz", keep_g)):
        with gzip.open(d / f, "wt") as fh:
            fh.write("\n".join(ids) + "\n")
    return d, keep_c, keep_g


def test_rdimr_matches_logcp10k_pca(raw, tmp_path):
    d, keep_c, keep_g = raw
    e, l = run("rdimr.py", tmp_path, "rd", "--rawdata_h5ad", d / "raw.h5ad",
               "--filtered_cellids", d / "cells.txt.gz", "--filtered_featureids",
               d / "genes.txt.gz", "--n_components", 4, "--random_seed", 42,
               "--min_gene_cells", 2)
    # the omni-bipca filter: g5 (0 cells) and g6 (1 cell) go; order is FILT's
    genes = [g for g in keep_g if g not in ("g5", "g6")]
    assert list(e.index) == keep_c and list(l.index) == genes

    X = pd.DataFrame(COUNTS, index=CELLS, columns=GENES).loc[keep_c, genes].to_numpy()
    Y = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)
    U, S, _ = np.linalg.svd(Y - Y.mean(axis=0), full_matrices=False)
    ref = U[:, :4] * S[:4]
    got = e.to_numpy()
    np.testing.assert_allclose(np.abs(got), np.abs(ref), atol=1e-8)   # up to sign


def test_rdimr_rejects_non_integer(raw, tmp_path):
    d, _, _ = raw
    a = ad.read_h5ad(d / "raw.h5ad")
    a.layers["counts"] = a.layers["counts"] * 0.5
    a.write_h5ad(tmp_path / "bad.h5ad")
    r = subprocess.run([sys.executable, str(ROOT / "rdimr.py"), "--output_dir", str(tmp_path),
                        "--name", "x", "--rawdata_h5ad", str(tmp_path / "bad.h5ad"),
                        "--filtered_cellids", str(d / "cells.txt.gz"),
                        "--filtered_featureids", str(d / "genes.txt.gz"),
                        "--n_components", "4", "--random_seed", "42"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "non-integer" in r.stderr
