"""Shared test fixtures.

The heavy ones come from the benchmark's published be1 fixture dataset — the
same files the DATA stage serves to a real run — rather than hand-rolled
synthetic blobs, so what the tests exercise is the covariance structure the
module actually meets in production. They are downloaded once into a user
cache; with no network the tests that need them skip instead of failing.
"""

import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

FIXTURE_URL = "https://omnibenchmark.mls.uzh.ch/datasets/be1-fixture/"
CACHE = Path(
    os.environ.get("OMNI_FIXTURE_CACHE")
    or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "omni-scrna-fixtures"
)


def _fixture_file(name):
    """Path to a be1 fixture file, downloading it into CACHE on first use."""
    dest = CACHE / name
    if not dest.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            urlretrieve(FIXTURE_URL + name, part)
        except (URLError, OSError) as e:
            part.unlink(missing_ok=True)
            pytest.skip(f"be1 fixture {name} unavailable ({e}); "
                        f"point OMNI_FIXTURE_CACHE at a local copy to run offline")
        part.rename(dest)
    return dest


@pytest.fixture(scope="session")
def be1_h5ad():
    """be1: 2000 cells x 36753 genes, raw counts in layers['counts'],
    8 balanced cell lines in obs['clusters.truth']."""
    return _fixture_file("be1-fixture.h5ad")


@pytest.fixture(scope="session")
def be1_truth():
    """cell_id -> ground-truth label, from the published truth TSV."""
    rows = _fixture_file("be1-fixture.clusters_truth.tsv").read_text().splitlines()[1:]
    return dict(ln.split("\t", 1) for ln in rows if ln)


@pytest.fixture(scope="session")
def be1_pcas_tsv(be1_h5ad, tmp_path_factory):
    """A real PCA embedding of be1, in the TSV layout knn.py reads.

    Standard scanpy preprocessing, i.e. what the NORM/FEAT/PCA stages upstream
    of this module do; 20 PCs is plenty to separate the cell lines and keeps the
    subprocess runs cheap.
    """
    import anndata as ad
    import scanpy as sc

    a = ad.read_h5ad(be1_h5ad)
    a.X = a.layers["counts"]
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, subset=True)
    sc.pp.pca(a, n_comps=20, svd_solver="arpack", random_state=0)

    X = a.obsm["X_pca"]
    p = tmp_path_factory.mktemp("be1") / "be1_pcas.tsv"
    with open(p, "w") as fh:
        fh.write("cell_id\t" + "\t".join(f"PC{i + 1}" for i in range(X.shape[1])) + "\n")
        for cid, row in zip(a.obs_names, X):
            fh.write(cid + "\t" + "\t".join(repr(float(v)) for v in row) + "\n")
    return p
