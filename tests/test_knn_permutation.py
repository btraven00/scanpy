"""--permutation_seed: same seed must give the same graph, run to run.

The whole point of the flag is to measure how much cell order changes a
clustering. That measurement is only meaningful if the permutation itself is
reproducible -- otherwise a rerun is a different experiment and nothing is
comparable. These pin that, plus the invariants that let downstream modules
ignore the permutation entirely: outputs stay keyed by barcode, and the set of
cells is unchanged.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# The embedding is the be1 fixture's real PCA (conftest.be1_pcas_tsv): a
# synthetic three-blob cloud makes the approximate backend look far more
# order-stable than it is, which is exactly the property under test here.


def run(tmp, pcas, perm, name="t", transformer="pynndescent"):
    out = tmp / f"perm{perm}"
    subprocess.run(
        [sys.executable, str(ROOT / "knn.py"), "--output_dir", str(out), "--name", name,
         "--embedding_tsv", str(pcas), "--n_neighbors", "5", "--flavor", "umap",
         "--random_seed", "42", "--transformer", transformer,
         "--permutation_seed", str(perm)],
        check=True, capture_output=True,
    )
    return out / f"{name}_neighbors.h5"


def read(p):
    with h5py.File(p, "r") as h5:
        return ([s.decode() for s in h5["cell_ids"][:]],
                h5["indices"][:], h5["data"][:], h5["indptr"][:])


def test_same_permutation_seed_is_reproducible(tmp_path, be1_pcas_tsv):
    """The control the whole design rests on: rerun must be byte-identical."""
    a = read(run(tmp_path / "a", be1_pcas_tsv, 7))
    b = read(run(tmp_path / "b", be1_pcas_tsv, 7))
    assert a[0] == b[0], "cell order differed between runs at the same seed"
    for x, y in zip(a[1:], b[1:]):
        np.testing.assert_array_equal(x, y)


def test_permutation_seed_0_is_identity(tmp_path, be1_pcas_tsv):
    """0 is the control arm, so it must not shuffle anything."""
    src = [ln.split("\t")[0] for ln in be1_pcas_tsv.read_text().splitlines()[1:]]
    ids, *_ = read(run(tmp_path, be1_pcas_tsv, 0))
    assert ids == src, "seed 0 should preserve the input order"


def test_different_seeds_reorder_but_preserve_the_cell_set(tmp_path, be1_pcas_tsv):
    """Downstream modules join on barcode, so the SET must be invariant."""
    a, b = read(run(tmp_path / "a", be1_pcas_tsv, 1)), read(run(tmp_path / "b", be1_pcas_tsv, 2))
    assert a[0] != b[0], "different seeds gave the same order"
    assert set(a[0]) == set(b[0])
    assert len(a[0]) == len(set(a[0])) == len(b[0])


def test_on_disk_order_is_exactly_the_declared_permutation(tmp_path, be1_pcas_tsv):
    """The strongest form: the cell_ids WRITTEN TO THE FILE must equal the input
    ids reordered by default_rng(seed).permutation.

    The other tests only establish that *some* reordering happened. This pins
    which one, so a change of RNG, or applying the order to the embedding but
    not to the ids (which would silently decouple barcodes from coordinates),
    fails here instead of quietly corrupting every downstream join.
    """
    src = [ln.split("\t")[0] for ln in be1_pcas_tsv.read_text().splitlines()[1:]]
    for seed in (3, 11):
        ids, *_ = read(run(tmp_path / f"s{seed}", be1_pcas_tsv, seed))
        expected = [src[i] for i in np.random.default_rng(seed).permutation(len(src))]
        assert ids == expected, f"on-disk order is not rng({seed}).permutation"
        assert ids != src, "permuted run kept the input order"


def test_file_hash_is_stable_per_seed_and_differs_across_seeds(tmp_path, be1_pcas_tsv):
    """Strongest on-disk statement: the sha256 of the ARTIFACT itself.

    The other tests read arrays back out; this pins the bytes. Verified that
    h5py writes deterministically here, so a same-seed rerun is byte-identical.
    If HDF5 ever starts embedding a timestamp this test breaks loudly, which is
    the right outcome -- byte-stable artifacts are what makes an omnibenchmark
    output hash meaningful.
    """
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    a1 = sha(run(tmp_path / "a1", be1_pcas_tsv, 3))
    a2 = sha(run(tmp_path / "a2", be1_pcas_tsv, 3))
    assert a1 == a2, "same seed produced different bytes"
    assert sha(run(tmp_path / "b", be1_pcas_tsv, 7)) != a1
    assert sha(run(tmp_path / "c", be1_pcas_tsv, 0)) != a1


def _edges(ids, indices, indptr):
    """Edges as barcode pairs, so they are comparable across permutations."""
    return {(ids[r], ids[c]) for r in range(len(ids))
            for c in indices[indptr[r]:indptr[r + 1]]}


def test_permutation_actually_perturbs_the_approximate_graph(tmp_path, be1_pcas_tsv):
    """The flag exists to perturb the graph. If this ever passes as 'identical',
    the flag has become a no-op and every stability number measured with it is
    meaningless -- so assert the perturbation, not its absence.

    (Exact backends ARE order-invariant; this asserts the approximate one is not,
    which is the whole phenomenon under study.)
    """
    base = read(run(tmp_path / "a", be1_pcas_tsv, 0))
    perm = read(run(tmp_path / "b", be1_pcas_tsv, 3))
    assert _edges(base[0], base[1], base[3]) != _edges(perm[0], perm[1], perm[3])


def test_degrees_stay_uniform(tmp_path, be1_pcas_tsv):
    """Structural sanity on the writer: every cell keeps k neighbours. Weak by
    design -- scanpy emits a fixed row length -- but it catches a truncated or
    misaligned CSR, which the barcode-level checks would not."""
    for seed in (0, 3):
        _, _, _, indptr = read(run(tmp_path / f"s{seed}", be1_pcas_tsv, seed))
        assert len(set(np.diff(indptr).tolist())) == 1


def _by_barcode(p, group=""):
    """Graph as {(barcode, barcode): weight}, comparable across permutations."""
    with h5py.File(p, "r") as h5:
        ids = [s.decode() for s in h5["cell_ids"][:]]
        g = h5[group] if group else h5
        data, indices, indptr = g["data"][:], g["indices"][:], g["indptr"][:]
    return {(ids[r], ids[c]): w for r in range(len(ids))
            for c, w in zip(indices[indptr[r]:indptr[r + 1]], data[indptr[r]:indptr[r + 1]])}


def test_exact_backend_graph_is_order_invariant(tmp_path, be1_pcas_tsv):
    """With an exact search, permuting cells must leave the graph unchanged by
    barcode -- so in an exact arm, permutation_seed perturbs only the order the
    clusterer walks, not the graph. Weights are compared exactly: a float-level
    drift here would be a graph perturbation too."""
    base = run(tmp_path / "a", be1_pcas_tsv, 0, transformer="sklearn")
    for seed in (3, 11):
        perm = run(tmp_path / f"s{seed}", be1_pcas_tsv, seed, transformer="sklearn")
        for group in ("", "connectivities"):
            assert _by_barcode(perm, group) == _by_barcode(base, group), \
                f"exact graph ({group or 'distances'}) changed under permutation {seed}"
