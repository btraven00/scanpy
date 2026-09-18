"""Seed derivation for repeated clustering.

The failure this guards against is silent: correlated seeds still produce a
number, just a number that overstates agreement. Nothing downstream can detect
it, so it gets pinned here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".."))
from cluster_repeats import spawn_seeds  # noqa: E402


def test_deterministic_from_the_base():
    """A repeat must be reproducible from its plan line."""
    assert spawn_seeds(42, 50) == spawn_seeds(42, 50)


def test_no_overlap_between_base_seeds():
    """The whole reason this is not `base + i`, which would share 92 of 100."""
    assert set(spawn_seeds(42, 100)).isdisjoint(spawn_seeds(50, 100))


def test_distinct_within_a_run():
    seeds = spawn_seeds(42, 100)
    assert len(set(seeds)) == 100


def test_prefix_is_stable_under_more_repeats():
    """repeats=100 must extend repeats=10, not reshuffle it -- otherwise
    raising repeats invalidates every cached run instead of extending it."""
    assert spawn_seeds(42, 100)[:10] == spawn_seeds(42, 10)


def test_fits_int32_for_igraph():
    assert all(0 <= s < 2**31 - 1 for s in spawn_seeds(42, 100))


def test_call_into_cluster_leiden_stays_valid():
    """cluster_repeats reuses cluster.py's clusterer, so it is coupled to that
    signature. It drifted once already: this module was written against a
    3-argument cluster_leiden and the branch had grown to 5 (flavor,
    partition_type), which argparse cannot catch -- it failed only in the run.
    """
    import ast
    import inspect

    import cluster

    src = (Path(__file__).parent / ".." / "cluster_repeats.py").read_text()
    call = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "cluster_leiden")
    # Bind the real signature with that many positional args; raises on drift.
    sig = inspect.signature(cluster.cluster_leiden)
    sig.bind(*range(len(call.args)))


def test_repeats_exposes_every_method_param_cluster_does():
    """A repeat must be the same clustering, so the knobs must match."""
    import re

    def flags(path):
        return set(re.findall(r'add_argument\(\s*"(--[a-z_]+)"', Path(path).read_text()))

    here = Path(__file__).parent / ".."
    missing = flags(here / "cluster.py") - flags(here / "cluster_repeats.py")
    assert not missing, f"cluster_repeats is missing {missing}"
