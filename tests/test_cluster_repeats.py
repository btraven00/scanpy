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
