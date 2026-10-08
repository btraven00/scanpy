#!/usr/bin/env python3
"""Run one or more typed steps (src/steps.py) in a single process (prototype).

  --steps PCA,NNG,CLUST   fused: intermediates pass in memory, no read-back
  --steps PCA             split: exactly one stage, load -> run -> save

Only the external inputs of the chain are loaded from files. Every output of
every step is saved, because downstream metrics need them. All saves happen
after the last compute phase, so they fall outside the timed steps. Each step
gets an obkit phase named after its stage.
"""

import argparse
import dataclasses
import gc
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import cli  # noqa: E402
from phases import phase  # noqa: E402
from obkit.logger import emit, init_logger  # noqa: E402
from steps import STEPS, plan  # noqa: E402

import anndata as ad  # noqa: E402


def _head(v, n):
    """The first n cells of a loaded input, for the warm-up run."""
    if isinstance(v, ad.AnnData):
        return v[:n].copy()
    def cut(x):
        if isinstance(x, list):
            return x[:n]
        if hasattr(x, "shape") and len(x.shape) == 2 and x.shape[0] == x.shape[1]:
            return x[:n, :n]  # cell x cell graph
        return x[:n] if hasattr(x, "shape") else x
    return dataclasses.replace(v, **{f.name: cut(getattr(v, f.name)) for f in dataclasses.fields(v)})


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--steps", required=True)
    known, _ = pre.parse_known_args(argv)
    stages = known.steps.split(",")
    external = plan(stages)

    p = argparse.ArgumentParser(description=f"fused scanpy steps: {known.steps}")
    cli.add_base_args(p)
    p.add_argument("--steps", required=True)
    p.add_argument("--replicate", type=int, default=0)  # unused; separates same-seed replicate dirs
    # In-process replicates after one warm-up: replicate r writes to rep<r>/ (and r=0 also to
    # the declared outputs), with every *_random_seed + r * seed_stride (0: same-seed replicates).
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--seed_stride", type=int, default=0)
    # Run-wide, not a step parameter: fuse-prof.sh pins N cores and sizes every pool to N.
    p.add_argument("--threads", type=int, default=0)
    # Run-wide: run the chain once on the first N cells under warmup:* phases and discard
    # it, so JIT/kernel compilation and device init land there, not in the timed phases.
    # N must be large enough to take the same code paths (scanpy kNN: >= 4096 cells).
    p.add_argument("--warmup_cells", type=int, default=0)
    for k in external:
        p.add_argument(f"--{k}", type=Path, required=True)
    # step parameters are namespaced by stage: --pca_dtype, --nng_n_neighbors, --clust_random_seed
    for st in (STEPS[s] for s in stages):
        for k, t in st.params.items():
            p.add_argument(f"--{st.stage.lower()}_{k}", type=t, required=True)
    return p.parse_args(argv), stages, external


def _chain(env, stages, a, prefix=""):
    produced = []
    for st in (STEPS[s] for s in stages):
        with phase(prefix + st.stage.lower()):
            res = st.run({k: env[k] for k in st.inputs}, {k: a[f"{st.stage.lower()}_{k}"] for k in st.params})
        env.update(res)
        produced += [(k, st.outputs[k]) for k in res]
    return produced


def _rss_mb():
    """Current (not peak) RSS, for the creep check across replicates. Linux only."""
    try:
        return round(int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)
    except OSError:
        return None


def main(argv=None):
    args, stages, external = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    init_logger(str(out))
    a = vars(args)
    state = {"replicate": None, "done": 0}

    # Why we quit, in the event log: ok / sigterm (the runner's time limit) / error. Python runs
    # the handler between bytecodes, so a long C call delays it; the runner's own record of
    # SIGTERM / kill / OOM is authoritative, this says where the module was.
    def on_term(*_):
        emit("exit", "end", attrs=dict(state, reason="sigterm"))
        sys.exit(128 + signal.SIGTERM)
    signal.signal(signal.SIGTERM, on_term)
    try:
        with phase("load"):
            env = {k: t.load(a[k]) for k, t in external.items()}
        if args.warmup_cells:
            _chain({k: _head(v, args.warmup_cells) for k, v in env.items()}, stages, a, "warmup:")
        for r in range(args.replicates):
            state["replicate"] = r
            ar = {k: v + r * args.seed_stride if k.endswith("_random_seed") else v for k, v in a.items()}
            if r:  # fresh inputs and a collected heap, so a replicate can't lean on the last one's state
                env = None
                gc.collect()
                with phase("load"):
                    env = {k: t.load(a[k]) for k, t in external.items()}
            with phase("replicate") as attrs:
                attrs.update(replicate=r, seeds={k: v for k, v in ar.items() if k.endswith("_random_seed")})
                saves = _chain(env, stages, ar)
                attrs["rss_mb"] = _rss_mb()
            with phase("write"):  # after each replicate, so a SIGTERM keeps the finished ones
                for d in [out / f"rep{r}"] + ([out] if r == 0 else []):
                    d.mkdir(exist_ok=True)
                    for k, t in saves:
                        t.save(env[k], d / f"{args.name}{t.suffix}")
            state["done"] = r + 1
    except Exception as e:
        emit("exit", "end", attrs=dict(state, reason="error", error=f"{type(e).__name__}: {e}"))
        raise
    emit("exit", "end", attrs=dict(state, reason="ok"))


if __name__ == "__main__":
    main()
