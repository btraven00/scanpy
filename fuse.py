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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import cli  # noqa: E402
from phases import phase  # noqa: E402
from obkit.logger import init_logger  # noqa: E402
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


def main(argv=None):
    args, stages, external = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    init_logger(str(out))
    a = vars(args)

    with phase("load"):
        env = {k: t.load(a[k]) for k, t in external.items()}
    if args.warmup_cells:
        _chain({k: _head(v, args.warmup_cells) for k, v in env.items()}, stages, a, "warmup:")
    saves = _chain(env, stages, a)
    with phase("write"):
        for k, t in saves:
            t.save(env[k], out / f"{args.name}{t.suffix}")


if __name__ == "__main__":
    main()
