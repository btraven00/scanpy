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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import cli  # noqa: E402
from phases import phase  # noqa: E402
from obkit.logger import init_logger  # noqa: E402
from steps import STEPS, plan  # noqa: E402


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
    for k in external:
        p.add_argument(f"--{k}", type=Path, required=True)
    seen = {}
    for st in (STEPS[s] for s in stages):
        for k, t in st.params.items():
            assert seen.setdefault(k, t) is t, f"param {k} has two types"
    for k, t in seen.items():
        p.add_argument(f"--{k}", type=t, required=k != "blas_threads", default=0 if k == "blas_threads" else None)
    return p.parse_args(argv), stages, external


def main(argv=None):
    args, stages, external = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    init_logger(str(out))
    a = vars(args)

    with phase("load"):
        env = {k: t.load(a[k]) for k, t in external.items()}
    saves = []
    for st in (STEPS[s] for s in stages):
        with phase(st.stage.lower()):
            res = st.run({k: env[k] for k in st.inputs}, {k: a[k] for k in st.params})
        env.update(res)
        saves += [(k, st.outputs[k]) for k in res]
    with phase("write"):
        for k, t in saves:
            t.save(env[k], out / f"{args.name}{t.suffix}")


if __name__ == "__main__":
    main()
