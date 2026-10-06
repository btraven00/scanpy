# Scanpy

Scanpy-backed PCA module for omnibenchmark scRNA pipelines.

## Setup

```sh
pixi install
pixi run check
```

`pixi run check` imports all runtime dependencies and prints `OK`. Run it after install to confirm the environment is healthy.

## Usage

### PCA

```sh
pixi run python pca.py \
  --output_dir <dir> \
  --name <name> \
  --normalized.h5 <normalized.h5> \
  --selected.genes <selected.genes.gz> \
  --solver <arpack|randomized> \
  --n_components <int> \
  --random_seed <int>
```

Output: `<output_dir>/<name>_<solver>_n_<n_components>.h5` — see [`docs/pca_output.md`](docs/pca_output.md) for the full format spec.

#### Validation


```sh
pixi run validate <output_dir>/<name>_pca.h5
```

Exit codes: `0` = valid, `1` = validation failure, `2` = IO / usage error.

## Conda environment export

```sh
pixi run export-env
```

Exports the resolved environment to `envs/scanpy.yml`. The environment is named after the repo root folder.

## Citation

If you use this module in your research, please cite it using the information in `CITATION.cff`.

## Solver controls and how they compare across tools

The scanpy, rapids-singlecell and seurat modules expose the same knobs, with
defaults equal to each module's historical behaviour (so existing results are
unchanged). Checked 2026-10-05 on tm-facs.

**Leiden iterations (`--n_iterations`, `cluster` entrypoint)**

| module | backend | default | meaning | until convergence |
|---|---|---|---|---|
| scanpy | igraph (`flavor igraph`) / leidenalg | 2 | iterations of the Leiden algorithm | **yes**: any negative value runs until an iteration no longer improves quality |
| rapids-singlecell | cuGraph `leiden(max_iter=)` | 100 | a *cap* on levels/iterations; stops early at convergence | effectively: a large cap (the default 100) |
| seurat | leidenbase `num_iter` (`FindClusters n.iter`) | 10 | runs *exactly* this many iterations, no early stop | **no**: values < 1 are rejected; use a large value (cost grows linearly: 2 = 4.3 s, 10 = 11.9 s, 50 = 68.5 s per resolution on tm-facs) |

The units are not identical (igraph iterations vs cuGraph levels vs leidenbase
iterations). For comparisons across tools use **7 in all three**. 2 is enough
for scanpy and Seurat (Seurat modularity 0.96261 at 2 vs 0.96293 at 10 vs
0.96292 at 50 on tm-facs) but NOT for rapids: cuGraph caps aggregation levels,
and at 2 it returned ~3,200 clusters at every resolution. rapids' cluster
counts at res 0.5 / 1.5 / 2.5 on a tm-facs SNN graph: 2 -> 3175/3183/3181,
5 -> 107/114/129, 7 -> 48/70/84, 10 and 100 -> 48/68/83.

**Randomized / iterative PCA (`pca` entrypoint)**

| module | solver | knobs | default |
|---|---|---|---|
| scanpy | `randomized` with `--dense true` (sklearn) | `--n_iter`, `--n_oversamples` | `auto` (= 7 power iterations here), 10 |
| rapids-singlecell | `randomized-halko` | `--n_iter`, `--n_oversamples` | 7, 10 (matches scanpy) |
| seurat | `approximate` (irlba, Krylov) | `--irlba_work`, `--irlba_maxit`, `--irlba_tol` | nv + 7, 1000, 1e-5 (runs to tolerance) |

scanpy and rapids are matched (7 / 10). irlba is a different algorithm run to
convergence; its knobs do not map onto power iterations / oversamples. Note:
rapids' halko PCA is not bit-reproducible between identical runs (GPU; max
|diff| ~2e-3 on tm-facs, the size of scanpy's whole seed effect).

