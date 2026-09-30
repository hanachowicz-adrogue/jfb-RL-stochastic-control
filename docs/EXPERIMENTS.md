# Experiments and reproducibility

This guide records how the existing scripts are organized and how to run them.
It does not change their configurations. Commands assume the repository root
and an activated Python environment installed as described in the
[main README](../README.md#installation).

## RL comparisons

| Benchmark | Command | Current script configuration |
| --- | --- | --- |
| Standard Van der Pol | `python jfb-for-implicit-oc/examples-RL/vanderpol_comparison.py` | 300 epochs, batch 64, seed 42 |
| Hard Van der Pol | `python jfb-for-implicit-oc/examples-RL/hard_vdp_comparison.py` | 300 epochs, batch 64, seed 1 |
| Hard Gain Van der Pol | `python jfb-for-implicit-oc/examples-RL/hard_gain_vdp.py` | 200 epochs, batch 32, seed 1 |

All three comparison scripts use Adam with learning rate `3e-3`. These scripts
contain experiment configurations in Python rather than a CLI configuration
file. Inspect the configuration block before treating a command as reproduction
of a particular archived figure.

The methods are:

1. An implicit policy trained with RLS-estimated dynamics derivatives.
2. The implicit-policy variant supplied with oracle dynamics Jacobians.
3. A direct MLP control policy trained by BPTT through differentiable Euler steps.

The first two use the inherited value-network/implicit-control architecture.
The BPTT policy has a different architecture, so its cost gap is a method-level
comparison rather than an isolated estimate of JFB gradient error.

### Evaluation protocol

Fresh initial conditions are sampled for scoring. RLS is initialized again and
adapted using eight warm-up rollouts on the evaluation batch. This test-time
interaction is part of the comparison protocol, not zero-interaction evaluation.
Standard Van der Pol scores 256 initial conditions; the two hard variants score
128. The hard variants bound controls to `[-2,2]` and use terminal weight 5.

The figures already committed with the project report and presentations are
exports from particular runs. Checkpoints and raw histories are not bundled.
No multi-seed statistical equivalence is established by a single export.

## Portfolio allocation

```bash
python jfb-for-implicit-oc/examples-RL/portfolio_optimization_RL.py
python jfb-for-implicit-oc/examples-RL/evaluate_portfolio_rl.py
```

The training script runs 200 epochs and writes a checkpoint under
`results/PortfolioOC_RL/training/`. Evaluation locates the latest matching
`best_policy_JFB-RL_RLS_*.pth`; it fails if no checkpoint exists. It rebuilds the
problem and policy from its configuration and warms up a new estimator before
scoring 512 fresh initial conditions.

Keep training and evaluation parameters consistent. The scalar wealth benchmark
uses deterministic drift and a nonlinear allocation penalty; it is a financial
sanity check rather than a demonstration of learning stochastic asset returns.

## Liquidation application

The configurable runner is
`jfb-for-implicit-oc/examples/explicit_ustar/plot_liquidation_jfb.py`.

```bash
python jfb-for-implicit-oc/examples/explicit_ustar/plot_liquidation_jfb.py --help
```

### Short deterministic run

This is a small workflow check, not a converged reproduction of a report figure:

```bash
python jfb-for-implicit-oc/examples/explicit_ustar/plot_liquidation_jfb.py --n-assets 1 --gamma 2 --train-epochs 3 --batch-size 8 --nt 20 --phi-arch default --no-aa --fp-alpha auto --learned-costate-overlay off --no-diagnostics --tag SMOKE
```

### Reference 20-asset stochastic configuration

This retains the previous README's documented run. It trains JFB and Full-AD for
300 epochs each, then evaluates Monte-Carlo rollouts. It is a substantial
computation, particularly on CPU. Run from the repository root on Linux/macOS:

```bash
python jfb-for-implicit-oc/examples/explicit_ustar/plot_liquidation_jfb.py \
  --n-assets 20 --t-final 2.0 --nt 100 \
  --kappa 1e-5 --sigma 0.01414 --eta 0.5 \
  --sigma-S 0.02 --n-hutch 4 --n-paths 256 --noise-seed 42 \
  --gamma 1.9 --epsilon 1e-2 --alpha 30.0 \
  --q0-min 3 --q0-max 8 --S0 2.0 --X0 0.0 \
  --train-epochs 300 --batch-size 256 --lr 1e-3 \
  --fp-max-iters 200 --fp-tol 1e-6 \
  --phi-arch default --seed 42 \
  --tag FULL-STOCHASTIC-VS-DET-20-ASSETS \
  --no-aa --fp-alpha 0.9 --learned-costate-overlay off \
  --full-ad
```

The equivalent PowerShell command uses a single line:

```powershell
python jfb-for-implicit-oc/examples/explicit_ustar/plot_liquidation_jfb.py --n-assets 20 --t-final 2.0 --nt 100 --kappa 1e-5 --sigma 0.01414 --eta 0.5 --sigma-S 0.02 --n-hutch 4 --n-paths 256 --noise-seed 42 --gamma 1.9 --epsilon 1e-2 --alpha 30.0 --q0-min 3 --q0-max 8 --S0 2.0 --X0 0.0 --train-epochs 300 --batch-size 256 --lr 1e-3 --fp-max-iters 200 --fp-tol 1e-6 --phi-arch default --seed 42 --tag FULL-STOCHASTIC-VS-DET-20-ASSETS --no-aa --fp-alpha 0.9 --learned-costate-overlay off --full-ad
```

Add `--device cuda` when a compatible GPU environment is available.

| Flag | Meaning |
| --- | --- |
| `--sigma` | Inventory-risk coefficients in the running cost |
| `--sigma-S` | Price diffusion coefficients; zero selects deterministic dynamics |
| `--n-paths` | Monte-Carlo paths used for plotted rollouts |
| `--n-hutch` | Probes for stochastic HJB trace computation where invoked |
| `--full-ad` | Train a second implicit policy with all inner iterations tracked |
| `--phi-arch default` | Use the generic value surrogate rather than the runner's anchored default |
| `--no-aa` | Disable Anderson acceleration in the inner solve |
| `--seed`, `--noise-seed` | Parameter/initial-condition RNG and price-noise RNG controls |

The default HJB and adjoint penalty weights are zero. Passing `--n-hutch`
does not by itself enable optimization of a stochastic HJB residual. A
deterministic reference is not automatically a proof of stochastic optimality.
The report's parameter tables and this CLI run are different configurations.

## Original benchmark examples

From `jfb-for-implicit-oc/`:

```bash
python examples/example_multibicycle.py
python examples/example_multi_quadcopter.py
python examples/example_multiConsumption.py
```

These preserve applications from the upstream framework. Additional usage and
attribution are in [ORIGINAL-AUTHORS.md](../ORIGINAL-AUTHORS.md).

## Generated outputs

Trainer-based runs use the following layout, created on demand:

```text
jfb-for-implicit-oc/results/<ProblemClassName>/
├── training/
│   ├── best_policy_<tag>_<run_id>.pth
│   ├── history_<tag>_<run_id>.csv
│   ├── loss_curve_<tag>_<run_id>.png
│   └── training-plots/rollout_<tag>_<run_id>_<epoch>.png
├── rollouts/
│   ├── policy_rollout_<tag>_<run_id>.png
│   └── trajectory_<tag>_<run_id>.pth
├── reference/                       Analytical or numerical reference artifacts
└── benchmark/                       Comparisons and diagnostics
```

Some comparison scripts manage their own checkpoint folders and filenames inside
the problem's results directory. Portfolio evaluation uses `evaluation/`.
Inspect the printed output paths rather than assuming every runner writes the
same artifact bundle. Timestamped trainer artifacts separate runs; some fixed
benchmark export filenames can be overwritten by rerunning the same script.

For the 20-asset reference command, relevant benchmark figures include
`best_policy_same_axes_FULL-STOCHASTIC-VS-DET-20-ASSETS.png` and
`training_curves_jfb_vs_ad_FULL-STOCHASTIC-VS-DET-20-ASSETS.png`.
All runtime results are ignored by Git.

## Existing tests

Install `pytest` and `scipy` in addition to the root requirements. The tests use
both package imports and legacy flat imports; include the code root, `core/`,
`models/` and reference-runner directory on `PYTHONPATH`.

From the repository root on Linux/macOS:

```bash
cd jfb-for-implicit-oc
PYTHONPATH=.:core:models:examples/explicit_ustar python -m pytest benchmarking/tests -q
```

From the repository root in PowerShell:

```powershell
cd jfb-for-implicit-oc
$env:PYTHONPATH = "$PWD;$PWD/core;$PWD/models;$PWD/examples/explicit_ustar"
python -m pytest benchmarking/tests -q
```

| Test module | Coverage |
| --- | --- |
| `test_trajectory.py` | Deterministic and stochastic trajectory shapes and summaries |
| `test_plotter.py` | Plotting behavior for the shared trajectory representation |
| `test_closed_form_solver.py` | Analytical liquidation reference against BVP and boundary conditions |
| `test_stochastic_extension.py` | Diffusion contract, trace estimator, deterministic limit and multi-path rollout |

These tests do not establish descent of the estimated RL direction or
statistical equivalence of learned policies.

## Recording a reproducible run

Retain the exact script revision, configuration, seeds, library versions and
hardware. Save checkpoints, histories, test initial-condition generation and
identification warm-up settings alongside figures. Report whether the comparison
uses the same architecture, the same gradient construction, or only the same
objective. Keep report exports distinct from newly generated experiments.
