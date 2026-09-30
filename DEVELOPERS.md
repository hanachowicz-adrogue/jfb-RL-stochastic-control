# Architecture and development guide

This guide describes the current implementation, including the unknown-dynamics
extension. Installation is covered in the [root README](README.md); concrete
commands and evaluation protocols are in the [experiment guide](docs/EXPERIMENTS.md).
All Python paths below are relative to `jfb-for-implicit-oc/`.

## Package organization

The repository uses script entry points and explicit import-path bootstrapping;
it is not currently an installable Python distribution. Run examples from the
Python project directory rather than assuming `pip install -e .` is supported.

| Directory | Responsibility |
| --- | --- |
| `core/` | Known-dynamics optimal-control contracts, implicit policies, trainers and artifact management |
| `core_RL/` | Transition-only environments, Jacobian identification, implicit RL policies and estimated-adjoint training |
| `models/` | Problem-specific dynamics, costs, initial conditions and plotting adapters |
| `examples/` | Known-dynamics training entry points and a model template |
| `examples/explicit_ustar/` | Analytical/BVP comparison utilities for liquidation and bicycle problems |
| `examples-RL/` | RL training, oracle comparisons, evaluation and visualization scripts |
| `benchmarking/` | Trajectories, reference policies/solvers, diagnostics, metrics, plots and tests |
| `results/` | Generated artifacts, ignored by Git and created at runtime |

The two policy-training paths share the implicit-Hamiltonian idea but have
different information contracts. The benchmark simulator may know the dynamics;
the unknown-dynamics learner must not use their analytical derivatives.

## Known-dynamics path

| Module | Responsibility |
| --- | --- |
| `core/ImplicitOC.py` | Abstract problem, dynamics/cost derivatives and objective construction |
| `core/ImplicitNets.py` | Value network and fixed-point implicit control layer; JFB/FullAD modes |
| `core/DirectControlNets.py` | Direct-control neural baseline |
| `core/CVXPolicy.py` | Convex optimization policy support |
| `core/OptimalControlTrainer.py` | Optimization, best-policy handling, training history and plotting dispatch |
| `core/run_io.py` | Run identifiers and artifact naming |
| `core/paths.py` | Output directory layout |
| `core/log_format.py` | Log formatting |
| `core/utils.py` | Numerical/network utilities |

Concrete models include consumption, quadcopter, multiple bicycles and portfolio
liquidation. `LiquidationPortfolio.py` also includes the reduced formulation and
stochastic rollout machinery. Analytical reference plots are a separate workflow
from ordinary trained-policy rollouts.

## Unknown-dynamics path

| Module | Responsibility |
| --- | --- |
| `core_RL/Environment.py` | `Environment.step` and rollout helpers; `AnalyticalEnvironment` hides simulator dynamics behind detached transitions |
| `core_RL/JacobianEstimator.py` | Jacobian-estimator interface and recursive least squares identification |
| `core_RL/ImplicitOC_RL.py` | Known costs, estimated Hamiltonian gradient, backward adjoint and training surrogate |
| `core_RL/ImplicitNets_RL.py` | Implicit policy that receives the current control-Jacobian estimate |
| `core_RL/OptimalControlTrainer_RL.py` | RL trainer built on the shared training infrastructure |

Problem definitions live in `VanDerPolOC_RL.py`, `Hard_VDP_RL.py`,
`Hard_Gain_VDP_RL.py` and `PortfolioOC_RL.py`. Standalone comparison runners
implement some of their own training and output logic rather than delegating
everything to `OptimalControlTrainer_RL`.

### Information and tensor contracts

For batch size `B`, state dimension `n`, control dimension `m` and `N` steps:

| Quantity | Shape / convention |
| --- | --- |
| State passed to the environment | `(B, n)` |
| Control passed to the environment | `(B, m)` |
| `Environment.step(z, u, t)` result | `(B, n)`, detached from autograd |
| State trajectory | `(B, n, N+1)` |
| Control trajectory | `(B, m, N)` |
| Estimated state derivative `a_k` | `(B, n, n)` when supplied to the batch computation |
| Estimated control derivative `b_k` | `(B, m, n)`, storing the transpose of the dynamics control Jacobian |
| Network costate and backward adjoint | `(B, n)`, with different roles |

RLS fits a discrete affine transition model. For timestep `dt`, the code uses
`a = (A_disc - I) / dt` and `b = B_disc.T / dt`. The minimization Hamiltonian is
`H = L + p.T f`; its estimated control derivative is `grad_u L + b p`.
Do not transpose `b` again when forming that product. Current RLS estimates are
time-indexed and shared across samples, not independently fitted for each state.

The training flow is:

1. Optionally collect perturbed controls and transitions for identification.
2. Roll out the current policy without exploration noise to evaluate its cost.
3. Form a backward adjoint using the estimated derivatives and known cost derivatives.
4. Construct the JFB surrogate and update the neural parameters.

The neural costate constructs the control; the backward adjoint weights the
training surrogate. These are not interchangeable. An estimated-adjoint
surrogate is not automatically the exact gradient of a feedback policy.
See the [scientific notes](presentation_m/README.md#conventions-et-précisions-scientifiques)
for the limitations reflected in the interview deck.

## Benchmarking and visualization

| Module | Role |
| --- | --- |
| `benchmarking/trajectory.py` | Common trajectory representation |
| `benchmarking/policies.py` | Reference policy interfaces |
| `benchmarking/solvers.py` | Numerical/reference solvers |
| `benchmarking/metrics.py` | Comparison metrics |
| `benchmarking/gradient_checks.py` | Derivative checks |
| `benchmarking/diagnostics.py` | Numerical and stochastic diagnostics |
| `benchmarking/plotter.py` | Panel definitions and comparison plotting |
| `benchmarking/paths.py` | Benchmark output paths |
| `benchmarking/tests/` | Trajectory, plotting, solver and stochastic-extension tests |

Models implementing `panels()` and `to_trajectory()` use the common plotting
service. Some models, including quadcopter and multiple bicycles, retain a legacy
`plot_position_trajectories(..., save_path=...)` fallback. The RL visualization
scripts also contain specialized plots and animations.

## Artifacts and run identity

The shared trainer delegates filenames to `RunIO` and directories to `core/paths.py`.
Its conventional output organization is:

```text
results/<ProblemClassName>/
├── training/
│   ├── best_policy_<tag>_<run_id>.pth
│   ├── history_<tag>_<run_id>.csv
│   ├── loss_curve_<tag>_<run_id>.png
│   └── training-plots/
└── rollouts/
    ├── policy_rollout_<tag>_<run_id>.png
    └── trajectory_<tag>_<run_id>.pth
```

Artifacts depend on the runner, plotting support and options; this is not a
promise that every script creates an identical number of files. Comparison and
reference workflows may add `benchmark/` and `reference/` directories. Some
standalone RL runners use fixed figure names, so repeated executions may replace
previous plots. Archive outputs separately when comparing seeds or settings.

## Extending a known-dynamics problem

1. Add a subclass of `ImplicitOC` in `models/`. Implement the running/terminal
   costs, their required derivatives, dynamics and their derivatives, initial
   condition sampler and trajectory generation contract.
2. Prefer `panels()` and `to_trajectory()` for plotting so the trainer can use
   the common plotting service.
3. Copy `examples/example_TEMPLATE.py`, resolve its marked import and problem
   constructor placeholders, then choose the network, bounds, optimizer and
   training settings appropriate to the problem.
4. Start with a short run, inspect loss and feasibility, then add a reference
   comparison if an analytical or numerical solution is available.

The liquidation comparison examples are in `examples/explicit_ustar/`, notably
`liquidation_benchmark.py` and `plot_liquidation_jfb.py`; they are not at the
Python project root. Prefer the shared artifact helpers for new runners.

## Extending an unknown-dynamics problem

1. Implement an `Environment` with detached batched `step` outputs, matching
   dimensions, device and time grid. Do not expose dynamics derivatives to the learner.
2. Implement an `ImplicitOC_RL` subclass for the known costs, cost derivatives
   and initial-state distribution. Override the state cost derivative where needed.
3. Wire a Jacobian estimator and an `ImplicitNetOC_RL` policy; provide the
   appropriate estimate at each timestep, including during evaluation.
4. Specify exploration amplitude, forgetting/regularization settings and warmup
   behavior. Ensure identification data is informative before interpreting results.
5. Compare to oracle Jacobians and a direct-policy baseline where available,
   documenting architecture differences, seeds and any test-time adaptation.

Use the Van der Pol comparison runners as concrete references. Their BPTT
baseline is a direct MLP, not the same implicit policy differentiated exactly.
Do not attribute its entire performance difference to JFB approximation error.

## Validation and reproducibility

Tests currently cover a subset of benchmarking and stochastic liquidation
behavior; they do not certify the full RL method or its theoretical guarantees.
The [experiment guide](docs/EXPERIMENTS.md#existing-tests) contains the required import
paths and test commands. For a new model, also check derivative dimensions,
control bounds, trajectory costs and a short end-to-end training run.

Record configuration, seed, device, software versions, checkpoint selection,
evaluation initial conditions and estimator adaptation. Exported figures in the
presentation are historical results, not automatically regenerated fixtures.
Keep run artifacts outside Git; commit source and deliberately selected figures.
