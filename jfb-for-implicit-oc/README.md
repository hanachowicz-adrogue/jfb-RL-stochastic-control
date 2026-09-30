# Python research implementation

Implicit Hamiltonian optimal control with Jacobian-Free Backpropagation,
including the project's extension to unknown dynamics using recursive least
squares and estimated adjoints.

## Documentation

| Guide | Contents |
| --- | --- |
| [Project overview](../README.md) | Research context, contributions, results, repository layout and installation |
| [Experiment guide](../docs/EXPERIMENTS.md) | Runner commands, configurations, evaluation protocols and tests |
| [Architecture](../DEVELOPERS.md) | Module responsibilities, information contracts and extension recipes |
| [Upstream attribution](../ORIGINAL-AUTHORS.md) | Original framework, examples and citations |
| [Project report](../ML_for_finance_complex_systems.pdf) | Mathematical development and reported experiments |

## Entry points

Install the root `requirements.txt` first, then run from this directory.

```bash
# Unknown-dynamics benchmarks: RLS, oracle Jacobians and direct-policy BPTT
python examples-RL/vanderpol_comparison.py
python examples-RL/hard_vdp_comparison.py
python examples-RL/hard_gain_vdp.py

# Portfolio RL: train before checkpoint-based evaluation
python examples-RL/portfolio_optimization_RL.py
python examples-RL/evaluate_portfolio_rl.py

# Known-dynamics liquidation and reference comparison
python examples/example_liquidationportfolio.py
python examples/explicit_ustar/plot_liquidation_jfb.py --help
```

These are experiment commands, not quick unit tests. See the experiment guide
before running them for training budgets and evaluation warmup behavior.
The original consumption, bicycle and quadcopter examples remain in `examples/`.

## Source directories

- `core/`: known-dynamics problem contracts, networks, trainers and artifact helpers.
- `core_RL/`: transition-only environments, identification and implicit RL training.
- `models/`: concrete costs, simulators and problem definitions.
- `examples/`, `examples-RL/`: runnable experiments and visualization tools.
- `benchmarking/`: reference trajectories, metrics, diagnostics, solvers and tests.
- `results/`: runtime outputs, created as needed and ignored by Git.

This is a script-oriented research project, not an installable Python package.
No checkpoints or complete training histories are distributed with the source.
