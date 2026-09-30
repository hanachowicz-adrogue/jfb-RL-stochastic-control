---
title: Implicit Hamiltonian Control with Unknown Dynamics
theme: default
layout: default
class: text-left
highlighter: shiki
transition: none
mdc: true
canvasWidth: 980
aspectRatio: 16/9
fonts:
  sans: Arial
  provider: none
---

<h1 class="cover-title">Implicit Hamiltonian Control<br>with Unknown Dynamics</h1>

<p class="subtitle">Learning local derivatives to train an implicit feedback policy</p>

<div class="cover-authors">

**Marin Hanachowicz** · ETH Zurich, D-MATH

Research interview · MIT

</div>

<p class="source">Joint project with Ignacio E. Adrogué, Hector Stekelorom and François Barré.<br>My research focus: extending the JFB control pipeline to unknown dynamics.</p>

<!--
MAIN 1 | 0:30 | cumulative 0:30
My focus in this joint project was the RL extension. We start from an existing
implicit optimal-control method, then remove the assumption that the agent can
evaluate or differentiate the dynamics. I will focus on the method we implemented,
what the benchmarks show, and the research questions it leaves open.
Do not claim sole authorship of the complete joint implementation.
-->

---

# The target: unknown dynamics and an implicit policy

We observe transitions, while the running and terminal costs remain known.

$$
z_{k+1}=F(t_k,z_k,u_k),\qquad
J(\theta)=\mathbb E_{z_0\sim\rho}\!\left[\sum_{k=0}^{N-1}\Delta t\,L(t_k,z_k,u_{\theta,k})+G(z_N)\right]
$$

<table class="position-table">
  <thead><tr><th></th><th>Known dynamics</th><th>Unknown dynamics</th></tr></thead>
  <tbody>
    <tr><td>Explicit control law</td><td>Value-based optimal control</td><td>Data-driven control</td></tr>
    <tr><td>Implicit optimality condition</td><td>Implicit Hamiltonians + JFB<br>Gelphman et al.</td><td><strong>Our focus</strong><br>Local estimates + implicit JFB</td></tr>
  </tbody>
</table>

<p class="takeaway">Can we retain the implicit control structure using only transition data?</p>

<p class="source">Project report, §§2–3 and §5. This table describes the project's scope, without claiming a complete survey of prior work.</p>

<!--
MAIN 2 | 1:00 | cumulative 1:30
Define the setting carefully: full observed state, known differentiable costs,
unknown deterministic dynamics behind a simulator interface. We are not learning
from unknown rewards or partially observed data. The difficulty is the combination
of an implicit control law and unavailable dynamics derivatives. The finance
liquidation work reproduces/applies the original method; the unknown-dynamics
extension is our central contribution and the focus of this talk.
-->

---

# Starting point: an implicit control layer trained with JFB

The value network supplies a policy costate: $p_\theta(t,z)=\nabla_z\phi_\theta(t,z)$.

We use the minimization convention $H=L+p^\top f$ throughout.

$$
T_\theta(u;t,z)=\Pi_U\!\left[u-\alpha_{\mathrm{fp}}\nabla_u H(t,z,p_\theta,u)\right],
\qquad u_\theta=T_\theta(u_\theta;t,z)
$$

For an interior smooth fixed point, exact implicit differentiation gives

$$
\frac{\partial u_\theta}{\partial\theta}
=\left(I-\partial_uT_\theta\right)^{-1}\partial_\theta T_\theta
\quad\approx_{\mathrm{JFB}}\quad \partial_\theta T_\theta.
$$

<p class="takeaway">We inherit the value network, fixed-point solver and JFB approximation.</p>

<p class="source">Gelphman et al. (2025, 2026), Wu Fung et al. (2022), cited in the project report.<br>Projection handles control bounds. Smooth sensitivity formula applies away from active-set changes.</p>

<!--
MAIN 3 | 1:05 | cumulative 2:35
Spend little time on classical PMP/HJB. Explain the inherited architecture and
the one approximation in JFB: replace the inverse of I minus the operator's
control Jacobian by identity. Jacobian-free refers to differentiating this
implicit layer, not to having no dynamics Jacobians. Those dynamics Jacobians
are precisely what our RL extension has to estimate. Convergence and descent
require assumptions; do not claim that concavity alone guarantees convergence.
-->

---

# Our extension estimates the missing derivatives online

At each time step, fit an affine transition model by batched recursive least squares:

$$
\widehat F_k(z,u)=M_k\begin{bmatrix}z\\u\\1\end{bmatrix},
\qquad M_k=[\,A_k^{\mathrm{disc}}\ \ B_k^{\mathrm{disc}}\ \ c_k\,].
$$

Convert the fitted transition derivatives to the quantities used by the control layer:

$$
a_k=\frac{A_k^{\mathrm{disc}}-I}{\Delta t}\approx\partial_z f,
\qquad
b_k=\frac{(B_k^{\mathrm{disc}})^\top}{\Delta t}\approx(\partial_u f)^\top.
$$

<div class="two-col">
<div>

**Identification**

Noisy controls provide information about $b_k$. A forgetting factor limits the influence of old rollouts.

</div>
<div>

**Initialization**

$A_k^{\mathrm{disc}}=I$ gives $a_k=0$ before data arrive, avoiding an artificial adjoint collapse.

</div>
</div>

<p class="source">Project report §5.1. Stored dimensions: a is n × n; b is m × n.</p>

<!--
MAIN 4 | 1:20 | cumulative 3:55
RLS does not recover a global model of the nonlinear system. We fit a shared
affine estimate across the training batch, separately at each time step. It
should be interpreted as a local approximation on the visited distribution.
Make the discrete/continuous conversion explicit once. b is stored transposed
in the implementation so multiplying b by a state costate produces a control
gradient. The forgetting factor and exploration help, but do not guarantee
identifiability. Detailed update equations are in the appendix.
-->

---

# Estimated derivatives drive both control and training

The policy uses the learned costate $p_\theta$:

$$
\widehat T_{\theta,k}(u;z)=\Pi_U\!\left[u-\alpha_{\mathrm{fp}}
\left(\nabla_uL+b_kp_\theta(t_k,z)\right)\right].
$$

The training direction uses a separate backward costate $\widehat p$:

$$
\widehat p_N=\nabla G(z_N),\qquad
\widehat p_k=\widehat p_{k+1}+\Delta t\left(a_k^\top\widehat p_{k+1}+\nabla_z L_k\right).
$$

<div class="flow small">
<div><strong>1. Explore</strong>Noisy rollout updates RLS.</div>
<div><strong>2. Evaluate</strong>Clean rollout defines the cost.</div>
<div><strong>3. Propagate</strong>Estimated adjoint runs backward.</div>
<div><strong>4. Update</strong>Backpropagate the JFB surrogate through the value network.</div>
</div>

<p class="takeaway">No differentiation through the environment is needed.</p>

<!--
MAIN 5 | 1:30 | cumulative 5:25
Explain the two costates: p_theta generates the control, p_hat weights the
training surrogate. They need not agree during training. The exploration
trajectory is used for identification; the clean trajectory defines the policy
objective. State transitions, Jacobian estimates and backward costates are
detached. Autograd is used only for the theta-dependent operator in the
surrogate. When exploration is zero, the clean rollout also updates RLS.
These formulas display alphaL=alphaG=1; the implementation supports weights.
The backward recursion is the open-loop discrete adjoint used by the code.
Its relation to an exact closed-loop policy gradient needs care; see appendix.
-->

---

# A benchmark that requires an implicit control solve

**Van der Pol stabilization**: $z=(x_1,x_2)$, $T=3$, target $(0,0)$.

$$
\dot x_1=x_2,\qquad
\dot x_2=(1-x_1^2)x_2-x_1+g(x_1)u.
$$

$$
L=x_1^2+x_2^2+\ell(u),\qquad G_{\mathrm{eff}}=5(x_1^2+x_2^2).
$$

| Variant | Control penalty $\ell(u)$ | Control coupling |
|:--|:--|:--|
| Standard | $\tfrac12u^2$ | $g=1$; explicit control, sanity check |
| Hard | $\lambda_u(e^{u^2}-1)$ | $g=1$; nonlinear optimality condition |
| **Hard Gain** | $\lambda_u(e^{u^2}-1)$ | $g=1+\beta\tanh x_1$; state-dependent derivative |

<p class="takeaway">Hard Gain combines implicit optimality with a changing control Jacobian.</p>

<p class="source">Hard / Hard Gain: control bounds [−2, 2], penalty weight 0.05, terminal weight 5. Hard Gain coefficient: 0.5.</p>

<!--
MAIN 6 | 0:50 | cumulative 6:15
The standard problem has an explicit control solution, so it does not by itself
validate the implicit extension. The exponential penalty makes the condition
nonlinear in u; state-dependent gain then directly stresses estimation of b.
The dynamics are shown to explain the test, but are hidden from the RLS agent.
The report's blanket [-3,3] bound is corrected to the actual hard-run [-2,2].
-->

---

# Hard Gain: a 1.7% test-cost gap to BPTT in this run

<img src="/images/hard-gain-comparison.png" alt="Hard Gain Van der Pol: training cost, phase portraits, control bands and held-out test-cost distributions for RLS-JFB, Oracle-JFB and BPTT." class="figure figure-large" />

<p class="small">128 held-out initial conditions. Mean cost: <strong>RLS-JFB 5.718</strong>, Oracle-JFB 5.795, BPTT 5.624. Lower is better.</p>

<p class="source">Project report, Fig. 5. These are results from one displayed run, not a statistical equivalence claim.</p>

<!--
MAIN 7 | 1:30 | cumulative 7:45
Use the legend to distinguish the methods. RLS-JFB only sees transitions;
Oracle-JFB uses true dynamics Jacobians, isolating the estimation change;
BPTT trains an explicit MLP policy by differentiating through the known dynamics.
On the held-out batch, RLS costs 5.718 versus BPTT's 5.624: +1.67%, rounded
to 1.7%. Oracle costs 5.795. RLS is slightly lower in this run; this is not proof
that estimation helps or that RLS and oracle are equivalent. Training is
non-monotone, especially for oracle. Evaluation includes eight identification
warm-up rollouts; thus test-time environment interaction is part of the protocol.
Show the cost distribution and phase portrait, not every line in every panel.
-->

---

# The result is promising, but its scope is narrow

- **Test-time identification:** eight warm-up rollouts adapt estimates on the evaluation batch.
- **Single displayed run:** the 1.7% result does not establish robustness across seeds or horizons.
- **Different architectures:** BPTT uses a direct MLP policy, so its gap does not isolate JFB error.
- **Local estimates:** changing state distributions and weak exploration can degrade identification.
- **Approximate training direction:** JFB and Jacobian estimation introduce distinct errors.

The standard Van der Pol export has a larger RLS–BPTT gap: **13.7%**.

<p class="takeaway">Descent under simultaneous JFB and estimation error remains a conjecture.</p>

<p class="source">Project report §5.5 and §5.8; evaluation protocol checked against examples-RL/hard_gain_vdp.py and vanderpol_comparison.py.</p>

<!--
MAIN 8 | 1:15 | cumulative 9:00
The hard-gain run is a proof of feasibility, not a guarantee across problems.
Be transparent about evaluation warm-up: held-out initial conditions are not
used in training, but the estimator sees evaluation transitions before scoring.
Show the standard benchmark gap to avoid selecting only the most favorable
result. Horizon error propagation, observability/excitation and policy changes
are central limitations. The conjecture is in the report but not proved.
There is also an open question about the feedback-aware gradient, expanded in
the appendix. No RL wall-clock/memory advantage is established by these plots.
-->

---

# Descent with estimated Jacobians

Let $g$ be the true policy gradient and $\widehat g$ the implemented direction.

$$
\widehat g^\top g>0
$$

is the local condition for $-\widehat g$ to decrease the true objective.

**Proposed experiment**

Measure gradient alignment while varying the horizon, exploration and Jacobian error. Compare the current adjoint with a feedback-aware derivative.

**Research direction**

Relate admissible estimation error to fixed-point conditioning, then choose exploration from identification uncertainty.

<p class="takeaway">Transition-based learning can retain an implicit control structure.<br>The remaining challenge is to quantify when that structure is reliable.</p>

<!--
MAIN 9 | 1:00 | cumulative 10:00
Close with a concrete falsifiable question rather than a generic list of future
work. In an instrumented simulator we can compute the true gradient as a
diagnostic without supplying it to the learner. Sweep horizon, excitation and
estimation quality, then test whether angle/alignment predicts training failure.
Feedback dependence must be accounted for in the reference gradient. This is
proposed research, not an experiment already completed. Invite discussion here.
The next slide begins the optional appendix; stop the 10-minute talk here.
-->

---
src: ./appendix.md
---
