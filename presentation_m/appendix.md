# Appendix: discussion material

| Appendix | Topic |
|:--|:--|
| A2–A3 | Surrogate and feedback-gradient caveat |
| A4–A5 | RLS update and identification |
| A6–A8 | Baselines, standard and hard Van der Pol |
| A9–A10 | Portfolio problem and analytical reference |
| A11–A12 | Contractivity and value-network architecture |
| A13–A14 | Liquidation application and stochastic scope |
| A15 | References |

<!--
Appendices have no assigned talk time. They support the 20-minute discussion.
-->

---

# A2. The implemented JFB surrogate

Fix the clean trajectory $(\bar z_k,\bar u_k)$, estimates and backward costates.

$$
w_k=\Delta t\left(\nabla_u L(t_k,\bar z_k,\bar u_k)+b_k\widehat p_{k+1}\right).
$$

$$
S(\theta)=\sum_k\left\langle\widehat T_{\theta,k}(\bar u_k;\bar z_k),\operatorname{stopgrad}(w_k)\right\rangle,
\qquad \widehat g=\nabla_\theta S.
$$

Only the operator depends on $\theta$ through $p_\theta=\nabla_z\phi_\theta$.

| Quantity | Autograd in this pass? |
|:--|:--|
| States, controls, Jacobian estimates | No |
| Backward costates and weights $w_k$ | No |
| Value-network parameters inside $\widehat T$ | Yes |

<p class="source">Project report §5.3. Unit cost weights shown. Sign and transpose conventions follow core_RL/ImplicitOC_RL.py.</p>

---

# A3. Open-loop adjoint versus feedback-aware differentiation

The implemented backward pass uses $(\partial_z F_k)^\top\widehat p_{k+1}$ and treats the sampled controls as independent inputs.

For a feedback policy $u_k=\pi_\theta(t_k,z_k)$, the exact cost-to-go sensitivity $\lambda_k$ includes policy-state dependence:

$$
\lambda_k=\Delta t\,L_z+F_z^\top\lambda_{k+1}
+\pi_z^\top\!\left(\Delta t\,L_u+F_u^\top\lambda_{k+1}\right).
$$

$$
g=\sum_k\pi_\theta^\top\!\left(\Delta t\,L_u+F_u^\top\lambda_{k+1}\right),
\qquad \lambda_N=G_z.
$$

Here $\pi_\theta$ denotes a parameter partial derivative at fixed state.

<p class="takeaway">Away from stationarity, the open-loop adjoint need not give the exact feedback-policy gradient.</p>

<p class="source">Direct chain-rule derivation for the discretized feedback objective. The implemented surrogate is treated as an approximate direction, not an established exact gradient.</p>

---

# A4. Batched recursive least squares

At time $k$, let $X=[z\;u\;1]\in\mathbb R^{B\times d}$ and $Y=z_{k+1}\in\mathbb R^{B\times n}$, where $d=n+m+1$.

$$
Q_k^{\mathrm{new}}=\lambda Q_k^{\mathrm{old}}+(1-\lambda)q_0I+X^\top X,
$$

$$
M_k^{\mathrm{new}}=M_k^{\mathrm{old}}
+\left(Y-X(M_k^{\mathrm{old}})^\top\right)^\top X\left(Q_k^{\mathrm{new}}\right)^{-1}.
$$

The implementation uses a linear solve rather than explicitly forming the inverse.

- One affine model per time step, shared across the batch.
- Initialize $M_k=[I\;0\;0]$ and $Q_k=q_0I$.
- $\lambda$ controls forgetting; $q_0$ regularizes the regression.

<p class="source">Project report §5.1; core_RL/JacobianEstimator.py. This equation uses output-row matrices, consistently with the code.</p>

---

# A5. Identification and exploration

Exploration controls use $u_k^{\mathrm{exp}}=\Pi_U(u_{\theta,k}+\sigma\varepsilon_k)$ with $\varepsilon_k\sim\mathcal N(0,I)$.

**Why separate the two rollouts?**

The noisy rollout excites the system. The clean rollout evaluates the current policy without adding exploration cost to its objective.

**Diagnostics to examine**

- Regression precision and excitation in the control directions.
- Prediction residual on fresh transitions, not only the fitted batch.
- Estimated versus true Jacobian error in an instrumented benchmark.
- Fixed-point residual and sensitivity to horizon length.

<p class="takeaway">Small transition residuals alone do not identify the control derivative.</p>

<p class="source">Implemented two-rollout design: report §5.4 and core_RL/ImplicitOC_RL.py. The additional diagnostics are proposed validation work.</p>

---

# A6. Baselines and evaluation protocol

| Method | Access to derivatives | Training direction |
|:--|:--|:--|
| RLS-JFB | Estimated from transitions | Estimated JFB surrogate |
| Oracle-JFB | True dynamics Jacobians | Same JFB surrogate structure |
| Autodiff / BPTT | Differentiable known dynamics | Direct MLP policy trained through Euler rollout |

Hard Gain: 200 epochs, batch 32, Adam learning rate $3\times10^{-3}$, seed 1, $T=3$, 60 steps, terminal weight 5.

Test: 128 fresh initial conditions and **eight identification warm-up rollouts** for JFB evaluation before scoring.

<p class="small">RLS / oracle use the same implicit architecture. BPTT uses a two-hidden-layer MLP of width 64. Its cost gap is not an isolated JFB approximation error. Optimizer trajectories and evaluation adaptation also affect the comparison.</p>

<p class="source">examples-RL/hard_gain_vdp.py. Results are read from the checked-in export, not regenerated during presentation preparation.</p>

---

# A7. Standard Van der Pol: the less favorable result

<img src="/images/standard-vdp-comparison.png" alt="Standard Van der Pol comparison: RLS-JFB, Oracle-JFB and BPTT on 256 fresh initial conditions." class="figure figure-backup" />

<p class="small">Mean costs on 256 test initial conditions: RLS 8.792, oracle 7.995, BPTT 7.734. RLS–BPTT gap: 13.7%.</p>

---

# A8. Hard Van der Pol

<img src="/images/hard-vdp-comparison.png" alt="Hard Van der Pol comparison with exponential control penalty and constant control gain." class="figure figure-backup" />

<p class="source">Original presentation export. Constant gain isolates the nonlinear control penalty from the state-dependent gain in the main benchmark.</p>

---

# A9. Portfolio: an implicit financial sanity check

Wealth follows a deterministic drift model hidden from the agent:

$$
\dot W=rW+\pi(\mu-r)W,\qquad
L=\lambda(e^{\pi^2}-1),\qquad G=-\log(W/W_{\mathrm{ref}}).
$$

With $H=L+p f$, interior optimality requires

$$
2\lambda\pi e^{\pi^2}+p(\mu-r)W=0.
$$

For positive wealth and a constant allocation, the objective reduces to

$$
J(\pi)=T\left[\lambda(e^{\pi^2}-1)-r-\pi(\mu-r)\right]-\log(W_0/W_{\mathrm{ref}}).
$$

Thus the reference solves $2\lambda\pi^\star e^{(\pi^\star)^2}=\mu-r$.

<p class="source">Project report §5.7. A scalar, deterministic Merton-type benchmark; it does not validate learning under stochastic returns.</p>

---

# A10. Portfolio policy against its analytical reference

<img src="/images/portfolio-report.png" alt="Report Figure 4: learned wealth and allocation compared with the analytical portfolio optimum." class="figure figure-portfolio" />

<p class="small">The learned wealth tracks the reference, but the allocation is time-varying. This is a sanity check, not an exact recovery of the optimal feedback.</p>

<p class="source">Project report, Fig. 4. Allocation is approximately 0.79–0.82 versus a constant reference near 0.85.</p>

---

# A11. Contractivity requires a step-size condition

For an unprojected minimization step $T(u)=u-\alpha_{\mathrm{fp}}\nabla_u H$,

$$
\partial_uT=I-\alpha_{\mathrm{fp}}H_{uu}.
$$

If $\mu I\preceq H_{uu}\preceq LI$ uniformly on the region of interest, with $\mu>0$,

$$
0<\alpha_{\mathrm{fp}}<\frac{2}{L}
\quad\Longrightarrow\quad
\|\partial_uT\|_2\leq\max\{|1-\alpha_{\mathrm{fp}}\mu|,|1-\alpha_{\mathrm{fp}}L|\}<1.
$$

For the hard penalty,

$$
\ell''(u)=2\lambda_u e^{u^2}(1+2u^2).
$$

Control projection is nonexpansive, but cannot by itself establish contraction of the preceding map.

<p class="source">The hard runs use a fixed-point step of 0.05. This does not certify uniform contraction across [−2, 2]; empirical solver convergence must be checked.</p>

---

# A12. Value network and terminal condition

The inherited value surrogate uses a residual network and quadratic head:

$$
\phi_\theta(t,z)=w^\top\operatorname{ResNet}([z;t])
+\tfrac12[z;t]^\top A^\top A[z;t]+b^\top[z;t]+c.
$$

The policy queries $\nabla_z\phi_\theta$; it does not directly predict the control.

An alternative wrapper could enforce the terminal value exactly:

$$
\phi_\theta(t,z)=G(z)+(T-t)N_\theta(t,z).
$$

This wrapper is separate from the RL contribution. The report does not use its unsuccessful liquidation results as evidence.

<p class="source">Project report §4.2; inherited architecture in core/ImplicitNets.py.</p>

---

# A13. Liquidation is an application of the original framework

For $z=(q,S)$, absorb the cash flow into the running cost:

$$
\dot q_i=-u_i,\qquad \dot S_i=-\kappa_i u_i,
$$

$$
L'=\tfrac12\sum_i\sigma_i^2q_i^2-\sum_i S_i u_i
+\sum_i\eta_i(u_i^2+\epsilon)^{\gamma/2},
\qquad G'=\alpha\sum_iq_i(T)^2.
$$

At $\gamma=2$, using $H=L'+p^\top f$,

$$
u_i^\star=\frac{S_i+p_{q_i}+\kappa_i p_{S_i}}{2\eta_i}.
$$

Removing cash reduces the state and removes learned $p_X$ from the control curvature. Under the negative Hamiltonian convention, the full-state curvature is $p_X\eta\psi''(u)$, negative at the correct $p_X=-1$. Thus it does not establish structural divergence.

<p class="source">Application / reproduction context, not the main RL contribution. Reduced model checked against models/LiquidationPortfolio.py.</p>

---

# A14. Stochastic control is a separate extension

For price noise,

$$
\mathrm dS_i=-\kappa_i u_i\,\mathrm dt+\sigma_S\,\mathrm dW_i.
$$

Under the minimization convention, the value equation is

$$
V_t+\min_u\left[L+\nabla V^\top f
+\tfrac12\operatorname{Tr}(\Sigma\Sigma^\top\nabla^2V)\right]=0.
$$

- Control-independent diffusion adds no explicit diffusion derivative to the control first-order condition.
- The value gradient can still change; additive noise alone does not establish certainty equivalence for a general nonlinear problem.
- The RL benchmarks presented here use deterministic environment transitions.

<p class="takeaway">Learning unknown stochastic dynamics would require additional analysis and experiments.</p>

<p class="source">Project report §6 and stochastic infrastructure in core/ImplicitOC.py. No stochastic optimality or timing-premium claim is made in this talk.</p>

---

# A15. References

<div class="small">

- Hanachowicz, Adrogué, Stekelorom and Barré. *Implicit Hamiltonian Optimal Control with Unknown Dynamics via Jacobian-Free Backpropagation.* Project report in this repository, §§5.1–5.8 and Appendix A.6.
- Gelphman et al. (2025). *End-to-End Training of High-Dimensional Optimal Control with Implicit Hamiltonians via Jacobian-Free Backpropagation.* arXiv:2510.00359.
- Gelphman et al. (2026). *On the Convergence of Jacobian-Free Backpropagation for Optimal Control Problems with Implicit Hamiltonians.* arXiv:2602.00921.
- Wu Fung et al. (2022). *JFB: Jacobian-Free Backpropagation for Implicit Networks.* AAAI 36(4), 3849–3857.
- Eberhard, Vernade and Muehlebach (2025). *A Pontryagin Perspective on Reinforcement Learning.* L4DC, PMLR 283, 233–244.

</div>

<p class="source">Project report bibliography.</p>
