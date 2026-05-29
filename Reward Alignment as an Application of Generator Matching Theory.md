# Reward Alignment as an Application of Generator Matching Theory

## Part III in the Series

This note is intended as **Part III** after:
1. **Flow/Diffusion lecture notes** (`lecture_notes.pdf`), and
2. **Generator Matching Theory Companion Notes**.

It keeps the same notation:\
\(p_t\) (probability path), \(L_t\) (generator), \(L_t^*\) (adjoint), \(p_{1\mid t}(z\mid x)\) (posterior), and conditional-to-marginal averaging from GM.

We only cover **new reward-alignment concepts**, treating earlier flow/diffusion/GM basics as already known.

---

## 1. Reward Alignment in GM Language

### 1.1 Base path and aligned path

Let a pretrained generator induce marginals \((p_t)_{t\in[0,1]}\), with terminal \(p_1\approx p_{\text{data}}\).\
Given reward \(r(x)\), alignment usually targets a tilted terminal law:
\[
\pi_1(x) \propto p_1(x)\exp(\beta r(x)).
\]
A path-space version uses trajectory rewards \(R(X_{0:1})\):
\[
\frac{\mathrm d\Pi^{\text{align}}}{\mathrm d\Pi^{\text{base}}}(X_{0:1}) \propto \exp\big(\beta R(X_{0:1})\big),
\]
which is the Feynman--Kac viewpoint used in inference-time steering.

### 1.2 GM perspective on alignment

In GM terms, reward alignment is selecting a new generator family \(\widetilde L_t\) such that:
\[
\partial_t\langle \widetilde p_t,f\rangle = \langle \widetilde p_t,\widetilde L_t f\rangle,
\]
with terminal \(\widetilde p_1\) matching reward preferences.\
So alignment is **generator design/training under a reward-tilted target law**, not a separate paradigm.

### 1.3 A common regularized objective

Many alignment methods instantiate:
\[
\max_{q}\; \mathbb E_{x\sim q}[r(x)]-\tau\,D_{\mathrm{KL}}\big(q\|p_1\big),
\]
whose optimizer is exactly Gibbs-tilted \(q^*(x)\propto p_1(x)e^{r(x)/\tau}\).\
This objective appears across flow/diffusion RL fine-tuning, reward-guided sampling, and map distillation.

---

## 2. Paper-by-Paper: Main Reward-Alignment Ideas in GM Form

## 2.1 GLASS Flows (arXiv:2509.25170)
**Paper:** *GLASS Flows: Transition Sampling for Alignment of Flow and Diffusion Models*.

### Core contribution
GLASS introduces efficient **stochastic transition sampling** for alignment pipelines that previously relied on expensive SDE rollouts. Conceptually, it builds a transition kernel
\[
K_t^{\text{GLASS}}(x,\mathrm dy)
\]
from a pretrained flow/diffusion model so that we retain stochasticity (needed for search/SMC/reward steering) while preserving much of ODE-like efficiency.

### GM interpretation
- Base model gives generator/path \((L_t,p_t)\).
- GLASS gives a practical aligned transition operator usable by reward samplers.
- This is a computational realization of GM Principle 3/4 in an alignment setting: keep same modeling objects (Markov transitions/generators), change the sampling/training mechanics for reward objectives.

### Why it matters
It removes the usual tradeoff:
- deterministic ODE sampling: fast but weak for reward exploration,
- full SDE sampling: expressive but expensive.

GLASS provides an intermediate that supports Feynman--Kac-style steering and transition-based alignment at scale.

---

## 2.2 Diamond Maps (arXiv:2602.05993)
**Paper:** *Diamond Maps: Efficient Reward Alignment via Stochastic Flow Maps*.

### Core idea
Diamond Maps distill expensive aligned transition dynamics into a **stochastic map model** for efficient inference-time alignment.

A standard aligned target is written as
\[
\pi(x)\propto p_{\text{base}}(x)e^{r(x)}.
\]
Distillation objective (map-to-teacher matching) is typically KL-type:
\[
\min_{\phi} D_{\mathrm{KL}}\big(p_{\text{teacher}}\|p_\phi\big).
\]

Alignment-time objective is regularized reward maximization:
\[
\max_{p_\phi}\;\mathbb E_{x\sim p_\phi}[r(x)]-\beta D_{\mathrm{KL}}(p_\phi\|p_{\text{base}}).
\]

### GM interpretation
- Teacher dynamics (e.g., GLASS transitions) correspond to an implicit aligned generator process.
- Diamond map is an amortized approximation to that aligned Markov evolution.
- In GM language: learn a low-cost surrogate parameterization \(F_t^\phi\) that preserves reward-aligned marginals and transition behavior needed for SMC/search/value estimation.

### Application emphasis
Efficient reward alignment for high-cost generation settings (notably text-to-image inference pipelines).

---

## 2.3 Energy-based Generator Matching (EGM) (arXiv:2505.19646)
**Paper:** *Energy-based generator matching: A neural sampler for general state space*.

### Core idea
EGM replaces “data distribution target” with an **energy-defined target**:
\[
\pi(x)=\frac{e^{-E(x)}}{Z},\quad Z=\int e^{-E(x)}\,dx,
\]
and trains a Markov generator model directly toward this target, including continuous/discrete/mixed spaces.

A canonical training form is KL minimization:
\[
\min_\theta D_{\mathrm{KL}}(p_\theta\|\pi)
=\min_\theta\;\mathbb E_{x\sim p_\theta}[\log p_\theta(x)+E(x)] + \text{const}.
\]

Since \(\pi\) is unnormalized, EGM uses self-normalized importance weights (SNIS), e.g.
\[
\widetilde w_i=\frac{w_i}{\sum_j w_j},\qquad
w_i\propto \frac{e^{-E(x_i)}}{p_\theta(x_i)},
\]
and estimates expectations by \(\sum_i \widetilde w_i f(x_i)\).

### GM interpretation
This is a direct extension of GM/CGM to **reward-as-energy** supervision.\
If we set \(E(x)=-\beta r(x)\), EGM is exactly reward alignment by generator matching to a tilted target.

### New concept vs previous notes
The key new step is not generator decomposition, but **target specification via unnormalized energy/reward** and variance-controlled estimation for training.

---

## 2.4 Flow-GRPO (arXiv:2505.05470)
**Paper:** *Flow-GRPO: Training Flow Matching Models via Online RL*.

### Core idea
Flow-GRPO adapts online policy-gradient style optimization (GRPO family) to flow-matching generators.

Group-relative advantage form:
\[
A_i = r_i - \frac{1}{N}\sum_{j=1}^N r_j.
\]
A representative policy-gradient-style objective is:
\[
\mathcal L_{\text{GRPO}}(\theta)
= -\mathbb E\Big[\sum_i A_i\log p_\theta(\text{trajectory}_i\mid \text{context})\Big].
\]

### ODE-to-SDE bridge
Flow models are often sampled by deterministic ODE trajectories. RL exploration needs stochasticity, so Flow-GRPO uses an SDE view:
\[
\mathrm dX_t = f_\theta(X_t,t)\,dt + g_\theta(X_t,t)\,dW_t,
\]
with design choices that preserve target marginals while enabling stochastic rollout diversity for RL updates.

### GM interpretation
- Flow generator \(L_t\) is updated with reward-weighted gradients.
- This is GM Principle 4 with reward-dependent weighting/online sampling instead of pure supervised CGM.
- Alignment occurs by learning generator parameters that increase expected reward while controlling degeneration.

---

## 2.5 Discrete Flow Maps (arXiv:2604.09784)
**Paper:** *Discrete Flow Maps*.

### Core idea
For discrete data (e.g., language), map training must respect simplex geometry.\
Instead of Euclidean regression losses, use simplex-consistent losses such as cross-entropy/KL.

Let \(\Delta_K=\{p\in\mathbb R^K_+:\sum_k p_k=1\}\).\
A map outputs distributions on \(\Delta_K\), trained with e.g.
\[
\mathcal L(\theta)=\mathbb E\big[\mathrm{KL}(q(\cdot\mid x)\|\mu_\theta(\cdot\mid x))\big].
\]

### GM interpretation
This is the discrete-space analogue of choosing a valid generator parameterization and divergence in CGM.\
For reward alignment, replace \(q\) by reward-tilted or advantage-weighted targets over tokens/sequences.

### Application angle
Enables fast non-autoregressive discrete generation and provides an alignment-ready interface in token space.

---

## 2.6 DRIFT / Discrete Flow Matching for RL (arXiv:2605.12379)
**Paper:** *Discrete Flow Matching for Offline-to-Online Reinforcement Learning*.

### Core idea
A CTMC/discrete-flow policy pretrained offline is fine-tuned online using advantage weighting and trajectory-level regularization.

Representative structure:
\[
\mathcal L_{\text{AW-DFM}}(\theta)
=\mathbb E\Big[w(s,a)\,\ell_{\text{DFM}}(\theta;s,a,t)\Big],
\]
where \(w(s,a)\) is advantage-derived. Path-space regularization adds:
\[
\mathcal L_{\text{total}}=\mathcal L_{\text{AW-DFM}}+\lambda\,\mathcal L_{\text{path-reg}}.
\]

### GM interpretation
- Discrete generator/rate matrix learning (from GM + lecture Section 7) is retained.
- Reward alignment acts through advantage-weighted target flows plus regularization to keep dynamics near useful pretrained behavior.
- This is a clear example of **reward alignment as generator retargeting on discrete state spaces**.

---

## 3. Unifying Algorithmic Template (GM-Centric)

Given pretrained GM model \(L_t^{\theta_0}\):

1. **Specify reward signal** \(r\) (terminal or path-level).\
2. **Define aligned target law** (Gibbs/Feynman--Kac tilt).\
3. **Choose alignment engine**:
   - transition steering (GLASS/FKS),
   - amortized map distillation (Diamond),
   - energy-target matching (EGM),
   - online RL weighting (Flow-GRPO/DRIFT),
   - simplex/discrete map objectives (Discrete Flow Maps).
4. **Train/update generator parameterization** with KL/Bregman/reward-weighted losses.
5. **Sample with stochastic transitions** when exploration/search is needed.
6. **Evaluate reward-quality-diversity tradeoff** (reward, fidelity, diversity, robustness to reward hacking).

This is exactly “reward alignment as an application layer on top of GM”.

---

## 4. Key Definitions (New in Reward-Alignment Context)

1. **Reward-tilted target distribution**
\[
\pi(x)\propto p(x)e^{\beta r(x)}.
\]

2. **Path-space tilt (Feynman--Kac)**
\[
\frac{d\Pi^{\text{align}}}{d\Pi^{\text{base}}}(X_{0:1})\propto e^{\beta R(X_{0:1})}.
\]

3. **Amortized alignment map**
A stochastic map \(T_\phi\) trained to approximate aligned transition/output laws with reduced inference cost.

4. **Advantage-weighted generator matching**
GM/DFM losses reweighted by reward-derived advantages to shift generator mass toward high-reward regions.

5. **Path-space regularization**
A penalty on entire trajectory laws (not only terminal samples), stabilizing online alignment and preserving diversity.

---

## 5. Applications Reported Across the Main Papers

- **Text-to-image alignment:** higher preference/reward scores with better compute scaling (GLASS, Diamond, Flow-GRPO).
- **General-state-space reward sampling:** continuous + discrete + multimodal generation from energy/reward functions (EGM).
- **Discrete decision/control problems:** offline-to-online RL improvements via discrete flow/CTMC alignment (DRIFT).
- **Fast discrete generation:** simplex-consistent map training with alignment hooks in token probability space (Discrete Flow Maps).

---

## 6. Additional Papers (Main Concepts Only)

### 6.1 arXiv:2501.09685
**Inference-Time Alignment in Diffusion Models with Reward-Guided Generation: Tutorial and Review**
- Unifies inference-time reward alignment methods (guidance, SMC/search, value-based steering).
- Important for this note because it frames alignment as **sampling/control over pretrained generative trajectories**, exactly compatible with GM’s generator/process viewpoint.

### 6.2 arXiv:2604.27147
**How to Guide Your Flow: Few-Step Alignment via Flow Map Reward Guidance**
- Main concept: few-step alignment by reward-guided flow-map updates, aiming to reduce alignment compute while retaining quality.
- Relevance to GM: efficient approximation of reward-aligned generator dynamics in low-step regimes.

### 6.3 arXiv:2502.06061
**Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization**
- Main concept: online reward-weighted flow matching with explicit transport regularization (Wasserstein) to avoid collapse/reward hacking.
- GM view: reward-weighted CGM-style updates + geometric regularization to preserve distributional coverage.

---

## 7. What Is New Relative to the Previous Two Documents

This Part III adds:
- reward/objective tilting as first-class target specification,
- inference-time alignment as Markov transition control,
- map distillation for efficient aligned sampling,
- online RL fine-tuning of flow/discrete generators,
- path-space regularization in alignment loops.

It intentionally does **not** re-derive:
- continuity/Fokker--Planck basics,
- conditional-to-marginal posterior identities,
- full GM generator decomposition proofs,
- core CTMC preliminaries from lecture/companion notes.

---

## 8. Practical Takeaway

If GM is the “physics” of generative dynamics through generators and KFE, then reward alignment is an **optimal-control layer** that retargets these dynamics toward preference-weighted distributions. The recent papers above can be read as different engineering choices for this same mathematical program: change the effective generator/transition law to optimize reward while preserving sample quality, diversity, and tractability.
