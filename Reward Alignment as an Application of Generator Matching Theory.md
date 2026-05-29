# Lecture Notes: Reward Alignment as an Application of Generator Matching Theory

## Part III in the Series

These notes are written as **lecture notes** for the reward-alignment layer that sits on top of the probability-path, flow/diffusion, CTMC, and Generator Matching (GM) foundations developed earlier.

They assume familiarity with the following objects:
- \(p_t\): a probability path over states,
- \(L_t\): a time-dependent infinitesimal generator,
- \(L_t^*\): the adjoint generator in the Kolmogorov forward equation,
- \(K_{t+h\mid t}\): a Markov transition kernel,
- conditional-to-marginal averaging identities from Conditional Generator Matching (CGM).

The present document asks a more advanced question:

> Once a pretrained generator already produces high-quality samples from a base law, how should we **retarget the dynamics** so that the terminal or path-space law prefers high-reward outcomes while preserving sample quality, diversity, and computational tractability?

The core thesis is that reward alignment is not a separate paradigm from GM. It is a **change-of-target-law** problem, and therefore a **change-of-generator / change-of-transition-law** problem.

---

## Lecture 0 — Mathematical Preliminaries for Reward Alignment

### 0.1 Base model, aligned model, and terminal tilting

Let a pretrained generative model induce a path \((p_t)_{t\in[0,1]}\), with terminal marginal \(p_1\). In the base setting we think of \(p_1\) as approximating a data or preference-agnostic target. Reward alignment introduces a reward function \(r:S\to\mathbb R\) and seeks a new terminal law \(\pi_1\) that concentrates more mass on high-reward states.

The canonical aligned terminal law is the Gibbs tilt
\[
\pi_1(x) = \frac{1}{Z_\beta}p_1(x)e^{\beta r(x)},
\qquad
Z_\beta = \int p_1(x)e^{\beta r(x)}dx,
\]
where \(\beta>0\) controls alignment strength.

This form is not arbitrary. It is the optimizer of the KL-regularized variational problem
\[
\max_q \; \mathbb E_{x\sim q}[r(x)] - \tau D_{\mathrm{KL}}(q\|p_1),
\]
with \(\tau = 1/\beta\). Thus, reward alignment is a constrained optimization problem balancing reward seeking against deviation from the pretrained law.

### 0.2 Path-space tilting and the Feynman--Kac viewpoint

If the reward is path-dependent, one works not merely with terminal samples but with the full law \(\Pi^{\text{base}}\) over trajectories \(X_{0:1} = (X_t)_{t\in[0,1]}\). The aligned path law is then
\[
\frac{d\Pi^{\text{align}}}{d\Pi^{\text{base}}}(X_{0:1})
\propto e^{\beta R(X_{0:1})},
\]
where \(R\) is a terminal reward, a trajectory reward, or a cumulative potential.

This is the Feynman--Kac viewpoint: the reward reweights entire trajectories, not only final states. From a stochastic-process perspective, alignment can therefore be seen as transforming the effective semigroup or transition law so that the new process places more probability on reward-rich paths.

### 0.3 Generator-language statement of the alignment problem

Suppose the base process satisfies
\[
\partial_t p_t = L_t^* p_t.
\]
Reward alignment asks us to construct a new process with generator \(\widetilde L_t\) and marginals \(\widetilde p_t\) such that
\[
\partial_t \widetilde p_t = \widetilde L_t^* \widetilde p_t,
\]
and such that \(\widetilde p_1\) approximates the reward-tilted target law or its path-space analogue.

The engineering question is then:
- should alignment modify sampling only,
- should it distill a new transition operator,
- should it directly retrain the generator,
- should it use online reward-weighted updates,
- should it operate in continuous or discrete state spaces?

The papers below answer these questions in different ways.

### 0.4 A useful structural dichotomy

Across the literature, reward-alignment methods typically fall into two broad classes:

1. **Inference-time control of a fixed base generator**  
   The pretrained model remains unchanged, but sampling is altered through guidance, transition weighting, SMC, or reward-conditioned correction.

2. **Amortized / fine-tuned alignment**  
   One trains a new map, generator, or policy so that aligned generation becomes cheap at deployment time.

This distinction will organize the lecture sequence.

---

## Lecture 1 — GLASS Flows: Transition Sampling for Alignment of Flow and Diffusion Models

**Paper:** *GLASS Flows: Transition Sampling for Alignment of Flow and Diffusion Models*.

### 1.1 Problem setting

A major obstacle in aligning flow and diffusion models is the tradeoff between:
- **deterministic ODE sampling**, which is cheap but weak for exploration and reward-search,
- **full SDE sampling**, which preserves stochasticity but can be expensive.

GLASS addresses the middle regime: it seeks stochastic transition operators that preserve the useful geometry of the pretrained model while enabling efficient reward-aware sampling.

### 1.2 Transition-operator perspective

Let the pretrained base model induce short-time transitions
\[
K^{\text{base}}_{t+h\mid t}(x,dy).
\]
GLASS constructs a new transition operator
\[
K^{\text{GLASS}}_{t+h\mid t}(x,dy)
\]
that is cheap to sample from, retains controlled stochasticity, and is suitable for alignment-time procedures such as search, SMC, or reward reweighting.

Conceptually, this means that GLASS does not only change terminal samples; it changes the **transition geometry** by supplying a practical stochastic kernel compatible with the pretrained dynamics.

### 1.3 GM interpretation

In GM language, a Markov model is defined by its generator or equivalently by its infinitesimal transition law. GLASS is therefore naturally interpreted as an **approximate aligned transition mechanism** built on top of a pretrained generator family.

If the base law is governed by \((L_t,p_t)\), then GLASS provides a tractable kernel-level approximation to the aligned process that one would ideally obtain by reward tilting the path measure. It is not merely a heuristic sampler; it is an approximation to a new Markov evolution.

### 1.4 Why stochasticity matters mathematically

Reward alignment generally benefits from sampling diversity because the aligned objective often depends on rare-event discovery. Deterministic transport maps may be too rigid: once the trajectory is fixed by the initial seed, there is no local exploration around promising regions. Stochastic transitions instead allow the process to explore local neighborhoods, making particle-based and reward-weighted methods viable.

From the Feynman--Kac perspective, one often needs to repeatedly propagate and reweight a cloud of candidate trajectories. GLASS provides the missing object that makes such propagation computationally feasible in flow/diffusion settings.

### 1.5 Theoretical role in the alignment stack

GLASS should be read as a **sampling-layer paper**:
- it does not fundamentally redefine the base target law,
- it makes the aligned path law computationally accessible,
- it provides the transition-level substrate on which stronger alignment procedures can operate.

That is why it is best understood before map distillation papers such as Diamond Maps.

---

## Lecture 2 — Diamond Maps: Efficient Reward Alignment via Stochastic Flow Maps

**Paper:** *Diamond Maps: Efficient Reward Alignment via Stochastic Flow Maps* (arXiv:2602.05993).

### 2.1 Central objective

Diamond Maps starts from the observation that high-quality aligned samplers may be too expensive to run online. If a powerful teacher sampler already produces approximate aligned samples from
\[
\pi(x) \propto p_{\text{base}}(x)e^{r(x)},
\]
then one can attempt to **distill** that aligned behavior into a cheap stochastic map.

### 2.2 Distillation objective

The core mathematical form is teacher-student distribution matching, typically expressed as a KL minimization:
\[
\min_\phi D_{\mathrm{KL}}(p_{\text{teacher}}\|p_\phi).
\]
Here \(p_{\text{teacher}}\) is the distribution induced by the expensive aligned process, while \(p_\phi\) is the learned stochastic flow map.

This objective says: instead of solving the reward-alignment control problem from scratch at inference time, learn a fast amortized approximation to the aligned Markov evolution.

### 2.3 Variational meaning

The aligned target often arises from regularized reward maximization:
\[
\max_q \; \mathbb E_q[r(x)] - \beta D_{\mathrm{KL}}(q\|p_{\text{base}}).
\]
Diamond Maps inherits this structure indirectly. The teacher is already solving or approximating the regularized reward-alignment problem; the student then distills the resulting aligned law.

Thus, Diamond Maps is a **second-order approximation strategy**:
- first solve alignment approximately using a strong but costly process,
- then compress that aligned dynamics into a cheaper stochastic map.

### 2.4 GM interpretation

GM parameterizes a Markov generator through learnable objects \(F_t\). Diamond Maps can be read as learning an amortized surrogate \(F_t^\phi\) that preserves the reward-aligned marginals and transition behavior of the teacher. In other words, the paper replaces repeated expensive control with a learned **operator surrogate**.

This is important conceptually: reward alignment is still generator design, but Diamond Maps moves that design into an amortized approximation regime.

### 2.5 Why this matters

In practical systems, alignment must often be deployed many times. A slow teacher process may be acceptable during research but not at serving time. Diamond Maps therefore answers a deployment question central to aligned generation:

> Can we retain the geometry of reward-aware stochastic transport without paying the full alignment cost at test time?

Its answer is yes, by learning a stochastic map that approximates the teacher process closely enough.

### 2.6 Limitation and theory tension

The main tension is compression error. If the student map underfits the aligned teacher, then reward performance drops; if it overfits narrow reward modes, diversity collapses. Thus the paper lives in the classical triple tradeoff among reward, fidelity to the base law, and amortization quality.

---

## Lecture 3 — Energy-Based Generator Matching (EGM): Reward as Energy, Generator as Sampler

**Paper:** *Energy-based generator matching: A neural sampler for general state space*.

### 3.1 Shift in viewpoint

EGM changes the target-specification layer. Instead of saying “match a data distribution,” it says “sample from an energy-defined target law”
\[
\pi(x)=\frac{e^{-E(x)}}{Z}.
\]
If we define
\[
E(x) = -\beta r(x),
\]
then reward alignment becomes an energy-based sampling problem.

This is conceptually powerful because it removes the need to begin from a normalized data law. The target may be specified only up to an unknown partition function, yet training can still proceed.

### 3.2 KL objective and reward form

A canonical optimization target is
\[
\min_\theta D_{\mathrm{KL}}(p_\theta\|\pi)
= \min_\theta \; \mathbb E_{x\sim p_\theta}[\log p_\theta(x) + E(x)] + \text{const}.
\]
When \(E=-\beta r\), this becomes
\[
\min_\theta \; \mathbb E_{x\sim p_\theta}[\log p_\theta(x) - \beta r(x)] + \text{const},
\]
which is precisely reward seeking regularized by model entropy / likelihood structure.

### 3.3 Importance weighting and normalization difficulty

Because \(\pi\) is unnormalized, expectations under \(\pi\) cannot usually be computed directly. EGM therefore uses self-normalized importance sampling (SNIS):
\[
\widetilde w_i = \frac{w_i}{\sum_j w_j},
\qquad
w_i \propto \frac{e^{-E(x_i)}}{p_\theta(x_i)}.
\]
Then
\[
\mathbb E_{x\sim \pi}[f(x)] \approx \sum_i \widetilde w_i f(x_i).
\]

This is theoretically significant because it makes reward alignment possible even when the aligned target is only known up to proportionality.

### 3.4 GM interpretation

EGM is the most direct reward-alignment extension of GM. GM says: design generators to realize a target path law. EGM says: let the target be defined by an energy rather than by data samples. The generator then becomes a neural sampler for the energy-defined aligned law.

So, relative to previous GM notes, the novelty is not generator decomposition; it is **target specification under unnormalized reward-defined measures**.

### 3.5 Why it generalizes well

Because the framework is stated on general state spaces, it naturally extends to:
- continuous domains,
- discrete domains,
- hybrid domains,
- mixed compositional generation problems.

That makes EGM especially relevant when reward alignment must be performed in spaces that do not fit simple Euclidean score-matching assumptions.

### 3.6 Main theoretical caveat

Importance-weighted estimation can have high variance. Thus the practicality of reward-as-energy alignment depends critically on proposal quality, estimator stability, and the overlap between current model mass and high-reward regions.

---

## Lecture 4 — Flow-GRPO: Training Flow Matching Models via Online RL

**Paper:** *Flow-GRPO: Training Flow Matching Models via Online RL*.

### 4.1 The central problem

Flow matching is usually trained by supervised objectives derived from a prescribed probability path. But reward alignment is not supervised: one observes scalar rewards or preferences only after generating candidates. Flow-GRPO therefore imports online RL ideas into flow-model training.

### 4.2 Relative-advantage form

A representative group-relative advantage is
\[
A_i = r_i - \frac{1}{N}\sum_{j=1}^N r_j.
\]
This centers rewards within a sampled group and emphasizes relative preference rather than absolute scale.

A policy-gradient-style objective then takes the form
\[
\mathcal L_{\text{GRPO}}(\theta)
= -\mathbb E\left[\sum_i A_i \log p_\theta(\text{trajectory}_i\mid \text{context})\right].
\]

Although the exact implementation details depend on the flow parameterization, the conceptual point is clear: update generator parameters in the direction of trajectories with higher empirical advantage.

### 4.3 Why an ODE-only picture is insufficient

A pure flow model is often interpreted through deterministic ODE trajectories. But RL-style optimization benefits from exploration. Hence Flow-GRPO introduces or leverages an SDE interpretation,
\[
dX_t = f_\theta(X_t,t)dt + g_\theta(X_t,t)dW_t,
\]
so that stochastic rollouts can be used for reward discovery while still respecting the flow-matching structure.

This is a deep conceptual move: deterministic transport is recast as part of a stochastic family suitable for online optimization.

### 4.4 GM interpretation

GM teaches that a model class is specified by generators. Flow-GRPO says that those generator parameters need not be learned only by supervised conditional losses; they can also be updated by reward-weighted online objectives. In that sense, Flow-GRPO is a **reinforcement-learning instantiation of GM Principle 4**: generator parameters are optimized to satisfy a task objective, not merely a data-fit criterion.

### 4.5 What is gained

- reward-aware adaptation beyond the offline dataset,
- online discovery of behaviors absent from demonstrations,
- principled integration of flow-model structure with RL-style exploration.

### 4.6 What becomes difficult

Online reward optimization risks mode collapse, reward hacking, or loss of base-model fidelity. Hence Flow-GRPO should be understood as a controlled generator-retargeting method, not as unconstrained reward maximization.

---

## Lecture 5 — Discrete Flow Maps

**Paper:** *Discrete Flow Maps* (arXiv:2604.09784).

### 5.1 Why discrete spaces are special

In discrete generation, the model outputs probabilities over tokens or categories. The ambient geometry is therefore the simplex
\[
\Delta_K = \{p\in\mathbb R_+^K : \sum_{k=1}^K p_k = 1\},
\]
not Euclidean space.

This changes the mathematics in an essential way. Euclidean regression losses may be poorly aligned with the geometry of probability vectors, whereas cross-entropy and KL divergences are natural because they respect simplex structure.

### 5.2 Proper training objective

A standard objective is
\[
\mathcal L(\theta)
= \mathbb E\big[ D_{\mathrm{KL}}(q(\cdot\mid x) \| \mu_\theta(\cdot\mid x)) \big],
\]
where \(q\) is a target distribution and \(\mu_\theta\) is the learned discrete map.

The significance is not just numerical convenience. The divergence is part of the model definition because it encodes what it means to move correctly on the simplex.

### 5.3 Generator interpretation in discrete space

Discrete flow maps connect naturally to CTMC-style generators
\[
[L_t f](x)=\sum_{y\in S}Q_t(y\mid x)(f(y)-f(x)),
\]
with forward equation
\[
\partial_t p_t(x) = \sum_y \big(Q_t(x\mid y)p_t(y)-Q_t(y\mid x)p_t(x)\big).
\]

Thus, discrete alignment is still generator alignment, but the admissible operators and losses must respect discrete probability geometry.

### 5.4 Reward-alignment relevance

For reward alignment, one replaces the target distribution \(q\) by reward-tilted token or sequence targets, or by advantage-weighted local targets. This makes Discrete Flow Maps an important bridge between token-level generation and alignment-time optimization.

### 5.5 Broader significance

This paper matters because many aligned systems eventually pass through discrete bottlenecks:
- language tokens,
- classifier outputs,
- latent mode variables,
- discrete decision policies.

A reward-alignment theory that ignores simplex geometry is therefore incomplete.

---

## Lecture 6 — DRIFT / Discrete Flow Matching for Offline-to-Online Reinforcement Learning

**Paper:** *Discrete Flow Matching for Offline-to-Online Reinforcement Learning*.

### 6.1 Offline-to-online transition

DRIFT addresses a setting where one begins with a pretrained discrete-flow or CTMC-style policy learned from offline data, then performs online improvement using rewards collected during deployment.

This is the discrete counterpart of the continuous alignment story: start from a useful base generator, then retarget it toward higher reward while controlling departure from pretrained behavior.

### 6.2 Advantage-weighted objective

A representative form is
\[
\mathcal L_{\text{AW-DFM}}(\theta)
= \mathbb E\Big[w(s,a)\,\ell_{\text{DFM}}(\theta;s,a,t)\Big],
\]
where \(w(s,a)\) is advantage-derived and \(\ell_{\text{DFM}}\) is a discrete-flow matching loss.

The structure is revealing:
- the base discrete-flow loss remains,
- reward enters through importance or advantage weighting,
- the generator is shifted toward higher-value actions without discarding the pretrained geometry.

### 6.3 Path-space regularization

To prevent instability, one often adds a path regularizer:
\[
\mathcal L_{\text{total}} = \mathcal L_{\text{AW-DFM}} + \lambda \mathcal L_{\text{path-reg}}.
\]
This matters because online reward optimization can distort not only terminal behavior but the full trajectory law. Path regularization constrains the entire generated dynamics to remain close to the useful base process.

### 6.4 GM interpretation

In GM terms, DRIFT is a clean example of **generator retargeting on discrete state spaces**. The base rate matrix or discrete transition law is not discarded; it is updated using reward-sensitive weights while preserving the structural form that makes the model trainable.

### 6.5 Conceptual importance

DRIFT closes a gap in the alignment literature. Much alignment theory is developed for continuous samplers or language-model-style objectives. DRIFT shows that offline-to-online reward alignment can also be expressed in the language of discrete flows and CTMC generators.

---

## Lecture 7 — Unified View: What the Six Papers Are Actually Doing

### 7.1 Same objective, different intervention points

All six lectures can be organized by **where** they intervene in the aligned-generation pipeline:

- **GLASS** modifies transition sampling to make aligned exploration feasible.
- **Diamond Maps** amortizes expensive aligned dynamics into a fast stochastic map.
- **EGM** changes target specification by treating reward as an energy.
- **Flow-GRPO** performs online reward-weighted updates of continuous generator parameters.
- **Discrete Flow Maps** supplies the right geometry and loss structure for discrete aligned maps.
- **DRIFT** performs offline-to-online reward alignment in discrete-flow settings.

### 7.2 A common master objective

At a high level, each method can be understood as approximating some solution to
\[
\max_{\mathcal P} \; \mathbb E_{X\sim \mathcal P}[R(X)] - \lambda \, \mathrm{Reg}(\mathcal P \| \mathcal P_{\text{base}}),
\]
where:
- \(\mathcal P\) may be a terminal law, path law, map, or generator-induced process,
- \(R\) is a terminal or trajectory reward,
- \(\mathrm{Reg}\) measures deviation from the pretrained process.

The difference across papers lies in the choice of:
- representation of \(\mathcal P\),
- optimization strategy,
- sampling scheme,
- geometry of the state space,
- regularizer or trust region.

### 7.3 Continuous vs discrete alignment

The lecture sequence also clarifies a key dichotomy:
- In continuous spaces, reward alignment often looks like stochastic control over ODE/SDE dynamics.
- In discrete spaces, reward alignment becomes a problem of retargeting rate matrices, token distributions, or simplex-valued maps.

GM is valuable precisely because it gives one language that covers both.

### 7.4 Why GM is the right meta-framework

GM is not merely another training loss. It is a unifying theory of Markov generative models at the generator level. That makes it the right conceptual framework for reward alignment because every alignment method ultimately changes one of the following:
- the target law,
- the transition law,
- the generator parameters,
- the divergence used to compare local dynamics,
- the sampling process used to realize the aligned law.

All of those are native GM objects.

---

## Lecture 8 — Practical Design Recipe for Reading Future Reward-Alignment Papers

When reading a new paper in this area, the following checklist is useful.

### 8.1 Step 1: Identify the base object

Ask whether the paper starts from:
- a pretrained diffusion model,
- a flow model,
- a CTMC/discrete flow model,
- a generic GM generator.

### 8.2 Step 2: Identify the aligned target

Ask whether reward enters as:
- a Gibbs tilt,
- a path-space Feynman--Kac weight,
- an energy function,
- an advantage-weighted surrogate,
- a distillation target from an aligned teacher.

### 8.3 Step 3: Identify the intervention point

Ask whether the method changes:
- only inference-time sampling,
- the transition kernel,
- the generator parameters,
- the objective function,
- the geometry/divergence used in the local loss.

### 8.4 Step 4: Identify the regularizer

Every successful alignment method controls deviation from the base model. The regularizer may be explicit (KL, Wasserstein, path regularization) or implicit (teacher imitation, constrained transition design).

### 8.5 Step 5: Ask what is being preserved

A strong method preserves some subset of the following:
- sample quality,
- diversity,
- mode coverage,
- stability,
- computational efficiency,
- compatibility with the pretrained generator geometry.

This checklist reveals that most papers differ more in engineering choice than in underlying mathematics.

---

## Final Summary

These lecture notes present reward alignment as a single mathematical program:

1. Start with a pretrained Markov generative process.
2. Specify a reward-tilted terminal or path-space law.
3. Construct or approximate aligned transitions or generators.
4. Control deviation from the pretrained law through KL, path, transport, or teacher-based regularization.
5. Deploy the aligned process in either continuous or discrete state spaces.

Under this view:
- **GLASS** gives efficient stochastic transitions,
- **Diamond Maps** gives amortized aligned maps,
- **EGM** gives reward-as-energy target matching,
- **Flow-GRPO** gives online RL over continuous generators,
- **Discrete Flow Maps** gives simplex-correct discrete alignment,
- **DRIFT** gives offline-to-online discrete generator retargeting.

So the right slogan is:

> Reward alignment is not outside Generator Matching. It is Generator Matching with a reward-defined target law and a carefully chosen mechanism for realizing the new aligned dynamics.
