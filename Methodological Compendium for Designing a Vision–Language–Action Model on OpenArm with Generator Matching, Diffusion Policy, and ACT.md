# Lecture Notes: Vision–Language–Action Modeling with Generator Matching, Diffusion, ACT, and Reward Alignment

## How to Read These Notes

These notes are organized as a **lecture series**, with one lecture centered on each paper from the paper list:

1. arXiv:2410.20587  
2. arXiv:2304.13705  
3. arXiv:2406.09246  
4. arXiv:2602.05993  
5. arXiv:2604.09784  
6. arXiv:2412.06264

Each lecture is written in a theory-first style: assumptions, objectives, core equations, algorithmic implications, and links to robot-policy design.

---

## Chapter 1 — Lecture 1 (arXiv:2410.20587): Probability Paths, Flow/Diffusion Foundations, and Conditional Action Generation

### 1.1 Core modeling object
Let \(x \in S\), where \(S\) is either a Euclidean action space (continuous control) or a structured state space (tokens/sequences). Generative modeling is formulated via a probability path \((p_t)_{t\in[0,1]}\) with
\[
p_0 = p_{\text{simple}}, \qquad p_1 = p_{\text{data}}.
\]

A conditional path \(p_t(\cdot \mid z)\) with \(z\sim p_{\text{data}}\) induces \(p_t\) by marginalization:
\[
p_t(x) = \int p_t(x\mid z) p_{\text{data}}(z)\,dz.
\]

### 1.2 Flow and diffusion parameterizations
Two canonical dynamics over \(X_t\):

- **Flow/ODE model**
\[
\frac{dX_t}{dt} = u_t(X_t), \qquad \partial_t p_t = -\nabla\cdot(u_t p_t).
\]

- **Diffusion/SDE model**
\[
dX_t = u_t(X_t)dt + \sigma_t(X_t)dW_t,
\]
\[
\partial_t p_t = -\nabla\cdot(u_t p_t) + \frac12\nabla^2 : (\sigma_t\sigma_t^\top p_t).
\]

These equations establish that training a generative model is equivalent to learning dynamics that satisfy the chosen path constraints.

### 1.3 Why this matters for policies
For policy learning, replace data samples by expert actions or action chunks. Then
\[
p_{\text{data}}(a\mid o,c)
\]
becomes the target conditional law under observations \(o\) and language context \(c\). This turns control into conditional generative modeling of actions.

### 1.4 Theoretical takeaways
- Probability paths provide a unifying abstraction independent of architecture.
- Conditional losses can train tractable surrogate objects while targeting intractable marginals.
- These foundations support both continuous-time and discrete-time policy generators.

---

## Chapter 2 — Lecture 2 (arXiv:2304.13705): Action Chunking with Transformers (ACT)

### 2.1 Motivation and setup
ACT addresses low-level manipulation where one-step policies are brittle and autoregressive long-horizon control accumulates errors. The key idea is to predict **action chunks**:
\[
A_{t:t+H-1} = (a_t, a_{t+1},\dots,a_{t+H-1})
\]
conditioned on current perception and robot state.

### 2.2 Latent-variable sequence policy
A common formalization uses a latent \(z\):
\[
p_\theta(A_{t:t+H-1}\mid o_t) = \int p_\theta(A_{t:t+H-1}\mid z,o_t)p_\theta(z\mid o_t)dz.
\]
Training with demonstrations typically optimizes reconstruction plus latent regularization (e.g., ELBO-style decomposition):
\[
\mathcal L = \mathbb E_{q_\phi(z\mid A,o)}[-\log p_\theta(A\mid z,o)] + \beta\,D_{\mathrm{KL}}(q_\phi(z\mid A,o)\|p(z)).
\]

### 2.3 Temporal abstraction and stability
Chunk-level prediction provides:
- reduced compounding error compared with pure autoregressive one-step imitation,
- smoother control due to multi-step consistency constraints,
- implicit low-pass filtering over noisy supervision.

### 2.4 Robotics implications
For dexterous manipulation, ACT’s chunking naturally matches contact-rich sub-skills (reach, align, close gripper, retract). The chunk horizon \(H\) becomes a control knob balancing reactivity and coherence.

### 2.5 Limitations and theory questions
- Fixed \(H\) can be suboptimal for variable-duration skills.
- Latent collapse can reduce multimodal expressivity.
- Offline imitation objective may not align with reward-sensitive deployment objectives.

---

## Chapter 3 — Lecture 3 (arXiv:2406.09246): Vision–Language–Action (VLA) Modeling

### 3.1 Unified token-space perspective
VLA models represent images, language, and robot state/action in a shared transformer pipeline:
\[
h_t = F_\Theta(v_t, w, x_t), \qquad a_t \sim \pi_\Theta(\cdot\mid h_t).
\]
Here \(v_t\) are visual tokens, \(w\) language tokens, and \(x_t\) proprioception/state tokens.

### 3.2 Pretraining and adaptation objective
A practical decomposition is:
\[
\mathcal L_{\text{total}} = \lambda_{\text{vision}}\mathcal L_{\text{vision}} + \lambda_{\text{text}}\mathcal L_{\text{text}} + \lambda_{\text{act}}\mathcal L_{\text{act}}.
\]
The action term enforces control-relevant grounding while preserving foundation-model priors.

### 3.3 Action decoding choices
Given shared features \(h_t\), action heads can be:
- deterministic regressors,
- chunk predictors (ACT-style),
- diffusion decoders over action trajectories.

Thus VLA is an interface layer; the action-generation theory still comes from flow/diffusion/generator design.

### 3.4 Distribution shift and grounding
Generalization depends on preserving semantic grounding under embodiment changes. In theory terms, one seeks invariant conditional structure in
\[
p(a\mid o,c,\text{robot})
\]
across training and deployment embodiments.

### 3.5 Connection to lectures 1–2
- Lecture 1 supplies path/dynamics theory.
- Lecture 2 provides chunked action parameterization.
- VLA supplies multimodal conditioning and representation transfer.

---

## Chapter 4 — Lecture 4 (arXiv:2602.05993): Diamond Maps and Efficient Reward Alignment

### 4.1 Alignment target
Given a base model \(p_{\text{base}}\), reward alignment targets a tilted law:
\[
\pi(x) \propto p_{\text{base}}(x)\exp(\beta r(x)).
\]
This is the optimizer of
\[
\max_q \;\mathbb E_q[r(x)] - \tau D_{\mathrm{KL}}(q\|p_{\text{base}}).
\]

### 4.2 Distillation view
Diamond-style methods learn efficient stochastic maps that approximate expensive aligned dynamics:
\[
\min_\phi D_{\mathrm{KL}}(p_{\text{teacher}}\|p_\phi).
\]
The teacher may come from computationally heavy guided transitions; the student map amortizes alignment at inference time.

### 4.3 Markov interpretation
Let teacher transitions define \(K_t^{\text{teach}}\). Learned map \(K_t^\phi\) aims to preserve reward-relevant marginals and transition structure. This reframes alignment as **operator approximation under reward constraints**.

### 4.4 Why it is important
- Makes alignment practical under strict inference budgets.
- Preserves stochasticity needed for exploration/search.
- Fits directly into generative-policy deployment where repeated fast sampling is required.

---

## Chapter 5 — Lecture 5 (arXiv:2604.09784): Discrete Flow Maps

### 5.1 Discrete state geometry
For token outputs, model states lie on simplex \(\Delta_K\), not \(\mathbb R^d\). Therefore Euclidean regression losses can be geometrically inconsistent.

### 5.2 Proper objectives
Use simplex-compatible divergences:
\[
\mathcal L(\theta) = \mathbb E\left[D_{\mathrm{KL}}\big(q(\cdot\mid x)\|\mu_\theta(\cdot\mid x)\big)\right]
\]
or cross-entropy analogues.

### 5.3 Continuous-time discrete dynamics
Discrete flow constructions connect to CTMC generators
\[
[L_t f](x) = \sum_{y\in S} Q_t(y\mid x)(f(y)-f(x)),
\]
with Kolmogorov forward equation over probability vectors.

### 5.4 Relevance to VLA and decision tokens
If action generation is tokenized (or latent discrete), discrete flow maps provide principled non-autoregressive decoding and alignment-ready objectives.

### 5.5 Key theoretical insight
Valid discrete generative training requires respecting probability-simplex structure at both objective and transition levels; otherwise optimization may improve surrogate loss while degrading true token-law quality.

---

## Chapter 6 — Lecture 6 (arXiv:2412.06264): Generator Matching (GM)

### 6.1 Generator-first principle
Instead of directly fitting likelihoods, GM learns infinitesimal generators \(L_t\) whose induced marginals follow designed paths:
\[
\partial_t p_t = L_t^* p_t.
\]

### 6.2 Unified model class
GM shows flow, diffusion, and jump/CTMC dynamics are instances of a common generator formalism. In Euclidean spaces, an admissible generator can be decomposed into flow + diffusion + jump components.

### 6.3 Conditional Generator Matching (CGM)
Direct marginal objectives are intractable. CGM trains conditional objects and leverages conditional-to-marginal equivalence:
\[
\mathcal L_{\text{CGM}} = \mathbb E_{t,z,x\sim p_t(\cdot\mid z)} D\big(\dot f_z(t), L_t f_z(x)\big).
\]
This gives scalable optimization while preserving the target gradient of the original GM objective.

### 6.4 Markov superposition
If \(L_t^{(i)}\) are valid generators, then
\[
L_t^{\text{sup}} = \sum_i w_i L_t^{(i)},\quad w_i\ge0,\ \sum_iw_i=1
\]
is also a valid generator (under regularity conditions). This allows hybrid models combining deterministic flow, stochastic diffusion, and discrete jump/CTMC effects.

### 6.5 Policy-learning interpretation
For robot control, set state space to action trajectories and condition on observations/language. Then GM directly yields a principled family of generative policies with clear dynamics semantics and compositionality.

---

## Chapter 7 — Cross-Lecture Synthesis: A Single Theoretical Pipeline

### 7.1 Unified problem statement
Goal: learn
\[
\pi(a_{t:t+H-1}\mid o_t,c)
\]
that is expressive, stable, and alignable.

### 7.2 Mapping papers to roles
- **2410.20587:** path-space and flow/diffusion mathematical foundations.
- **2304.13705:** robust chunked action parameterization for manipulation.
- **2406.09246:** multimodal conditioning via VLA token architecture.
- **2602.05993:** practical reward alignment via efficient stochastic maps.
- **2604.09784:** discrete-space map theory for token/simplex outputs.
- **2412.06264:** generator-level unification and superposition framework.

### 7.3 End-to-end training concept
1. Build multimodal representation (VLA backbone).
2. Choose action-space generator family (flow/diffusion/chunk/discrete/hybrid).
3. Train with imitation via conditional objectives (ACT/CGM-style).
4. Apply reward alignment with regularized target tilt and map distillation.
5. Deploy with calibrated stochasticity and chunked receding-horizon control.

### 7.4 Theoretical guarantees sought in practice
- path consistency between base and learned marginals,
- stable conditional-to-marginal transfer,
- reward improvement under bounded divergence from base policy,
- computational efficiency preserving stochastic expressivity.

---

## Chapter 8 — Suggested Study Sequence and Exercises

### 8.1 Reading order
1. Lecture 1 (probability path foundations)
2. Lecture 6 (GM formalism)
3. Lecture 2 + Lecture 3 (policy architecture choices)
4. Lecture 4 + Lecture 5 (alignment and discrete extensions)

### 8.2 Theory exercises
1. Derive continuity and Fokker–Planck equations from generator definitions.
2. Show Gibbs tilt solves KL-regularized reward maximization.
3. Verify superposition validity for a flow+diffusion pair.
4. Compare Euclidean MSE vs cross-entropy on simplex outputs in discrete generation.

### 8.3 System-design exercises (OpenArm/VLA context)
1. Define action chunk horizon \(H\) and analyze stability-reactivity tradeoff.
2. Design a hybrid generator with continuous arm control and discrete skill modes.
3. Propose an alignment pipeline using teacher transitions + distilled stochastic maps.

---

## Closing Summary

These lecture notes reframe the six-paper set as one coherent theory stack:
- probability paths define the target evolution,
- generators define admissible dynamics,
- chunking and VLA define policy parameterization,
- reward alignment retargets terminal/path laws,
- discrete flow maps and superposition complete the mixed-state-space story.

This viewpoint gives a principled route from theory to deployable robot policies while keeping all major modeling decisions interpretable at the level of stochastic-process dynamics.
