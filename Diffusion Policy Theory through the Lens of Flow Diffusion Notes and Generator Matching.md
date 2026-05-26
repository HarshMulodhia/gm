# Diffusion Policy Theory through the Lens of Flow/Diffusion Notes and Generator Matching

## Overview and Goal

This document explains the theory behind Diffusion Policy ("Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", arXiv:2303.04137) using the notation and conceptual framework of the MIT flow/diffusion lecture notes and the Generator Matching (GM) companion notes and paper.[^1][^2][^3][^4][^5]

The aim is to:
- Express Diffusion Policy as a specific Markov generative model on action space, with a probability path, generator, and Kolmogorov Forward Equation (KFE).
- Align the DDPM-style training and sampling equations in the paper with the lecture notes’ probability paths, score functions, and SDEs.
- Interpret the policy as solving a conditional generator matching problem for action sequences conditioned on observations.
- Connect receding-horizon control and time-series diffusion Transformer to this probabilistic/generator-level picture.

Throughout, actions live in an appropriate Euclidean space (single step or action sequence), so all constructions are in the SRd setting of the notes and GM.[^3][^1]

***

## 1. Background: Probability Paths and Generators

### 1.1 Probability paths in lecture notes and GM

The lecture notes formulate generative modeling as sampling from a data distribution pdata over a state space S ⊂ Rd (images, videos, trajectories, etc.), starting from a simple prior pinit and designing a probability path pt that interpolates between them.[^2] A conditional probability path pt(·|z) interpolates between a simple base and a point mass at each data point z; marginalizing z gives a marginal path pt with p0 = pinit, p1 = pdata.[^2]

GM adopts exactly this picture, in measure notation ptdx, and emphasizes that pt can be realized by many different Markov processes whose generators Lt solve the KFE. On Rd, Theorem 1 in GM states that any generator can be decomposed into a flow (velocity field), diffusion (SDE noise), and jump component.[^1][^3]

### 1.2 Flow and diffusion models in the lecture notes

In the continuous setting, the lecture notes show:
- Flow models: ODE dXt = ut(Xt, t) dt with generator Lt f(x) = ∇f(x)·ut(x, t), and continuity equation ∂t p = −∇·(u p).[^2]
- Diffusion models: SDE dXt = ut(Xt, t) dt + σt(Xt) dWt, with generator
  \[
  L_t f(x) = \nabla f(x)\cdot u_t(x) + \tfrac{1}{2} \mathrm{Tr}(\sigma_t(x)\sigma_t(x)^\top \nabla^2 f(x)),
  \]
  and Fokker–Planck equation ∂t p = −∇·(u p) + 1/2 ∇2 : (σσ⊤ p).[^2]

Score-based diffusion models are obtained by parameterizing either the drift (probability-flow ODE) or the score ∇x log pt(x), and using the Fokker–Planck/KFE to connect them to the chosen probability path.[^2]

### 1.3 Generator Matching summary

GM generalizes this to “Markov generative models”: pick a probability path pt connecting a simple prior to pdata, then find a generator Lt whose Markov process Xt has marginals pt, and train Lt via a Conditional Generator Matching (CGM) loss built from Bregman divergences.[^3][^1]

Key GM objects:[^1][^3]
- Conditional path pt(·|z) and marginal path pt.
- Conditional generator Lzt for pt(·|z), and marginal generator Lt obtained via posterior averaging (Proposition 1).
- Linear parameterization Lt f = ⟨Kf, Ft⟩ with Ft parameterized by a neural network.
- GM loss LGM and CGM loss LCGM with Bregman D; Proposition 2 shows LGM and LCGM have identical gradients.

Diffusion models and flow matching appear as special cases where the parameterized object Ft is the vector field (flow) or score/diffusion coefficient, and the Bregman divergence is an MSE or KL-like term.[^3][^1]

***

## 2. Diffusion Policy as a Conditional Diffusion Model on Action Space

### 2.1 Action space and state space choice

Diffusion Policy treats the robot policy as a generative model over action sequences conditioned on observations. Let:[^5]
- S = RA be the action space for a single action (e.g., joint targets or Cartesian position commands), or
- S = RAdT be the space of action sequences of length Tp (stacked along time).

In either case S is Euclidean, so the GM and lecture-notes derivations for Rd apply.[^1][^2][^3]

A data point in this setting is an action sequence A0 ∈ S paired with an observation context Ot (e.g., recent image frames and robot state). The dataset contains pairs (Ot, A0) extracted from demonstrations.[^5]

### 2.2 Diffusion model on actions

Standard DDPMs define a forward Gaussian noising process and a reverse-time denoising process parameterized by a noise prediction network εθ. In Diffusion Policy, the variable x used in DDPM becomes the action (or action sequence) A, and the reverse denoising iteration is:[^5]
\[
\mathbf{x}^{k-1} = \alpha(\mathbf{x}^k - \gamma\,\epsilon_\theta(\mathbf{x}^k, k) + \mathcal{N}(0, \sigma^2 I)), \quad k = K,\dots,1,\tag{1}
\]
where α, γ, σ depend on the step k via a noise schedule.[^5]

Interpreted as a noisy gradient step, this is:
\[
\mathbf{x}' = \mathbf{x} - \gamma \nabla E(\mathbf{x}),\tag{2}
\]
with εθ(x, k) approximating ∇E(x).[^5]

To adapt this to actions and conditioning on observations Ot, Diffusion Policy uses:[^5]
\[
\mathbf{A}^{k-1}_t = \alpha\bigl(\mathbf{A}^k_t - \gamma\,\epsilon_\theta(\mathbf{O}_t, \mathbf{A}^k_t, k) + \mathcal{N}(0, \sigma^2 I)\bigr),\tag{4}
\]
which defines a Markov chain in action space whose stationary distribution approximates the conditional action distribution p(A0|Ot).

### 2.3 Conditional distribution p(A|O)

The key design choice is to model the conditional p(A_t|O_t) instead of a joint distribution over future observations and actions (as in planning-oriented diffusion models), so the denoising process acts only on actions, with observations entering purely as conditioning.[^5]

In the language of the lecture notes and GM:
- For each fixed conditioning context O, one has a data distribution pdata,A(·|O) over actions A.
- The diffusion model defines a conditional probability path pτ(A|O) over τ ∈ [^1], interpolating from a simple Gaussian prior N(0, I) at τ = 0 to pdata,A(·|O) at τ = 1.
- The reverse denoising dynamics (4) approximate a discretization of a reverse-time SDE or probability-flow ODE whose marginals follow pτ(·|O).[^2][^3][^5]

Thus, for each observation O, Diffusion Policy instantiates a conditional generative model in the same class as the image DDPMs studied in the lecture notes, but in action space.[^2][^5]

***

## 3. Forward and Reverse Diffusion Processes in Action Space

### 3.1 Conceptual forward (noising) process

As in DDPMs, training is organized by sampling a clean action A0 from the dataset, picking a random diffusion step k, and adding Gaussian noise εk with variance determined by the noise schedule.[^5] Although the paper does not explicitly write the forward SDE or discrete-time formula q(Ak|A0), it uses the standard DDPM/score-matching idea: A0 is corrupted to Ak = A0 + εk and the model is trained to predict εk from (Ak, k, Ot).[^2][^5]

Conceptually, this defines a forward probability path pτ(A|O) for τ corresponding to k/K, where p0(·|O) is a simple Gaussian and p1(·|O) ≈ pdata,A(·|O).[^2][^5]

### 3.2 Reverse (denoising) dynamics as a Markov generator

In the continuous-time perspective of the lecture notes and GM, a DDPM reverse process can be cast as either:
- A reverse-time SDE (score-based SDE) with drift involving the score ∇A log pτ(A|O), or
- A probability-flow ODE whose drift equals the score times an appropriate diffusion coefficient.[^3][^2]

The discrete-time update (4) is a time-inhomogeneous Markov kernel (over k) in action space, which can be associated with a time-dependent generator Lt acting on test functions f(A):[^1][^3]
\[
L_t f(A) = \nabla f(A) \cdot u_t(A, O) + \tfrac{1}{2} \mathrm{Tr}(\Sigma_t(A,O) \nabla^2 f(A)),
\]
for some drift u_t and covariance Σt that depend on the noise schedule and the noise-prediction network εθ.[^2][^3][^1][^5]

At the level of marginals, Lt and pτ(A|O) are related by the KFE (adjoint Fokker–Planck equation):
\[
\partial_\tau p_\tau(A|O) = L_\tau^* p_\tau(A|O),
\]
where Lτ* is the adjoint generator.[^3][^1][^2]

This matches the lecture notes’ description of score-based diffusion models: the learned score or drift is chosen so that the Fokker–Planck equation integrates the Gaussian prior into the data distribution along the chosen probability path.[^2]

### 3.3 Noise schedule and square-cosine schedule

The paper uses a “square cosine” noise schedule from improved DDPM work (iDDPM), which specifies how the noise variance and step-size change across k. In the continuous-time view, this corresponds to a particular time parameterization τ(k) and variance σ2(τ), which jointly determine the diffusion coefficient and drift in the SDE/ODE representation.[^3][^2][^5]

***

## 4. Training Objective as Conditional Score Matching / Generator Matching

### 4.1 DDPM noise-prediction loss

The base DDPM loss is:
\[
\mathcal{L} = \operatorname{MSE}(\boldsymbol{\epsilon}^k, \epsilon_\theta(\mathbf{x}^0 + \boldsymbol{\epsilon}^k, k)),\tag{3}
\]
where x0 is a clean data sample and εk is Gaussian noise for step k.[^5]

In Diffusion Policy, this becomes the conditional loss:
\[
\mathcal{L} = \operatorname{MSE}(\boldsymbol{\epsilon}^k, \epsilon_\theta(\mathbf{O}_t, \mathbf{A}_t^0 + \boldsymbol{\epsilon}^k, k)),\tag{5}
\]
for action sequences A0 conditioned on observation Ot.[^5]

This is exactly the denoising score-matching loss: εθ(O, A0 + ε, k) is trained to predict the “true noise” that relates Ak to A0; under the Gaussian forward noising, this is equivalent to learning the score ∇A log pτ(A|O) up to scaling.[^2][^3][^5]

### 4.2 Energy-based and score-based interpretation

The paper explicitly connects Diffusion Policy to energy-based models (EBMs) and their scores. An implicit policy is written as:[^5]
\[
p_\theta(\mathbf{a}|\mathbf{o}) = \frac{e^{-E_\theta(\mathbf{o}, \mathbf{a})}}{Z(\mathbf{o}, \theta)},\tag{6}
\]
with energy Eθ and intractable partition function Z(·, θ). Training such EBMs requires negative sampling (InfoNCE-style objectives) and is unstable.[^5]

Diffusion Policy instead learns the score:
\[
\nabla_\mathbf{a} \log p(\mathbf{a}|\mathbf{o}) = - \nabla_\mathbf{a} E_\theta(\mathbf{a}, \mathbf{o}) - \underbrace{\nabla_\mathbf{a} \log Z(\mathbf{o}, \theta)}_{=0} \approx -\epsilon_\theta(\mathbf{a}, \mathbf{o}),\tag{8}
\]
so εθ approximates the negative score field without ever evaluating Z(·, θ), avoiding the instability from negative sampling.[^2][^3][^5]

In GM terms, εθ parameterizes the generator via a linear map Kf (e.g., gradient) and Ft = Ft(O, ·) representing the score field; the MSE loss is a Bregman divergence on that parameter space, so the conditional loss (5) is a conditional GM loss.[^1][^3]

### 4.3 GM/flow-matching analogy

The lecture notes prove that a conditional mean-squared loss on conditional vector fields (flows or scores) is equivalent, up to constants, to a loss on the intractable marginal fields (Theorem 12 in the notes for flow matching, analogous to GM’s Proposition 2 for generator matching). Diffusion Policy’s conditional MSE loss (5) inherits this property:[^1][^2]
- It trains the conditional score field (or an equivalent parameterization of the generator) for each (O, A0) pair.
- By linearity of the GM framework, this implicitly trains the marginal generator conditional on O.[^3][^1]

Thus, Diffusion Policy is a conditional generator-matching model where:
- State space is action space S.
- Condition is observation O.
- Ft(O, A, k) is the time-dependent score/energy gradient over actions.
- The MSE is the Bregman divergence used in CGM.[^1][^3][^5]

***

## 5. Receding-Horizon Control and Action-Sequence Representation

### 5.1 Action sequences as high-dimensional outputs

Diffusion Policy predicts an entire sequence of future actions, not just a single action. Let:[^5]
- To: observation horizon (number of past observations used as context),
- Tp: action prediction horizon (length of action sequence output),
- Ta: execution horizon (prefix of the predicted sequence executed before re-planning).

At time t, the policy:
1. Takes the latest To steps of observation history Ot.
2. Samples a Tp-step action sequence A0,t ∈ RAdTp from the conditional diffusion model p(A0,t|Ot) via the reverse denoising iterations (4).
3. Executes only the first Ta actions of the generated sequence on the robot.
4. Shifts the horizon and repeats, re-planning as new observations arrive.[^5]

This is receding-horizon model predictive control (MPC) implemented with a generative model as the action sequence generator.[^5]

### 5.2 Theoretical interpretation in probability-path language

From the perspective of the lecture notes and GM:
- The model defines, for each Ot, a conditional path pτ(A|Ot) over the finite-horizon action sequence A ∈ RAdTp.
- Sampling A0,t by reverse diffusion is sampling from p1(A|Ot) ≈ pdata,A(·|Ot), the terminal distribution of that path.
- Executing a prefix Ta of this sequence and then re-sampling at the next time step corresponds to running an open-loop generative model in a closed-loop fashion via receding horizon.[^2][^3][^5]

Because the path is defined on action space rather than state space, the dynamical consistency with physical dynamics is learned implicitly from demonstrations. This is an “action diffusion policy” version of the generative modeling pipeline rather than explicit trajectory diffusion in state space.[^5]

### 5.3 Benefits: temporal consistency, robustness, latency

The paper highlights several advantages of this action-sequence formulation:[^5]
- Temporal action consistency: multimodal action sequences are sampled as coherent trajectories, avoiding the “mode-switching jitter” of independently sampled one-step actions.
- Robustness to idle actions: idle segments in demonstrations (e.g., waiting during pouring) are encoded within sequences, mitigating overfitting to “stay still forever” modes.
- Latency robustness: since the sequence predicts into the future, the controller can tolerate several control timesteps of computation and sensing latency while executing buffered actions.[^5]

These properties are natural consequences of using a high-dimensional action path pt over sequences and sampling from it via a Markov generator: the correlation structure over time is captured directly in the model’s state (action sequence).[^3][^1][^2][^5]

***

## 6. Visual Conditioning and Architecture

### 6.1 Conditional diffusion vs joint diffusion

Janner et al. and related planning work often model a joint distribution over future states and actions p(X, A|O0) and then condition on desired goals.[^5] Diffusion Policy instead models p(A|O) only.[^5]

In probabilistic terms:
- The lecture notes’ conditional/marginal path construction applies with “condition” z replaced by the observation O.
- For each O, there is a conditional path pt(A|O), and the marginal over O is the data distribution joint p(O, A) given by demonstrations. The generative model is the conditional family {p(A|O)}.[^2]

This design halves the dimension of the modeled variable (no future states) and yields faster inference, which is critical in robotics.[^5]

### 6.2 Visual encoders as part of condition

The visual encoder maps the raw images into a latent Ot that serves as the conditioning variable in εθ(Ot, A, k). The paper uses:[^5]
- ResNet-18 encoders per camera view, with spatial softmax pooling (instead of global average pooling) to preserve spatial information.
- GroupNorm instead of BatchNorm for stability with EMA, as standard in DDPMs.[^5]

In the generator-matching picture, this encoder implements a function φ: images → conditioning space Z, and the conditional generator is Lτ,z acting on actions given latent z = φ(O).[^1][^3][^5]

### 6.3 Time-series Diffusion Transformer vs CNN backbone

The noise-prediction network εθ can be implemented as:[^5]
- A 1D temporal CNN backbone that takes the noisy action sequence tokens and modulates them with observation features via FiLM, or
- A time-series Diffusion Transformer (adapted from minGPT), which treats timesteps as tokens and uses transformer decoder blocks with diffusion-step embeddings and observation embeddings.

From a theoretical perspective, both choices are parameterizations of the same conditional generator Lt (via εθ). The Transformer has more flexible inductive bias for high-frequency, rapid action changes (e.g., velocity control), while the CNN tends to favor smoother, low-frequency sequences.[^5]

***

## 7. Relationship to Flow Matching, Score-Based SDEs, and GM

### 7.1 Flow matching vs diffusion policy

Flow matching in the lecture notes constructs a deterministic ODE with drift u⋆t(x) such that its marginals follow a prescribed probability path pt. The target vector field u⋆t is obtained analytically for a Gaussian CondOT path and trained via a conditional MSE loss against a neural approximation uθ.[^2]

Diffusion Policy instead uses a stochastic reversetime diffusion process and parameterizes the score (or an equivalent drift) rather than the velocity of a deterministic flow. In terms of GM:[^3][^2][^5]
- Flow matching = GM with pure flow generator Lt f = ∇f·ut and MSE on ut.[^1][^3]
- Diffusion Policy = GM with a diffusion-type generator whose parameter is the score field (equivalently gradient of energy), trained via MSE on εθ, which is proportional to the score.[^3][^1][^5]

Both share the conditional-vs-marginal training equivalence (conditional flow/score matching), but Diffusion Policy leverages stochastic sampling and the robustness of score-based diffusion.[^2][^1][^3][^5]

### 7.2 Score-based SDEs and probability-flow ODEs

The lecture notes review Song et al.’s score-based generative modeling via SDEs: one specifies a forward SDE that noises data to a simple prior, and a reverse-time SDE or probability-flow ODE built from the score ∇x log pt(x) to generate samples.[^2]

Diffusion Policy uses exactly this architecture but with actions in place of images and observations as conditioning:[^2][^5]
- Forward Gaussian noising defines a probability path pτ(A|O).
- εθ(O, A, k) learns the conditional score.
- The reverse Markov process (4) approximates Langevin dynamics on the learned energy / score field.

In GM language, this is a diffusion generator Lt with state-dependent drift equal to the score times a schedule-dependent factor, and stationary distribution p1(A|O) ≈ pdata,A(·|O).[^1][^3]

### 7.3 GM view: conditional generator on action space

Putting everything together in GM notation:[^1][^3][^5]
- State space: S = RAdTp (action sequences).
- Data distribution: pdata(O, A0).
- For each O, define a Gaussian-like probability path pτ(A|O) between N(0, I) and pdata,A(·|O).
- Conditional generator Lzt ≡ Lt,O on S realizes pτ(A|O) via an SDE or discrete-time Markov chain.
- Linear parameterization: Ft(O, A, k) = εθ(O, A, k) (or a scaled version thereof), with a fixed Kf mapping test functions to gradients.
- Bregman divergence: squared Euclidean (MSE) on Ft.
- Conditional GM loss: LCGM = Eτ,O,A0,ε MSE(ε, εθ(O, A0 + ε, τ)).
- By Proposition 2, this optimizes the marginal GM loss on Ft, i.e., the generator for pτ(A|O).

Thus Diffusion Policy is a concrete instance of Generator Matching applied to conditional action distributions with a diffusion-style generator, implemented via DDPM/DDIM in discrete time.[^3][^1][^5]

***

## 8. Practical Implications for Robotic VLA Systems

### 8.1 Multimodality and high-dimensional actions

Diffusion Policy’s theoretical formulation directly explains its empirical properties in manipulation benchmarks:[^5]
- It handles multimodal actions because the score-based diffusion can represent arbitrary normalizable distributions over action sequences, without specifying a fixed number of mixture components.
- It scales to high-dimensional actions (e.g., sequences of 6-DoF poses) by leveraging the same scaling properties that make image diffusion work well in very high dimensions.

In GM terms, this is just the statement that the parameterized generator Ft lives in a very rich function class (deep networks), so the induced Markov process can realize complex paths pt on high-dimensional S.[^1][^3]

### 8.2 Stability vs implicit EBMs

The EBM comparison in the paper is also clarified: implicit policies that parameterize energy Eθ require negative sampling to approximate Z(·, θ) and are known to be unstable. Diffusion Policy fits within the GM/score-matching paradigm where only the score is learned and the partition function is never needed.[^5]

The lecture notes make the same point when comparing denoising score matching to likelihood-based training for diffusion models, and GM formalizes this via Bregman divergences on generator parameters without likelihood ratios.[^3][^1][^2]

### 8.3 Integration into VLA stacks

For VLA models in robotics, Diffusion Policy can be viewed as the “action head” that turns a visual-language (and possibly state) representation into a conditional Markov generator on actions.
- Vision/backbone: encodes images (and optionally language) into Ot.
- Diffusion Policy: conditional generator in action space, trained via score matching.
- Control: receding-horizon MPC that queries this generator to obtain action sequences, executed on the robot.

From the lecture notes + GM perspective, this is an end-to-end learned realization of a conditional probability path over actions, with closed-loop control realized by repeatedly sampling from the terminal marginal.[^1][^2][^3][^5]

***

## 9. Summary

Using the notation and concepts from the flow/diffusion lecture notes and Generator Matching, Diffusion Policy can be seen as:
- A conditional score-based diffusion model on action sequences.
- A specific conditional generator-matching instance where the generator is of diffusion type and parameterized by the action-score field.
- Trained by a conditional denoising score-matching loss, which is a Bregman divergence in the GM sense.
- Deployed as a receding-horizon controller that samples temporally consistent, multimodal action sequences conditioned on visual observations.

This unified view clarifies why Diffusion Policy is stable, expressive, and well-suited to high-dimensional, multimodal manipulation tasks—and how it relates mathematically to flow matching, score-based SDEs, and the broader Markov generative modeling framework.

---

## References

1. [Generator-Matching-Theory-Companion-Notes-Aligned-with-Flow-Diffusion-Lecture-Notation.md](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/88949578/cc3f901c-40fb-49b8-a880-1a8396c18125/Generator-Matching-Theory-Companion-Notes-Aligned-with-Flow-Diffusion-Lecture-Notation.md?AWSAccessKeyId=ASIA2F3EMEYEV35GYJIW&Signature=01RANLzaKnqj%2FuK2jCvdo4hwNY8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEK7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDze2803PliffpI%2BxNxAdK7QNji4fCQSPzSHCF4N1rqnQIhAPBiw64cZkwGFjIBr8pvFIL3Opww4L8v9849pn0HZVuqKvMECHcQARoMNjk5NzUzMzA5NzA1IgwssqcXN5l7CuLClk0q0AR%2FZKehE%2FmFdVz7OfUL9Idb%2FkBzGupkY2cxo8KvFj1BnNU6yPmVtnsDkDijVeQzKJv8Ep%2FRUXu3Pj%2B7Nnq3LQQsafBGkaNa5SC590mfULocLlpVwUAdR%2FhaWSSyYf9rmkFBKm0KdQAgosA4HxI52r1dqrBr5zBCOT4zaNicPzYDCUSpQWax2GoM%2B5fwrXpUo56siKlQuhxa749nFJoCMn1C22xrJP2Ng%2B12%2BGIgd4DIOY8J8DAwXElS0IqZutSvpoDNcuBpUwKWk8VLqBZfAn6OfvMZZ6PTSxBhVNbSMdwVacdGv18DyN28GMJi0jCIiCCStUK7%2BIMQQEt1n6pgCLWQrQ7HZFcu0%2F128vjCiWrZTn00Ou1Y0AvSdL7jyf6m7%2FzoXeusTMxRAiwqw5ExXRMLfbRHIlbGu4P7yt%2FYYrOJcyd4UB3UY4ovNfgvUGTpFeSq3V4oF2M1kL3TV4rBMvAniNsJgj6ecLik40FgIZfoYEsuwhX4y3hu0lwN%2BxwK4XHPf9YWZxwK3TcjfzshqusYev7E0p51MuL1beoLRI8M6%2BXgPbvxgsTXbhGtfWUkXOTSFqXOEytcNxucM275iq9vYnnWQDzCLC8JRZ7mUC4eyLFIfB1y6nbYIZ0qHdWW0lFAFWiwnHhhJxC58JJczj6G3oKZoY3H353CElsZzUwbAaxwYTZAODbyTTuXeTxFG3FeO%2FukSTcHU0KzMBYGul40fBLdpwUGGCOCLjY%2FlUPAgzps%2FvVaQ1uXjFQcDOUMAUEISPMUnP3ymU0pXqSQC%2FO3MMLq1NAGOpcBcmGbZ8fGP8ZjjnunCuIGPX%2BXn0%2B7kPLjvlcCZcP4UA%2FVyuy1Xrw5lLX2IB%2FfPxeyW9t4ssP55eTYI9xkZNVTvmoj6fvh%2F6M%2B06FOKjiC5Oqdj30uuQYC%2Fr5zfL%2BmrEQkmPPQArAVF4HBIsC9gJpgeq6ypIKR3jOQo9lAhlNlrBys2mu%2FD%2Ff5QZ%2BULQuQxlyoT6mGFaJLSQ%3D%3D&Expires=1779778325) - These notes are a companion to the paper Generator Matching A General Framework For Markov Generativ...

2. [lecture_notes.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/88949578/88d1e229-1ffb-4181-a72c-623d0ee92a65/lecture_notes.pdf?AWSAccessKeyId=ASIA2F3EMEYEV35GYJIW&Signature=pWVXgfrqNFqPMBp0Q8iaF6DKWiw%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEK7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDze2803PliffpI%2BxNxAdK7QNji4fCQSPzSHCF4N1rqnQIhAPBiw64cZkwGFjIBr8pvFIL3Opww4L8v9849pn0HZVuqKvMECHcQARoMNjk5NzUzMzA5NzA1IgwssqcXN5l7CuLClk0q0AR%2FZKehE%2FmFdVz7OfUL9Idb%2FkBzGupkY2cxo8KvFj1BnNU6yPmVtnsDkDijVeQzKJv8Ep%2FRUXu3Pj%2B7Nnq3LQQsafBGkaNa5SC590mfULocLlpVwUAdR%2FhaWSSyYf9rmkFBKm0KdQAgosA4HxI52r1dqrBr5zBCOT4zaNicPzYDCUSpQWax2GoM%2B5fwrXpUo56siKlQuhxa749nFJoCMn1C22xrJP2Ng%2B12%2BGIgd4DIOY8J8DAwXElS0IqZutSvpoDNcuBpUwKWk8VLqBZfAn6OfvMZZ6PTSxBhVNbSMdwVacdGv18DyN28GMJi0jCIiCCStUK7%2BIMQQEt1n6pgCLWQrQ7HZFcu0%2F128vjCiWrZTn00Ou1Y0AvSdL7jyf6m7%2FzoXeusTMxRAiwqw5ExXRMLfbRHIlbGu4P7yt%2FYYrOJcyd4UB3UY4ovNfgvUGTpFeSq3V4oF2M1kL3TV4rBMvAniNsJgj6ecLik40FgIZfoYEsuwhX4y3hu0lwN%2BxwK4XHPf9YWZxwK3TcjfzshqusYev7E0p51MuL1beoLRI8M6%2BXgPbvxgsTXbhGtfWUkXOTSFqXOEytcNxucM275iq9vYnnWQDzCLC8JRZ7mUC4eyLFIfB1y6nbYIZ0qHdWW0lFAFWiwnHhhJxC58JJczj6G3oKZoY3H353CElsZzUwbAaxwYTZAODbyTTuXeTxFG3FeO%2FukSTcHU0KzMBYGul40fBLdpwUGGCOCLjY%2FlUPAgzps%2FvVaQ1uXjFQcDOUMAUEISPMUnP3ymU0pXqSQC%2FO3MMLq1NAGOpcBcmGbZ8fGP8ZjjnunCuIGPX%2BXn0%2B7kPLjvlcCZcP4UA%2FVyuy1Xrw5lLX2IB%2FfPxeyW9t4ssP55eTYI9xkZNVTvmoj6fvh%2F6M%2B06FOKjiC5Oqdj30uuQYC%2Fr5zfL%2BmrEQkmPPQArAVF4HBIsC9gJpgeq6ypIKR3jOQo9lAhlNlrBys2mu%2FD%2Ff5QZ%2BULQuQxlyoT6mGFaJLSQ%3D%3D&Expires=1779778325) - page-1 MITClass6.S184GenerativeAIWithStochasticDierentialEquations,2026AnIntroductiontoFlowMatchinga...

3. [GM.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/88949578/cbf1a3d7-eacd-44d4-a6db-a631d8342886/GM.pdf?AWSAccessKeyId=ASIA2F3EMEYEV35GYJIW&Signature=gUMemCSzS7w4Suy0MjfhhYd9hQU%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEK7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDze2803PliffpI%2BxNxAdK7QNji4fCQSPzSHCF4N1rqnQIhAPBiw64cZkwGFjIBr8pvFIL3Opww4L8v9849pn0HZVuqKvMECHcQARoMNjk5NzUzMzA5NzA1IgwssqcXN5l7CuLClk0q0AR%2FZKehE%2FmFdVz7OfUL9Idb%2FkBzGupkY2cxo8KvFj1BnNU6yPmVtnsDkDijVeQzKJv8Ep%2FRUXu3Pj%2B7Nnq3LQQsafBGkaNa5SC590mfULocLlpVwUAdR%2FhaWSSyYf9rmkFBKm0KdQAgosA4HxI52r1dqrBr5zBCOT4zaNicPzYDCUSpQWax2GoM%2B5fwrXpUo56siKlQuhxa749nFJoCMn1C22xrJP2Ng%2B12%2BGIgd4DIOY8J8DAwXElS0IqZutSvpoDNcuBpUwKWk8VLqBZfAn6OfvMZZ6PTSxBhVNbSMdwVacdGv18DyN28GMJi0jCIiCCStUK7%2BIMQQEt1n6pgCLWQrQ7HZFcu0%2F128vjCiWrZTn00Ou1Y0AvSdL7jyf6m7%2FzoXeusTMxRAiwqw5ExXRMLfbRHIlbGu4P7yt%2FYYrOJcyd4UB3UY4ovNfgvUGTpFeSq3V4oF2M1kL3TV4rBMvAniNsJgj6ecLik40FgIZfoYEsuwhX4y3hu0lwN%2BxwK4XHPf9YWZxwK3TcjfzshqusYev7E0p51MuL1beoLRI8M6%2BXgPbvxgsTXbhGtfWUkXOTSFqXOEytcNxucM275iq9vYnnWQDzCLC8JRZ7mUC4eyLFIfB1y6nbYIZ0qHdWW0lFAFWiwnHhhJxC58JJczj6G3oKZoY3H353CElsZzUwbAaxwYTZAODbyTTuXeTxFG3FeO%2FukSTcHU0KzMBYGul40fBLdpwUGGCOCLjY%2FlUPAgzps%2FvVaQ1uXjFQcDOUMAUEISPMUnP3ymU0pXqSQC%2FO3MMLq1NAGOpcBcmGbZ8fGP8ZjjnunCuIGPX%2BXn0%2B7kPLjvlcCZcP4UA%2FVyuy1Xrw5lLX2IB%2FfPxeyW9t4ssP55eTYI9xkZNVTvmoj6fvh%2F6M%2B06FOKjiC5Oqdj30uuQYC%2Fr5zfL%2BmrEQkmPPQArAVF4HBIsC9gJpgeq6ypIKR3jOQo9lAhlNlrBys2mu%2FD%2Ff5QZ%2BULQuQxlyoT6mGFaJLSQ%3D%3D&Expires=1779778325) - page-2

4. [[2303.04137] Diffusion Policy - ar5iv - arXiv](https://ar5iv.labs.arxiv.org/html/2303.04137) - This paper introduces Diffusion Policy, a new way of generating robot behavior by representing a rob...

5. [Visuomotor Policy Learning via Action Diffusion - arXiv](https://arxiv.org/html/2303.04137) - In this work, we seek to address this challenge by introducing a new form of robot visuomotor policy...

