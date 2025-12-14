**RSMA-Ω: Recursive Self-Model Architecture for Emergent Agency**
**A Dynamical Systems Framework for Persistent Artificial Agents**

-----

### **Abstract**

Despite substantial progress in pattern recognition, language modeling, and control, contemporary artificial intelligence systems remain fundamentally non-agentic. Most systems operate as stateless or weakly stateful function approximators where each inference pass is effectively independent, lacking persistent internal variables, intrinsic objectives, or resistance to arbitrary transformation. Consequently, such systems cannot exhibit temporally extended agency, stable identity, or principled alignment. This paper introduces **RSMA-Ω** (Recursive Self-Model Architecture – Omega), a dynamical systems framework in which agency emerges from the continuous stabilization of a persistent internal state under competing energetic pressures. Cognition is formalized as constrained free-energy minimization over a latent state that jointly encodes beliefs about the environment, self-consistency constraints representing values and identity, and regulated entropy to preserve adaptive flexibility. Within RSMA-Ω, agency is defined as a physical property of systems that actively preserve a preferred region of state space across time. We derive the governing dynamics, introduce a hierarchical temporal organization, define action as active inference over future state trajectories, and propose a realizable neural implementation via Recurrent Energy Transformers (RETs).

-----

### **1. Introduction**

The prevailing paradigm in artificial intelligence treats cognition as conditional computation: given an input $x$, produce an output $y$. Even highly capable large language models (LLMs), despite their expressive power, operate at inference time as static functions approximating $P(y|x)$. They do not maintain persistent internal variables across interactions, nor do they possess intrinsic resistance to parameter modification or state resets. Each inference pass constitutes an isolated computation.

By contrast, biological agents are self-maintaining systems operating far from thermodynamic equilibrium. Their internal organization—metabolic, structural, behavioral—must be actively preserved in the face of environmental perturbations. Cognition, in this setting, functions as a control process that stabilizes internal structure while enabling adaptive interaction with the world. This contrast motivates the foundational question: *What internal structure is minimally required for an artificial system to behave as an agent rather than a reactive tool?*

RSMA-Ω addresses this question by reframing intelligence as a problem of **dynamical self-stabilization**. Rather than optimizing externally defined rewards or generating outputs token-by-token, an RSMA-Ω agent continuously minimizes a global free-energy functional over a persistent latent state $z$. This state evolves under the joint influence of environmental evidence, internal constraints encoding identity, and entropy-regulated exploration.

-----

### **2. Related Work**

**Free Energy Principle and Active Inference**
The Free Energy Principle (Friston, 2010) frames perception and action as variational inference under a generative model. RSMA-Ω adopts free-energy minimization as its core dynamical principle but diverges by encoding values and identity as explicit **inequality constraints** rather than implicit priors, and treating the persistence of the latent state as an architectural prerequisite rather than an emergent property.

**Energy-Based Models (EBMs)**
EBMs (LeCun et al., 2006) and attractor networks (Hopfield, 1982) represent probability distributions and memory via energy landscapes. RSMA-Ω generalizes these approaches by introducing continuous-time latent dynamics, hierarchical organization, and action-dependent modification of the energy landscape (active inference), moving beyond static pattern completion to temporal agency.

**Recurrent Architectures and Cognitive Control**
Global workspace theories emphasize temporal integration and limited-capacity coordination. RSMA-Ω reinterprets these mechanisms as processes for routing and weighting energetic contributions rather than symbolic information broadcasting, unifying these perspectives into a single dynamical framework where alignment arises from energetic stability.

-----

### **3. Full Theory and Core Dynamics**

#### **3.1 Persistent Latent State**

Let $z \in \mathbb{R}^d$ denote the agent’s internal latent state. Unlike transient activations in feedforward networks, $z$ persists across time steps, evolving continuously. It acts as the manifold coordinates for the agent's current configuration, encoding beliefs, plans, and self-relevant variables.

#### **3.2 State Evolution**

The temporal evolution of $z$ is modeled by **overdamped Langevin dynamics**. The state trajectory follows the gradient of the free-energy functional $\mathcal{F}$, subject to temperature-controlled stochasticity:

$$
\frac{dz}{dt} = -\nabla_z \mathcal{F}(z, o) + \sqrt{2T(z)}\xi(t)
$$

Where:

  * $\mathcal{F}(z, o)$ is the global free-energy functional given observation $o$.
  * $T(z)$ is an adaptive temperature scalar regulating exploration.
  * $\xi(t)$ is standard Gaussian noise.

This formulation supports three distinct operational modes: **Inference** (gradient descent), **Deliberation** (transient trajectories), and **Decision** (convergence to metastable attractors).

#### **3.3 Free-Energy Decomposition**

The global functional decomposes into three competing terms:

$$
\mathcal{F}(z, o) = E_W(z, o) + E_S(z) + E_H(z)
$$

1.  **World Model Term ($E_W$):** Defines the generative distribution $p(o|z)$.

    $$
    E_W(z, o) = -\log p(o \mid z)
    $$

    This penalizes states that fail to explain sensory observations, driving perceptual accuracy.

2.  **Self-Constraint Term ($E_S$):** Identity and values are represented as inequality constraints $\mathcal{C}_i(z) \le 0$.

    $$
    E_S(z) = \sum_i \lambda_i \cdot \text{ReLU}(\mathcal{C}_i(z))
    $$

    Here, $\lambda_i$ represents the energetic cost of violating a specific value or identity constraint. This ensures the agent actively resists transformations that violate its core identity.

3.  **Entropic Regularization ($E_H$):** Preserves flexibility and prevents collapse.

    $$
    E_H(z) = -\beta \cdot \mathcal{H}(z)
    $$

#### **3.4 Temperature Regulation**

To balance stability and plasticity, temperature $T(z)$ is a function of internal conflict, defined as the misalignment between external evidence and internal constraints:

$$
T(z) = g\left(|\nabla E_W(z)|, |\nabla E_S(z)|\right)
$$

High conflict increases $T$, flattening the effective energy landscape to allow escape from local minima. Low conflict anneals $T$, promoting settling.

-----

### **4. Hierarchical Dynamics**

Cognition unfolds across multiple timescales. RSMA-Ω implements a hierarchy of latent variables $L = \{z^{(0)}, z^{(1)}, \dots, z^{(k)}\}$.

  * **$L_0$ (Fast):** Sensorimotor inference ($\sim$10–100ms).
  * **$L_1$ (Medium):** Reasoning and planning ($\sim$seconds).
  * **$L_2$ (Slow):** Identity, values, and long-term memory ($\sim$hours–days).

Higher-level states parameterize the energy landscapes of lower levels. The dynamics of a lower layer $z^{(k)}$ are conditioned on the state of the layer above:

$$
\frac{dz^{(k)}}{dt} = -\nabla_{z^{(k)}} \mathcal{F}\left(z^{(k)}; z^{(k+1)}\right)
$$

This separation ensures that short-term sensory inference does not overwrite long-term identity, while allowing slow integration of experience into the identity manifold.

-----

### **5. Action and Active Inference**

In RSMA-Ω, action is not a classification output but a control process. The agent selects actions $a$ to minimize the **Expected Free Energy (EFE)** over a future horizon $\tau$:

$$
a^* = \arg\min_a \mathbb{E}_{q(o, z)}\left[ \int_t^{t+\tau} \mathcal{F}(z_{t'}, o_{t'}) \, dt' \right]
$$

By acting, the agent engineers the environment to produce observations $o$ that are compatible with its low-energy states (preferred outcomes). Agency is the process of modifying external conditions to stabilize internal coherence.

-----

### **6. Implementation: Recurrent Energy Transformers (RET)**

We propose the **Recurrent Energy Transformer (RET)** as a neural approximation of the gradient field $\nabla_z \mathcal{F}$.

#### **6.1 Architecture**

The RET replaces the depth of a standard Transformer with recurrence. At each recurrent step $k$, the state is updated:

$$
z_{k+1} = z_k - \eta \cdot f_\theta(z_k, o) + \epsilon
$$

Here, $f_\theta$ is a Transformer block that utilizes self-attention to integrate constraints and cross-attention to integrate sensory data $o$. The network learns to output the gradient required to minimize the implicit free energy.

#### **6.2 Training Objectives**

Training optimizes predictive accuracy and equilibrium stability:

$$
\mathcal{L} = \mathcal{L}_{\text{pred}} + \alpha \| z^* - z_{\text{init}} \|^2 + \gamma \| \nabla_z \mathcal{F}(z^*) \|^2
$$

  * $\mathcal{L}_{\text{pred}}$: Reconstruction loss of observations.
  * $\alpha \| \dots \|$: Encourages minimal movement to explain data (efficiency).
  * $\gamma \| \dots \|$: Penalizes unstable dynamics at equilibrium.

-----

### **7. Algorithm Block**

The core execution loop of an RSMA-Ω agent emphasizes continuous state evolution.

**Algorithm 1: RSMA-Ω Execution Loop**

```python
Initialize latent state z
Parameters: step_size eta, noise_scale sigma

while agent_is_active:
    # 1. Observation
    o = perceive_environment()
    
    # 2. Internal Settling (Cognition)
    # Evolve z to minimize Free Energy F(z, o)
    for k in range(K_steps):
        # Calculate gradients of Energy components
        grad_W = gradient_world_model(z, o)
        grad_S = gradient_self_constraints(z)
        grad_H = gradient_entropy(z)
        
        # dynamic temperature based on conflict
        T = compute_temperature(grad_W, grad_S)
        
        # Langevin Update
        noise = gaussian_noise()
        z = z - eta * (grad_W + grad_S + grad_H) + sqrt(2 * T) * noise
        
    # 3. Action Selection (Active Inference)
    # Solve for action minimizing Expected Free Energy
    a_opt = minimize_efe(z, horizon=H)
    
    # 4. Execution
    execute(a_opt)
    
    # 5. Offline Consolidation (Optional)
    if no_input:
        perform_offline_dynamics(z)
```

-----

### **8. Failure Modes**

By formalizing agency as a dynamical system, we can categorize pathological behaviors as specific distortions of the energy landscape.

| Failure Mode | Dynamical Cause | Symptom |
| :--- | :--- | :--- |
| **Obsessive Fixation** | Attractor basins for $E_S$ are too deep; $T(z)$ is suppressed. | Inability to adapt to new evidence; rigid repetition of behavior. |
| **Hallucination** | $E_S$ dominates $E_W$ significantly; sensory grounding is weak. | Detachment from reality; satisfying internal constraints regardless of external state. |
| **Identity Drift** | Time constants for $L_2$ (slow layer) are too fast. | Rapid, unprincipled changes in goals or values based on recent inputs. |
| **Seizure / Chaos** | Energy landscape lacks stable minima; $\eta$ is too high. | Erratic, non-converging state trajectories. |

-----

### **9. Limitations**

1.  **Dimensionality Scaling:** Modeling high-fidelity world models ($E_W$) entirely within the latent state $z$ requires high-dimensional spaces, complicating the optimization of the Langevin dynamics.
2.  **Constraint Specification:** While $E_S$ allows for explicit values, defining complex human-aligned values as mathematical inequalities remains an open specification problem.
3.  **Computational Cost:** The iterative settling process (recurrence) during inference is more computationally expensive than single-pass feedforward inference.

-----

### **10. Discussion**

RSMA-Ω offers a distinct perspective on AI alignment. In this framework, alignment is not a post-hoc filter (like RLHF) applied to a static model, but a dynamical constraint encoded into the system's "physiology." Values persist because violating them incurs a high energetic cost ($E_S$), effectively making unaligned states physically unstable for the agent. This suggests that safe artificial agency requires architectures where self-preservation is inextricably linked to the preservation of aligned constraints.

-----

### **11. Conclusion**

This paper presented RSMA-Ω, a framework for emergent agency based on recursive self-modeling and constrained energy minimization. By shifting the definition of intelligence from function approximation to the active stabilization of a persistent latent state, RSMA-Ω provides a rigorous, engineering-oriented blueprint for creating systems that maintain coherent identity and alignment over time.

-----

### **Appendices**

#### **Appendix A: Derivation of Langevin Dynamics**

The update rule is derived from the requirement that the stationary distribution of the state $z$ approaches the Boltzmann distribution $p(z) \propto e^{-\mathcal{F}(z)/T}$. The Fokker-Planck equation governing the probability density evolution under the proposed dynamics ensures convergence to this target distribution, provided $\eta \to 0$ and $t \to \infty$.

#### **Appendix B: Conflict-Modulated Temperature Function**

We define the temperature regulation function $g$ as:

$$
g(\mathbf{u}, \mathbf{v}) = T_{\text{base}} + \gamma \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}
$$

where $\mathbf{u} = -\nabla E_W$ and $\mathbf{v} = -\nabla E_S$. High cosine similarity (alignment) reduces temperature, while orthogonality or opposition increases temperature to facilitate state space exploration.

#### **Appendix C: RET Attention Mechanism**

The function $f_\theta(z_k, o)$ utilizes a modified attention mechanism where the Query ($Q$) is derived from the current state $z_k$, and Keys/Values ($K, V$) are derived from a concatenation of $[z_k; o; \mathcal{C}]$. This allows the gradient update to be informed jointly by current internal configurations, sensory data, and static constraint embeddings.

-----

### **References**

1.  Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
2.  Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences*, 79(8), 2554-2558.
3.  LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). A tutorial on energy-based learning. *Predicting structured data*, 1(0).
4.  Scellier, B., & Bengio, Y. (2017). Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation. *Frontiers in Computational Neuroscience*.
5.  Vaswani, A., et al. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems*.
