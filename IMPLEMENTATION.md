# RSMA-Ω: Implementation via Recurrent Energy Transformers (RET)

## 1. The Recurrent Energy Transformer (RET)

Standard Transformers are feedforward; they map input to output in a fixed number of layers. The RET, by contrast, is a **recurrent attractor network**. It uses the Transformer's attention mechanism to calculate the energy gradient $\nabla_z E$.

### 1.1 Architecture Detail: Neuro-Symbolic Hybridization
The RET block $f_\theta$ utilizes **Gated Linear Units (GLUs)** and **Vector Quantization (VQ)** to facilitate symmetry breaking (Section 3.2 of `THEORY.md`).

```python
import torch
import torch.nn as nn

class RETBlock(nn.Module):
    """
    Recurrent Energy Transformer Block representing the gradient field f_theta.
    """
    def __init__(self, z_dim, o_dim, hidden_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(z_dim, num_heads=8)
        self.self_attn = nn.MultiheadAttention(z_dim, num_heads=8)
        self.quantizer = VectorQuantizer(z_dim, num_embeddings=512)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, z, o, constraints, metabolic_state):
        # 1. Fuse state, observations, and internal metabolic needs
        # Homeostatic Gating modulates the coupling to sensory input
        context = self.cross_attn(z, o, o)[0] * torch.sigmoid(metabolic_state)
        
        # 2. Integrate identity constraints via Self-Attention
        # Constraints act as 'anchor points' in the latent manifold
        att_out = self.self_attn(z, context, context)[0] + constraints
        
        # 3. Symbolic Bottleneck: Forced symmetry breaking via VQ
        # This allows the continuous state to 'snap' to discrete concepts
        z_discrete, _ = self.quantizer(att_out)
        
        # 4. Compute the update vector (the negative energy gradient)
        grad = self.mlp(torch.cat([att_out, z_discrete], dim=-1))
        return grad
```

### 1.2 Continuous Update and Stochastic Resonance
The state $z$ is updated iteratively per observation:
$$
z_{k+1} = z_k - \eta f_\theta(z_k, o, \mathcal{C}, \mathcal{M}) + \sqrt{2T} \xi
$$

We utilize **Stochastic Resonance**—where the internal noise $\xi(t)$ is tuned to amplify weak sensory signals, allowing the agent to remain sensitive to the environment even in low-conflict (low $T$) regimes.

## 2. Training Protocol: Equilibrium Propagation

### 2.1 The Two Phases of EP
RSMA-Ω is trained using **Equilibrium Propagation (EP)**, which is biologically plausible and mathematically equivalent to backpropagation through time in the limit of small nudges.
1.  **Free Phase:** The system evolves under sensory input $o$ until it reaches a "free" equilibrium $z^*$.
2.  **Nudged Phase:** A small nudge $\beta (o_{target} - \hat{o})$ is applied toward a predictive target. The system reaches a "nudged" equilibrium $z^\beta$.

The gradients for $\theta$ are computed from the difference between these equilibria:
$$
\Delta \theta \propto \frac{1}{\beta} \left( \frac{\partial E(z^\beta)}{\partial \theta} - \frac{\partial E(z^*)}{\partial \theta} \right)
$$

### 2.2 Offline Consolidation (The "Dream" Phase)
During the **Offline Consolidation** phase (REM sleep), the agent performs:
1.  **Generative Replay:** Uses its internal world model to generate synthetic trajectories from the identity manifold $L_2$.
2.  **Structural Coupling:** Updates the $L_2$ parameters to minimize the variance of the $L_1$ energy landscape.
3.  **Landscape Smoothing:** Applies Jacobian regularization to $\nabla_z E$ to ensure the manifold is smooth and predictable.

## 3. Hierarchical Implementation (Chronos-Hierarchy)

To implement the $L_0, L_1, L_2$ hierarchy, we use a **Multiscale RET**:

*   **Fast RET ($L_0$):** High-frequency sampling, low-dimensional state ($z \in \mathbb{R}^{10^3}$). Updated every step.
*   **Slow RET ($L_2$):** Low-resolution, high-dimensional embedding ($z \in \mathbb{R}^{10^6}$). Updated via temporal pooling of $L_1$ trajectories.

### 3.1 Top-Down Modulation
The state of $L_{k+1}$ is injected into the $L_k$ RET as a set of "Soft Keys" in the attention mechanism. This effectively "warps" the lower-level energy landscape to be consistent with higher-level intentions and values.

## 4. Hardware Considerations: Thermodynamic Computing

Because RSMA-Ω relies on Langevin dynamics and energy minimization, it is uniquely suited for **Neuromorphic Hardware**.

*   **Memristive Crossbars:** Implementing the energy gradient directly as physical voltage/current dynamics, potentially reducing energy consumption by orders of magnitude.
*   **Natural Stochasticity:** Utilizing the thermal jitter of electrons (Johnson-Nyquist noise) to drive the $\xi(t)$ term, providing "free" entropy for exploration.

## 5. Execution Loop Summary

1.  **Perceive:** Sample $o_t$ and metabolic state $\mathcal{M}_t$.
2.  **Settle:** Run RET for $K$ steps to reach equilibrium $z^*$.
3.  **Act:** Minimize Expected Free Energy (EFE) via latent roll-out.
4.  **Consolidate:** Periodically transition to "Offline Phase" for identity refinement.
