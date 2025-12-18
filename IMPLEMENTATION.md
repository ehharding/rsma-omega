# RSMA-Ω: Implementation via Recurrent Energy Transformers (RET)

This document details the neural architecture and training protocols required to realize the RSMA-Ω dynamical system.

## 1. The Recurrent Energy Transformer (RET)

Standard Transformers are feedforward; they map input to output in a fixed number of layers. The RET, by contrast, is a **recurrent attractor network**. It uses the Transformer's attention mechanism to calculate the energy gradient $\nabla_z \mathcal{F}$.

### 1.1 Architecture Detail: Neuro-Symbolic Hybridization
The RET block $f_\theta$ utilizes **Gated Linear Units (GLUs)** and **Vector Quantization (VQ)** to facilitate symmetry breaking (Section 3.2 of THEORY.md).

```python
class RETBlock(nn.Module):
    def forward(self, z, o, constraints, metabolic_state):
        # 1. Fuse state, observations, and internal metabolic needs
        # The metabolic_state modulates the attention weights (Homeostatic Gating)
        context = self.cross_attn(z, o, metabolic_state)
        
        # 2. Integrate identity constraints via Self-Attention
        # Anchors represent high-level 'Virtues' or 'Rules'
        att_out = self.self_attn(z, context, constraints)
        
        # 3. Symbolic Bottleneck: Forced symmetry breaking via VQ
        # This allows the continuous state to 'snap' to discrete concepts when needed
        z_discrete = self.quantizer(att_out)
        
        # 4. Output the update vector (the negative gradient)
        grad = self.mlp(torch.cat([att_out, z_discrete], dim=-1))
        return grad
```

### 1.2 Continuous Update and Stochastic Resonance
The state $z$ is updated iteratively:
$z_{k+1} = z_k - \eta f_\theta(z_k, o, \mathcal{C}, \mathcal{M}) + \sqrt{2T} \cdot \text{noise}$

We utilize **Stochastic Resonance**—where the internal noise $\xi(t)$ is tuned to amplify weak sensory signals, allowing the agent to remain sensitive to the environment even in low-conflict (low $T$) regimes.

## 2. Training Protocol: Equilibrium Propagation and Offline Consolidation

### 2.1 The Two Phases of EP
1.  **Free Phase:** The system evolves under sensory input $o$ until it reaches a "free" equilibrium $z^*$.
2.  **Nudged Phase:** A small nudge is applied toward the target state (e.g., a prediction of the next observation). The system reaches a "nudged" equilibrium $z^\beta$.

### 2.2 Offline Consolidation (The "Dream" Phase)
True AGI cannot learn everything in real-time. RSMA-Ω implements an **Offline Consolidation** phase (analogous to REM sleep). During this phase:
1.  **Generative Replay:** The agent uses its internal world model to generate synthetic observations.
2.  **Structural Coupling:** The $L_2$ layer (Slow Identity) is updated to minimize the long-term average free energy of the replayed trajectories.
3.  **Landscape Smoothing:** The energy landscape is regularized to prevent over-fitting to transient sensory noise.

## 3. Hierarchical Implementation (Chronos-Hierarchy)

To implement the $L_0, L_1, L_2$ hierarchy, we use a **Multiscale RET**:

*   **Fast RET ($L_0$):** High-frequency sampling, low-dimensional state. Updated every step ($\sim$10ms).
*   **Slow RET ($L_2$):** Low-resolution, high-dimensional embedding. Updated via temporal pooling of $L_1$ trajectories ($\sim$hours).

### 3.1 Top-Down Modulation
The state of $L_{k+1}$ is injected into the $L_k$ RET as a set of "Soft Keys" in the attention mechanism. This effectively "warps" the lower-level energy landscape to be consistent with higher-level intentions.

## 4. Hardware Considerations: Thermodynamic Computing

Because RSMA-Ω relies on Langevin dynamics (noise), it is uniquely suited for **Neuromorphic Hardware** and **Thermodynamic Computing**.

*   **Memristive Crossbars:** Implementing the energy gradient directly as physical voltage/current dynamics, potentially reducing energy consumption by orders of magnitude compared to GPUs.
*   **Natural Stochasticity:** Utilizing the thermal jitter of electrons to drive the $\xi(t)$ term, drastically reducing the energy cost of random number generation.

## 5. Summary of the Execution Loop

1.  **Perceive:** Sample $o_t$ and internal metabolic state $\mathcal{M}_t$.
2.  **Settle:** Run RET for $K$ steps to find the equilibrium $z^*$.
3.  **Act:** Minimize Expected Free Energy (EFE) via a short-horizon roll-out in the latent space.
4.  **Learn/Consolidate:** Periodically transition to "Offline Phase" for structural identity updates.
