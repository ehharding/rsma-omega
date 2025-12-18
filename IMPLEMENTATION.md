# RSMA-Ω: Implementation via Recurrent Energy Transformers (RET)

This document details the neural architecture and training protocols required to realize the RSMA-Ω dynamical system.

## 1. The Recurrent Energy Transformer (RET)

Standard Transformers are feedforward; they map input to output in a fixed number of layers. The RET, by contrast, is a **recurrent attractor network**. It uses the Transformer's attention mechanism to calculate the energy gradient $\nabla_z \mathcal{F}$.

### 1.1 Architecture Detail
The RET block $f_\theta$ takes the current state $z_k$ and observation $o$ as input.

```python
class RETBlock(nn.Module):
    def forward(self, z, o, constraints):
        # 1. Fuse state and observations via Cross-Attention
        context = self.cross_attn(z, o)
        
        # 2. Integrate identity constraints via Self-Attention
        # The constraints act as "static" anchors in the attention space
        att_out = self.self_attn(z, context, constraints)
        
        # 3. Output the update vector (the negative gradient)
        grad = self.mlp(att_out)
        return grad
```

### 1.2 Continuous Update
The state $z$ is updated iteratively:
$z_{k+1} = z_k - \eta f_\theta(z_k, o, \mathcal{C}) + \epsilon$

This is equivalent to finding the fixed point of the energy landscape defined by $\theta$.

## 2. Training Protocol: Equilibrium Propagation

Backpropagation through time (BPTT) is computationally expensive and biologically implausible for long-lived agents. Instead, we use **Equilibrium Propagation (EP)** (Scellier & Bengio, 2017).

### 2.1 The Two Phases of EP
1.  **Free Phase:** The system evolves under sensory input $o$ until it reaches a "free" equilibrium $z^*$.
2.  **Nudged Phase:** A small nudge is applied toward the target state (e.g., a prediction of the next observation). The system reaches a "nudged" equilibrium $z^\beta$.

The gradient for the parameters $\theta$ is then:
$$
\nabla_\theta \mathcal{L} \propto \frac{1}{\beta} \left( \frac{\partial \mathcal{F}(z^\beta, \theta)}{\partial \theta} - \frac{\partial \mathcal{F}(z^*, \theta)}{\partial \theta} \right)
$$

This allows the network to learn the energy landscape directly from its own equilibrium dynamics.

## 3. Hierarchical Implementation (Chronos-Hierarchy)

To implement the $L_0, L_1, L_2$ hierarchy, we use a **Multiscale RET**:

*   **Fast RET ($L_0$):** High-resolution, low-dimensional state. Updated every step.
*   **Slow RET ($L_2$):** Low-resolution, high-dimensional embedding. Updated via temporal pooling of $L_1$ trajectories.

### 3.1 Top-Down Modulation
The state of $L_{k+1}$ is injected into the $L_k$ RET as a set of "Soft Keys" in the attention mechanism. This effectively "warps" the lower-level energy landscape to be consistent with higher-level intentions.

## 4. Hardware Considerations: Stochastic Computing

Because RSMA-Ω relies on Langevin dynamics (noise), it is uniquely suited for **Stochastic Computing** and **Neuromorphic Hardware**.

*   **Thermodynamic Computing:** Utilizing natural thermal noise to drive the $\xi(t)$ term in the Langevin update.
*   **Memristive Crossbars:** Implementing the energy gradient directly as physical voltage/current dynamics, potentially reducing energy consumption by orders of magnitude compared to GPUs.

## 5. Summary of the Execution Loop

1.  **Perceive:** Sample $o_t$.
2.  **Settle:** Run RET for $K$ steps to find $z^*$.
3.  **Act:** Minimize Expected Free Energy via a short-horizon roll-out in the latent space.
4.  **Learn:** Periodically apply Equilibrium Propagation during "sleep" or offline consolidation.
