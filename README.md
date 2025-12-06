# Artificial Hyperbolization: Structural Instability of Neural ODEs

**Status:** Manuscript in Preparation (Dec 2025)  
**Author:** Nasser Mohammed  
**Links:** [Project CV Entry](#) | [Preprint (Coming Soon)](#)

## Abstract
Neural Ordinary Differential Equations (Neural ODEs) are powerful tools for system identification, but their structural stability near degenerate fixed points remains an open question. In this study, we investigate a reaction-diffusion system exhibiting a **Saddle-Node bifurcation** (non-hyperbolic fixed point). 

We demonstrate that standard Neural ODEs trained with MSE loss exhibit **spectral bias**, effectively "hyperbolizing" the ghost point. This creates artificial exponential stability where polynomial decay should exist, distorting the geometry and period of the associated limit cycle. We propose and implement an **Inverse-Velocity Weighted Loss** to correct this gradient starvation.

## Research Questions
This project investigates three core hypotheses regarding the learnability of degenerate topologies:

1.  **The "Artificial Hyperbolization" Mechanism:** Does the spectral bias of standard neural networks compel the model to approximate locally flat (cubic) dynamics with linear (exponential) dynamics, thereby shifting the eigenvalues from zero to non-zero real parts?
2.  **The "Gradient Starvation" Cause:** Is the failure to learn non-hyperbolic dynamics driven by "Magnitude Bias," where the vanishing vector field ($\|\dot{x}\| \to 0$) causes the loss contribution to become negligible?
3.  **The "Weighted" Solution:** Can structural stability be recovered by re-weighting the optimization landscape inversely to the phase-space velocity?

## Preliminary Results

### 1. The "Ghost" Failure Mode
Standard training fails to capture the critical slowing down near the saddle-node ghost at $(0, 0.2)$. The neural network approximates the cubic "flat" region with a linear slope, causing trajectories to move too fast through the bottleneck.

![Ghost Race Animation](Media/neural_ode_race_first.gif)
*Figure 1: Comparison of Ground Truth (Blue) vs. Standard Neural ODE (Red). Note how the Neural ODE fails to "hang" at the ghost point (Middle Plot).*

### 2. Artificial Hyperbolization
The topological classification of the fixed point is altered by the model.
- **Ground Truth:** Non-Hyperbolic ($\lambda = 0.0$)
- **Neural ODE:** Hyperbolic Sink ($\lambda \approx -0.05$)

![Phase Portrait](Media/System%20comparison.png)
*Figure 2: Vector field comparison showing the "smoothing" of the limit cycle geometry.*

## Methodology: Inverse-Velocity Weighted Loss
To address the magnitude bias where vanishing gradients ($\dot{x} \approx 0$) cause the optimizer to ignore slow dynamics, we introduce a state-dependent importance weighting:

```math
\mathcal{L} = \frac{1}{N} \sum_{t} \frac{1}{\| \dot{x}_{true}(t) \|^2 + \epsilon} \| x_{pred}(t) - x_{true}(t) \|^2
