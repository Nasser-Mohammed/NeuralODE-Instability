# On the Unlearnability of Saddle-Node Ghosts: Structural Instability in Neural ODEs

**Status:** Manuscript in Preparation (Dec 2025)  
**Author:** Nasser Mohammed  
**Topic:** Scientific Machine Learning (SciML), Dynamical Systems, Bifurcation Theory

## Abstract
Neural Ordinary Differential Equations (Neural ODEs) have emerged as a powerful tool for system identification, but their topological fidelity near critical transitions remains unproven. In this study, we investigate the learnability of **Saddle-Node Bifurcations** (specifically the "ghost" region) in a reaction-diffusion system.

We demonstrate that standard neural architectures suffer from **"Artificial Hyperbolization,"** consistently regularizing degenerate non-hyperbolic equilibria ($\lambda=0$) into stiff hyperbolic sinks ($\lambda \ll 0$). Through a comprehensive ablation study involving Inverse-Velocity Weighting, Sobolev (Vector Field) Training, and L-BFGS optimization, we show that this failure is robust to training strategy. We conclude that the inductive bias of piecewise-linear networks creates a representational barrier, preventing the capture of quadratic tangencies ($f(x) \sim x^2$) without destroying global limit cycles.

---

## 1. The Problem: Spectral Bias vs. Critical Slowing Down
Biological systems often operate near tipping points characterized by **Saddle-Node on Invariant Circle (SNIC)** bifurcations. These regions exhibit "ghost" dynamics where the trajectory slows down algebraically ($O(t^{-1})$).

We hypothesize that standard Neural ODEs, driven by **Spectral Bias** and **Lipschitz constraints**, cannot represent the "flat" cubic contact of a ghost point, effectively smoothing out the bottleneck.

### The System
We analyze a custom Predator-Prey model derived to exhibit a local saddle-node ghost at $z^* = (0, 0.2)$.
- **Ground Truth Eigenvalues:** $\lambda_1 = 0.00$ (Center), $\lambda_2 = -3.11$ (Stable).
- **Topology:** Non-Hyperbolic.

---

## 2. Key Results: The Hierarchy of Failure

We subjected the Neural ODE to a "Steel-Man" suite of training regimes. All failed to capture the topology, revealing fundamental architectural limitations.

### A. Artificial Hyperbolization (Standard MSE)
Standard training ignores the vanishing gradients at the ghost point ("Gradient Starvation").
- **Result:** The model learns a "Black Hole" sink with $\lambda \approx -14.5$.
- **Visual:** The model trajectory (Red) flies past the ghost point where the Truth (Blue) hangs.

![Ghost Race](Media/neural_ode_race_first.gif)
*Figure 1: Trajectory comparison. Note the failure of the Neural ODE (Red) to capture the critical slowing down timescale.*

### B. Bifurcation Collapse (Sobolev & L-BFGS)
Even when trained with **Second-Order Optimization (L-BFGS)** or **Explicit Vector Field Matching (Sobolev Loss)**, the model fails to maintain the saddle-node degeneracy.
- **Result:** The ghost is converted into a weak sink ($\lambda \approx -3.01$).
- **Consequence:** This spurious attractor "sucks in" the global limit cycle, destroying the oscillation entirely.

![Bifurcation Collapse](Media/phase_portrait_comparison.png)
*Figure 2: Topological catastrophe. The Neural ODE (Right) regularizes the ghost into a sink, collapsing the limit cycle that exists in the Ground Truth (Left).*

---

## 3. Mechanism of Failure

### The Magnitude Gap
The failure is driven by the inability of the network to fit the "flat" quadratic bottom of the vector field magnitude. The network approximates the parabolic $|f(x)| \sim x^2$ with a V-shaped linear approximation $|f(x)| \sim |x|$, resulting in a non-zero derivative at the minimum.

![Magnitude Proof](Media/magnitude_proof.png)
*Figure 3: Vector field magnitude along the center manifold. The Neural ODE (Red) cannot capture the zero-tangency of the Truth (Blue), enforcing a minimum velocity > 0.*

### Summary of Experiments

| Training Method | Learned Eigenvalues ($\lambda_1, \lambda_2$) | Topological Outcome |
| :--- | :--- | :--- |
| **Ground Truth** | **$0.00, -3.11$** | **Saddle-Node Ghost** |
| Standard MSE (Adam) | $-14.51, -5.16$ | Stiff Sink (Drift) |
| Inverse-Weighted | $-2.21, +0.01$ | Unstable Saddle |
| L-BFGS (2nd Order) | $-4.57, -1.03$ | Spurious Attractor |
| Sobolev (Vector) | $-3.01, -0.09$ | **Bifurcation Collapse** |

---

## 4. Conclusion
Our results suggest that **standard MLP architectures are mathematically incapable** of representing SNIC bifurcations without introducing structural instability. The piecewise-linear inductive bias forces the model to trade off between capturing the slow manifold (ghost) and the fast manifold (limit cycle); it cannot satisfy both simultaneously.

Future work must move beyond loss function engineering and explore **Rational Networks** or **Bifurcation-Informed Priors** to enforce polynomial degeneracy constraints.

---

## Project Structure
* `system_definitions.py`: PyTorch implementation of the Predator-Prey system.
* `train_ablation.py`: Script to run the full suite (MSE, Weighted, Sobolev, L-BFGS).
* `visualize.py`: Generates the phase portraits and animations.

## Usage
To reproduce the failure analysis:
```bash
python train_ablation.py --mode lbfgs --visualize
