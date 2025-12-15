# Structural Instability of Neural ODEs Near Bifurcations and Invariant Boundaries

**Status:** Manuscript in Preparation (Dec 2025)  
**Author:** Nasser Mohammed  
**Topic:** Scientific Machine Learning (SciML), Dynamical Systems, Bifurcation Theory

## Abstract
Neural Ordinary Differential Equations (Neural ODEs) have emerged as a powerful tool for system identification, but their topological fidelity near critical transitions remains unclear. In this study, we investigate the learnability of **Saddle-Node Bifurcations** (specifically the "ghost" region) in a reaction-diffusion system.

We demonstrate that this inability to learn complex vector fields is not due to a lack of data, but rather an inherent limit of neural ODEs. We analyze a system near a saddle-node bifurcation, characterizing 3 distinct fixed point topologies (no fixed point => non-hyperbolic fixed point => sink and saddle). We then train a neural ODE for each parameter range, and for increasing amounts of data. This also highlights how vector fields with less fixed points can sometimes require more data, due to ghost dynamics. Finally, this also highlights a difficulty in trusting neural ODEs, as the loss converges to 0 in all of these scenarios, however, the true fixed point topology is far off. Furthermore, the neural ODE struggles with keeping its trajectories bounded in the invariant region, even if the vector field looks accurate.


## 1. The Problem: Ghost Dynamics and Global Structure
Biological systems often operate near tipping points characterized by **Saddle-Node** bifurcations. These regions exhibit "ghost" dynamics where the trajectory slows down algebraically ($O(t^{-1})$). Furthermore, many of these systems are only biologically relevent for certain invariant regions. A common example is when the differential equations model
concentrations of a species, which is naturally bounded in $[0,1]^n$. Neural ODEs of this type will only be trained on data that the user can obtain (i.e. in $[0,1]^n$), however, we will see that neural ODEs can fail to maintain this invariant region, invalidating the dynamics they produce.

We found that standard Neural ODEs struggle with simultaneously capturing the local dynamics of fixed points near bifurcations, as well as the invariant region boundaries. We hypothesize that this is due to the relationship between invariant regions and their isolated structure with respect to a global vector field. Furthermore, the dynamics near saddle-node bifurcations are highly sensitive, making approximation a challenge. 

### The System
We analyze a custom derived Predator-Prey model which has the following features:
- Forward invariant on the unit square $[0,1]^2$.
- Saddle-node bifurcation as the parameter $\alpha$ is shifted through the 3 distinct regions: $ \alpha < 20, \alpha = 20, \alpha > 20$. The existence of fixed points near the point $(0, 0.2)$ dissapears for $\alpha < 20$.
- Persistent limit cycle near $(0,0.2)$. The unstable spiral that generate this fixed point is centered around $(0.34, 0.406)$.
- A persistent boundary saddle node at $(0,1)$, which helps maintain the limit cycle.

  The model is given by this coupled reaction-diffusion system:

 $$
\frac{du}{dt} = \frac{10u(1-u)}{1+e^{-12(v-0.4)}} - \frac{4u}{1+e^{-12(0.55-v)}} \\
\frac{dv}{dt} = \alpha v(1-v)(v-0.4) + 0.8(1-v) - 3.6 uv
$$





## 2. Key Results: The Hierarchy of Failure

We subjected the Neural ODE to a "Steel-Man" suite of training regimes. All failed to capture the topology, revealing fundamental architectural limitations.

### A. Artificial (Standard MSE)
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
