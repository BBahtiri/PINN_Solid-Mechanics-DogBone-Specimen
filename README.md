# PINN for Hyperelastic Solid Mechanics

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains a TensorFlow 2 implementation of a **Physics-Informed Neural Network (PINN)** to solve a 2D solid mechanics problem. The project simulates a quasi-static tensile test of a hyperelastic dog-bone specimen, demonstrating how deep learning can be used to solve complex non-linear partial differential equations (PDEs).

The core of this work is a PINN that directly solves the strong form of the equilibrium equation, with a hybrid optimization scheme that combines ADAM and L-BFGS for efficient and accurate training.

<br>

![PINN Architecture](/pinn.png)
*Figure 1: The architecture of the Physics-Informed Neural Network.*

---

## 🏗️ Problem Formulation

The PINN is trained to find a displacement field $\mathbf{u}(\mathbf{X})$ that satisfies the governing equations of solid mechanics for a given domain $\Omega$ with boundary $\partial \Omega$.

### Governing Equation

The simulation solves the static equilibrium equation in its strong form, where body forces are neglected:

$$\nabla \cdot \mathbf{P}(\mathbf{X}) = \mathbf{0} \quad \forall \mathbf{X} \in \Omega$$

Here, $\mathbf{P}$ is the first Piola-Kirchhoff (PK1) stress tensor.

### Hyperelastic Constitutive Law

The material is modeled as hyperelastic. The relationship between deformation and stress is derived from the strain energy density function. The key kinematic quantities and the constitutive law are:

1.  **Deformation Gradient**: $\mathbf{F} = \mathbf{I} + \nabla \mathbf{u}$
2.  **PK1 Stress**: The code implements a compressible Neo-Hookean-like model, where the PK1 stress tensor $\mathbf{P}$ is given by:
    $$\mathbf{P} = \mu \mathbf{F} + (\lambda \ln(J) - \mu) \mathbf{F}^{-T}$$
    where $J = \det(\mathbf{F})$, and $\mu$ and $\lambda$ are Lamé's first and second parameters.

### Boundary Conditions

The dog-bone specimen is subjected to the following boundary conditions:
* **Prescribed Displacement**: A tensile displacement, $u_x = u_{\text{applied}}$, is applied incrementally to the right-hand face of the specimen.
* **Fixed/Symmetry Condition**: The left-hand face is fixed in the x-direction ($u_x = 0$).
* **Traction-Free**: All other boundaries are traction-free, meaning the forces on these surfaces are zero:
    $$\mathbf{P} \cdot \mathbf{N} = \mathbf{0}$$
    where $\mathbf{N}$ is the outward normal vector to the surface.

---

## 🧠 PINN Implementation Details

### Loss Function

The neural network is trained by minimizing a composite loss function that includes the physics residual and the boundary conditions:

$L_{total} = w_f L_{PDE} + w_{t} L_{Traction} + w_{d} L_{Displacement}$

where:
* $L_{PDE}$ is the mean squared error of the equilibrium equation residuals at interior collocation points.
    $$L_{PDE} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left\| \nabla \cdot \mathbf{P}(\mathbf{X}_i) \right\|^2$$
* $L_{Traction}$ is the mean squared error of the traction forces on the free boundaries.
    $$L_{Traction} = \frac{1}{N_t} \sum_{i=1}^{N_t} \left\| \mathbf{P}(\mathbf{X}_i) \cdot \mathbf{N}_i \right\|^2$$
* $L_{Displacement}$ is the mean squared error for the prescribed displacement boundary condition.
  

### Training Strategy

This project employs a two-phase training strategy for each displacement increment:
1.  **ADAM Optimizer**: The network is first trained for a number of epochs using the ADAM optimizer to quickly find a good region in the loss landscape.
2.  **L-BFGS Optimizer**: Subsequently, the `scipy.optimize.minimize` function with the L-BFGS algorithm is used to fine-tune the network parameters. L-BFGS is a quasi-Newton method that can achieve higher precision, which is often crucial for solving physics problems. The `utils/scipy_loss.py` module provides a wrapper to make the TensorFlow model compatible with SciPy's optimizer.

---

## ✨ Key Features

* **Hyperelastic Model**: Implements a non-linear, hyperelastic constitutive model suitable for large deformations.
* **Hybrid Optimization**: Combines the **ADAM** and **L-BFGS** optimizers for robust and precise training.
* **Incremental Loading**: Simulates a quasi-static analysis by applying displacement in small, discrete steps, allowing the model to solve a sequence of non-linear problems.
* **Strong-Form PINN**: Solves the strong form of the PDE, with Dirichlet boundary conditions enforced directly in the network's output layer.
* **Modular Code**: The project is organized into clear modules for solvers, plotting, and utilities.

