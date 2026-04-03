# 🌌 Pseudospectra Computation 

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **Master Informatique Project** (Sorbonne University, May 2026)  
> **Authors:** Viktoriia Skrypnyk & Marina Svintsitska

This repository contains a high-performance Python implementation of several algorithms for computing and analyzing the **$\varepsilon$-pseudospectra** of non-normal complex matrices. 

The project bridges theoretical linear algebra with practical, parallelized high-performance computing to evaluate matrix stability under perturbations.

## ✨ Features & Implemented Algorithms

We have implemented and optimized four distinct mathematical approaches to compute pseudospectra:

1. **Grid Algorithm**
   - Computes the smallest singular value $\sigma_{\min}(A - zI)$ over a dense complex grid.
   - Reliable and robust for full topological visualization.

2. **Predictor-Corrector Algorithm (Curve Tracing)**
   - A fast, path-following algorithm that directly traces the contour of the pseudospectrum.
   - **Optimized:** Features a custom parallelized multi-threading architecture for finding starting points and tracing disconnected boundary components, achieving massive speedups.

3. **Parallel Componentwise Pseudospectrum**
   - Analyzes sensitivity to structured, componentwise perturbations using the spectral radius of the absolute resolvent.
   - Highly parallelized to efficiently generate high-resolution 3D stability topographies.

4. **Criss-Cross Algorithm**
   - A sequence of targeted 1D geometric searches (horizontal/vertical lines and radial arcs).
   - Designed for lightning-fast computation of specific scalar bounds: the **pseudospectral abscissa** and **pseudospectral radius**, without evaluating the entire complex plane.


## ⚙️ Installation

Clone the repository and install the required dependencies.
