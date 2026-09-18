# Fractional Dynamics and Scientific Machine Learning

## Overview

This project summarizes my work on nonlinear and fractional-order dynamical systems and the transition from analytical / numerical dynamics to scientific machine learning.

The research background comes from modelling lossy electrical transmission-line systems, where nonlinear effects, fractional operators and parameter changes can produce bifurcation and chaotic behaviour.

## Research Components

### Analytical modelling
- nonlinear transmission-line models
- fractional-order formulations
- travelling-wave reduction from PDE form to ODE form
- analytical solution construction with modified expansion methods

### Dynamical analysis
- bifurcation analysis
- phase portraits and Poincaré sections
- Lyapunov-exponent-based chaos characterization
- parameter sensitivity and stability analysis

### Numerical verification
Analytical results are checked against numerical integration rather than treated as sufficient by themselves. Numerical verification is important because an apparently closed-form solution may still fail when substituted back into the governing system.

### Scientific machine learning direction
My current interest is to extend this work toward:
- Physics-Informed Neural Networks (PINNs)
- fractional PINNs
- Fourier Neural Operators (FNO)
- DeepONet
- long-horizon prediction of nonlinear / chaotic dynamics

A central question is how much useful forecast horizon can be obtained in systems where small state errors grow rapidly.

## Related Publication

My paper on fractional, bifurcation and chaotic behaviour in lossy electrical transmission-line models has been accepted by *Pramana – Journal of Physics*.

See: `../../publications/README.md`

## Current Status

Research documentation is being organized for public release. Only code that can be shared without publisher or collaboration restrictions will be added to this repository.
