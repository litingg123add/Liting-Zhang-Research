# Liting Zhang — Research Portfolio

Master's student in Electronic Information at Shanghai Dianji University  
Research interests: Scientific Machine Learning, Predictive Maintenance, Nonlinear Dynamics, Industrial AI

## Research Profile

My current work connects data-driven modelling with engineering knowledge for complex dynamical systems. I am particularly interested in condition monitoring under changing operating conditions, physics-guided machine learning, nonlinear dynamics, and interpretable health indicators for industrial systems.

## Featured Project

### Wind Turbine Predictive Health Monitoring

A reproducible research prototype for SCADA-based condition monitoring with emphasis on **operating-condition-aware normal-behaviour modelling**.

Main questions:
- How can genuine degradation be separated from changes caused by wind speed, load and operating regime?
- How should a normal-behaviour model be validated without time leakage?
- Can residual-based indicators provide earlier and more interpretable warning signals around maintenance events?

Code and documentation: `projects/Wind-PHM-AI/`

Current pipeline:
1. SCADA preprocessing and timestamp handling
2. Operating-condition feature engineering
3. Normal-behaviour model training
4. Residual construction and EWMA smoothing
5. Maintenance-event-oriented evaluation

A synthetic end-to-end demo is included so the pipeline can be run without redistributing the original dataset.

## Other Research Directions

### Fractional Dynamics + Scientific Machine Learning
Work on nonlinear and fractional-order dynamical systems, bifurcation and chaotic behaviour, numerical simulation, and future integration with PINNs / neural operators.

See: `projects/Fractional-Dynamics-AI/`

### Industrial Digital Twin / Sensing
Engineering work related to sensor acquisition, physical modelling, embedded systems and real-time monitoring.

See: `projects/Industrial-Digital-Twin/`

## Publications

Selected accepted research outputs and publication status are summarized in:

`publications/README.md`

## Technical Stack

**Programming:** Python, MATLAB, C/C++, Java  
**Data / ML:** pandas, NumPy, scikit-learn, PyTorch, time-series analysis  
**Scientific ML:** PINNs, neural operators, data-driven dynamical modelling  
**Engineering:** sensor systems, embedded development, data acquisition, industrial monitoring

## Repository Notes

This repository is a research portfolio rather than a dump of raw project files. Public code is organized to make the methodology easy to inspect and reproduce. Raw industrial or licensed data are not uploaded.

## Contact

GitHub: https://github.com/litingg123add
