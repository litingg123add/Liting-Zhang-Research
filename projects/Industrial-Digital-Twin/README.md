# Industrial Sensing and Digital-Twin Prototype

## Overview

This project documents an engineering workflow developed from hands-on work in industrial sensing, calibration and monitoring.

The emphasis is not on a purely graphical "digital twin", but on the full chain:

```
sensor → acquisition → physical interpretation → monitoring logic → HMI
```

## Engineering Background

During my technology-development internship in Germany, I worked on a pressure-calibration / monitoring prototype involving embedded sensing, data acquisition and a browser-based HMI.

Publicly shareable technical elements include:
- Raspberry Pi based edge computing
- ADC and distance-sensor integration
- SPI / I2C device communication
- real-time measurement collection
- Flask-based HMI
- live curves and multilingual interface
- engineering threshold logic
- basic physical interpretation using fluid-flow models

## System View

### 1. Sensor layer
Signals are collected from multiple sensors with different sampling and interface characteristics.

### 2. Acquisition layer
Raw values are converted into a consistent digital representation with timing and calibration information.

### 3. Physical layer
Measurements are interpreted with engineering constraints rather than treated as isolated numbers. Depending on the subsystem, this can include flow, pressure, geometric or operating-condition relationships.

### 4. Monitoring layer
Thresholds and trend logic are used to identify abnormal states and support calibration / testing.

### 5. HMI layer
The monitoring state and live measurements are shown through a lightweight web interface for engineering use.

## Why It Matters for My Research

This experience motivates my interest in combining data-driven models with physical knowledge. In industrial monitoring, a model has to work with imperfect sensors, changing operating conditions and real system constraints.

## Confidentiality Note

No proprietary company source code, calibration constants, customer data or internal technical documents are included in this public repository.
