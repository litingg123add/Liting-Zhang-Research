# Industrial Sensing and Embedded Monitoring

## Overview

This section documents engineering work in industrial sensing, calibration, embedded acquisition and real-time monitoring.

The core engineering chain is:

```
sensor → acquisition → physical interpretation → monitoring logic → HMI
```

It is intentionally described as an **industrial sensing / monitoring prototype**, rather than presenting the internship system as a full digital twin.

## Engineering background

During my technology-development internship in Germany, I worked on a pressure-calibration and monitoring prototype involving embedded sensing, data acquisition and a browser-based HMI.

Publicly shareable technical elements include:
- Raspberry Pi 5 based edge computing;
- ADC, pressure / load and distance-sensor integration;
- SPI / I2C device communication;
- real-time measurement collection;
- Python calibration and monitoring logic;
- Flask-based HMI;
- live curves and multilingual interface;
- engineering threshold logic;
- physical interpretation using pressure–flow relationships.

## System view

### 1. Sensor layer
Signals are collected from sensors with different interfaces and measurement characteristics.

### 2. Acquisition layer
Raw measurements are converted into a consistent digital representation with timing and calibration information.

### 3. Physical interpretation
Measurements are interpreted using engineering constraints rather than treated as isolated numbers. Depending on the subsystem, this includes pressure, flow, geometry and operating-condition relationships.

### 4. Monitoring layer
Threshold and trend logic are used to support calibration, testing and abnormal-state identification.

### 5. HMI layer
Live measurements and monitoring state are displayed through a lightweight browser-based interface.

## Why it matters for my research

This experience is one reason I am interested in combining data-driven models with physical knowledge. Real industrial monitoring has to cope with sensor imperfections, changing operating conditions and hardware constraints, not only offline benchmark accuracy.

## Confidentiality

No proprietary MEDTRON source code, calibration constants, customer data or internal technical documents are included in this public repository. This page describes only the engineering workflow and tools that can be disclosed safely.
