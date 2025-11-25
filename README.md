# Arotor Hackathon – Fluid Bearing Designer

This repository contains our hackathon project: an interactive tool for analyzing
fluid and air journal bearings and their interaction with a flexible rotor.

The app consists of:

- A **Python/Flask backend** (`backend.py`) that performs the numerical analysis
- A **browser-based frontend** (HTML/JS) that provides a web UI for:
  - Bearing parameter input
  - Stiffness and damping matrix computation (K, C)
  - Eccentricity calculation
  - Pressure field visualization
  - Rotor critical speed analysis and Campbell diagram
  - Example setups based on Ross / Friswell

---

## Features

- Journal bearing model (extendable to air bearings / compressible Reynolds)
- Computation of:
  - Stiffness matrix **K**
  - Damping matrix **C**
  - Bearing eccentricity **ε**
  - Pressure field **p(θ, z)**
- Rotor–bearing system:
  - Finite-element rotor model
  - Critical speeds and Campbell diagram
  - Mode shape visualization
- Preset examples (e.g. Friswell case)

---

