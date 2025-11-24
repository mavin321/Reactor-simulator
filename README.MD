
---

# **CSTR Nonlinear Reactor Simulator**

*A single-file C program for solving stiff, nonlinear chemical engineering reactor models.*

---

## **Overview**

This repository contains a fully self-contained C program (`reactor_sim.c`) that models, simulates, and analyzes a **Continuous Stirred-Tank Reactor (CSTR)** undergoing an **irreversible, exothermic, first-order reaction**:

[
A \rightarrow B
]

The system includes **nonlinear reaction kinetics (Arrhenius)** and **coupled mass & energy balances**, making the reactor prone to multiple steady states, ignition/extinction behavior, and thermal runaway — phenomena essential in real chemical & process engineering.

This simulator is designed to show deep engineering understanding, numerical methods expertise, and clean C programming structure.

---

## **Key Features**

### ✔ **Nonlinear Reactor Model**

Implements full dynamic CSTR equations:

**Mass balance:**
[
\frac{dC_A}{dt} = \frac{C_{Af} - C_A}{\tau} - k_0 e^{-E/RT} C_A
]

**Energy balance:**
[
\frac{dT}{dt} = \frac{T_f - T}{\tau}
-\frac{\Delta H}{\rho C_p} , r_A
-\frac{UA}{\rho C_p V}(T - T_c)
]

Supports both **cooled** and **adiabatic** reactors.

---

### ✔ **Steady-State Solver (Newton–Raphson)**

* Computes steady states by solving
  [
  F(x) = 0
  ]
* Uses **numerical Jacobian**,
* **2×2 linear solver**,
* **Line search damping** for robustness.

This can capture multiple steady states depending on parameters.

---

### ✔ **Dynamic Simulation (Runge–Kutta 4)**

* Solves stiff ODEs over time
* Tracks concentration, temperature, reaction rate
* Generates CSV output for plotting

---

### ✔ **Parameter Sweep + CSV Export**

Performs automatic continuation over feed temperature to generate full bifurcation-style data:

```
T_f, C_A_ss, T_ss, conversion, rA
```

Useful for plotting steady-state curves, ignition points, and S-shaped reactor profiles.

---

### ✔ **Interactive Parameter Editor**

All parameters are editable during runtime:

* Kinetics (k0, E, ΔH)
* Physical properties (ρ, Cp)
* Reactor size and cooling
* Feed concentration & temperature
* Residence time
* Adiabatic toggle

---

### ✔ **Single-File, No Dependencies**

Compile with:

```bash
gcc -Wall -Wextra -pedantic -std=c11 reactor_sim.c -lm -o reactor_sim
```

Run with:

```bash
./reactor_sim
```

---

## **Typical Workflow**

### 1. **Inspect or Edit Parameters**

From the main menu:

```
1) Show current parameters
2) Edit parameters
```

### 2. **Compute a Steady State**

Enter an initial guess and optionally enable verbose Newton iterations:

```
3) Solve for steady state
```

### 3. **Dynamic Simulation**

Simulate transient response and optionally save results:

```
4) Dynamic simulation
```

CSV output can be plotted using Python, Excel, MATLAB, etc.

---

## **Example Plot (Interpretation)**

When sweeping feed temperature, the output CSV can be used to generate a characteristic S-curve:

* Lower branch: low-temperature stable steady state
* Middle branch: unstable (ignition point)
* Upper branch: high-temperature runaway steady state

These are classic nonlinear reactor phenomena.

---

## **File Structure**

| File            | Description                                                                            |
| --------------- | -------------------------------------------------------------------------------------- |
| `reactor_sim.c` | The full simulator (ODE solver, Newton solver, model equations, interactive interface) |
| `README.md`     | Documentation (this file)                                                              |

---

## **Why This Project Is Impressive**

This program demonstrates:

### 🔹 **Process Engineering Knowledge**

* Reaction kinetics
* Heat & mass balances
* Reactor stability theory
* Thermal runaway and multiple steady states

### 🔹 **Numerical Methods Expertise**

* Newton–Raphson with Jacobian
* Line search
* Runge–Kutta ODE integration
* Numerical continuation

### 🔹 **Software Engineering Discipline**

* Modular C design
* Clear structs and state management
* Robustness (bound checks, stability tools)
* Meaningful physical units everywhere

### 🔹 **Practical Deliverables**

* CSV outputs for scientific plotting
* Interactive simulations
* Fully self-contained, professional-grade example code

Perfect for demonstrating engineering, mathematical, and programming capability.

---

## **License**

MIT License (or any you prefer).

---


