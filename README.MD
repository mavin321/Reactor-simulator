
---

# **Advanced Nonlinear CSTR Simulator**

A comprehensive, single-file ANSI C simulation environment for a nonlinear, exothermic **Continuous Stirred–Tank Reactor (CSTR)** with full dynamic and steady-state capabilities.

This project is designed for **chemical engineering education, numerical methods demonstrations, reactor stability analysis, and parameter studies**.

The simulator includes robust ODE solvers, steady-state analyzers, parameter editing, dimensional analysis, and sweep utilities—all inside one C file.

---

## ⚙️ **Features**

### **Core Reactor Model**

* Irreversible, first-order exothermic reaction: **A → B**
* Mass and energy balances with optional cooling:

  * Adiabatic mode
  * Non-adiabatic cooling via heat transfer jacket
* Temperature-dependent Arrhenius kinetics
* Reaction rate ( r_A = k_0 e^{-E/(RT)} C_A )

---

## 📊 **Numerical Capabilities**

### **Steady-State Analysis**

* Newton–Raphson solver with:

  * Numerical Jacobian (central differences)
  * Backtracking line search
  * Singular matrix protection
* Automatic steady-state stability classification:

  * Jacobian eigenvalues
  * Node/saddle/focus/stability type determination

---

### **Dynamic Simulation**

Supports **four** explicit integration methods:

| Integrator    | Order   | Notes                               |
| ------------- | ------- | ----------------------------------- |
| Euler         | 1st     | Simple and fast                     |
| Heun / RK2    | 2nd     | Good accuracy, stable               |
| Classic RK4   | 4th     | Default fixed-step workhorse        |
| Adaptive RK45 | 4th/5th | Step-size control, error estimation |

Adaptive RK45 includes:

* Local truncation error estimation
* Automatic step-size adjustment
* Min/max dt limits
* Safety factors
* Rejection/acceptance logic

---

### **Parameter Tools**

* Fully interactive runtime parameter editor
* Preset reactor scenarios:

  * Default moderate exotherm
  * Adiabatic/hot feed
  * Near-isothermal strong cooling
  * Mild reaction
  * Strongly exothermic system
* Load/save parameter files (plain text)
* Dimensionless group calculator (Da, β, γ)
* Parameter sensitivity analysis using symmetric finite differences

---

### **Sweep Tools**

* Automatic sweep of feed temperature ( T_f )
* CSV export of:

  * Steady-state concentration
  * Reactor temperature
  * Conversion
  * Reaction rate

---

## 📁 **File Structure**

```
project-folder/
│
├── cstr_advanced.c     # The full simulator (1000+ lines single file)
└── README.md           # This documentation
```

---

## 🧪 **Model Equations**

The simulator solves:

### **Mass Balance**

[
\frac{dC_A}{dt} = \frac{C_{Af} - C_A}{\tau} - k_0 e^{-E/(RT)} C_A
]

### **Energy Balance**

[
\frac{dT}{dt} = \frac{T_f - T}{\tau}

* \left(\frac{-\Delta H}{\rho C_p}\right) r_A

- \left(\frac{UA}{\rho C_p V}\right)(T - T_c)
  ]

(Last term omitted in adiabatic mode.)

---

## 🔧 **Compiling**

Compile using **GCC** or any C11-compliant compiler:

```bash
gcc -O2 -Wall -std=c11 cstr_advanced.c -lm -o cstr_sim
```

Run:

```bash
./cstr_sim
```

---

## 🖥️ **Program Menu Overview**

```
================== ADVANCED CSTR REACTOR SIMULATOR ==================
1) Show current parameters and dimensionless groups
2) Edit parameters / load presets / config files
3) Solve for steady state (Newton-Raphson)
4) Analyze stability at a steady state
5) Basic dynamic simulation (fixed-step RK4)
6) Advanced dynamic simulation (multi-integrator, adaptive)
7) Sweep feed temperature and compute steady states
8) Steady-state sensitivity to a chosen parameter
0) Exit
=====================================================================
```

---

## 📈 **Output Files**

Many tools output CSV files containing:

* Time histories
* Reaction rates
* Steady-state branches
* Sensitivity analysis results

These can be imported into:

* Excel
* MATLAB
* Python (Pandas)
* Mathematica
* Plotting tools (Origin, gnuplot, etc.)

---

## 📦 **Parameter File Format**

Example of a saved configuration:

```
k0 1.0000000000000000e+06
E  8.0000000000000000e+04
dH -5.0000000000000000e+04
rho 1.0000000000000000e+03
Cp 4.1800000000000000e+03
V 1.0000000000000000e+00
tau 1.0000000000000000e+02
U 5.0000000000000000e+02
A 1.0000000000000000e+01
C_Af 2.0000000000000000e+03
T_f 3.5000000000000000e+02
T_c 3.0000000000000000e+02
adiabatic 0
```

---

## 📚 **Educational Value**

This simulator demonstrates:

* Nonlinear reactor behavior
* Multiple steady states
* Thermal runaway phenomena
* Bifurcation behavior (via sweeps)
* Numerical integration techniques
* Newton–Raphson and stability concepts
* Sensitivity and parametric dependence

Perfect for:

* Chemical engineering students
* Numerical methods courses
* Reaction engineering research
* Control engineering and simulation exercises

---

## 📌 **Planned Extensions (Optional)**

If you want, I can help you add:

* PID control loop (manipulate coolant temperature)
* Two-species model (C_A, C_B, T)
* Recycle stream and RTD effects
* Multiple reactions
* GUI (SDL2 or ImGui)

Just ask!

---

## 📝 **License**

This project is free to use, modify, and distribute for any educational or research purpose.

---
