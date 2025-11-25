#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

/*
 * =====================================================================
 *      ADVANCED NONLINEAR CSTR REACTOR SIMULATOR (SINGLE FILE C)
 * =====================================================================
 *
 * This file is an extended, "industrial strength" educational simulator
 * for a non-isothermal continuous stirred tank reactor (CSTR) with a
 * single irreversible exothermic reaction A -> B.
 *
 * Original minimal version:
 *   - 2 state variables (C_A, T)
 *   - Newton-Raphson for steady state
 *   - 4th-order Runge-Kutta for dynamics
 *   - Simple feed temperature sweep
 *
 * Extended version features (high-level):
 *   - Multiple integration schemes:
 *       * Explicit Euler
 *       * Heun / RK2
 *       * Classic RK4
 *       * Adaptive stepsize RK45 (Runge-Kutta-Fehlberg style)
 *
 *   - Steady-state analysis:
 *       * Newton-Raphson solver with line search
 *       * Stability classification using Jacobian eigenvalues
 *
 *   - Parameter handling:
 *       * Preset scenarios (adiabatic, cooled, etc.)
 *       * In-program editor for parameters
 *       * Save/load parameter sets from text configuration files
 *
 *   - Analysis tools:
 *       * Feed temperature sweeps (steady states)
 *       * Parameter sensitivity analysis for selected parameters
 *       * Dimensionless groups calculation (Da, beta, etc.)
 *
 * The aim is to provide a single-file, self-contained, reasonably
 * realistic and numerically careful simulator that is still readable.
 *
 * =====================================================================
 *           MODEL OVERVIEW (SINGLE-REACTION NONLINEAR CSTR)
 * =====================================================================
 *
 * States:
 *   C_A  [mol/m^3]   - concentration of A in reactor
 *   T    [K]         - temperature of reacting fluid
 *
 * Parameters:
 *   k0   [1/s]       - pre-exponential factor
 *   E    [J/mol]     - activation energy
 *   dH   [J/mol]     - reaction enthalpy (negative for exothermic)
 *   rho  [kg/m^3]    - density
 *   Cp   [J/kg/K]    - heat capacity
 *   V    [m^3]       - reactor volume
 *   tau  [s]         - residence time = V / volumetric_flow
 *   U    [W/m^2/K]   - heat transfer coefficient
 *   A    [m^2]       - heat transfer area
 *   C_Af [mol/m^3]   - feed concentration
 *   T_f  [K]         - feed temperature
 *   T_c  [K]         - coolant temperature
 *   adiabatic (0/1)  - if 1, no external cooling term
 *
 * Reaction:
 *   A -> B, first order in A:
 *   r_A = k0 * exp(-E/(R*T)) * C_A
 *
 * Dynamic balances:
 *
 *   dC_A/dt = (C_Af - C_A)/tau - r_A
 *
 *   dT/dt   = (T_f - T)/tau
 *             + (-dH/(rho*Cp))*r_A
 *             - (U*A/(rho*Cp*V))*(T - T_c)   (if non-adiabatic)
 *
 * At steady state, derivatives are zero.
 *
 * =====================================================================
 *                           IMPLEMENTATION NOTES
 * =====================================================================
 *
 *  - The code intentionally favors clarity and explicitness over
 *    compactness. Many comments are included for teaching purposes.
 *
 *  - The numerics are NOT guaranteed bulletproof, but several safeguards
 *    are included (e.g. clamped exponentials, line searches, etc.).
 *
 *  - This file is long by design, to show how a more "comprehensive"
 *    simulator might look while still fitting in a single source file.
 *
 * =====================================================================
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------------------------------------------------------------
 * Global constants
 * ------------------------------------------------------------------ */

/* Universal gas constant [J/mol/K] */
static const double R_gas = 8.314462618;

/* Numerically safe large/small values */
#define VERY_SMALL 1e-300
#define VERY_LARGE 1e300

/* ---------------------------------------------------------------------
 * Data structures
 * ------------------------------------------------------------------ */

/* Process and kinetic parameters */
typedef struct {
    double k0;       /* 1/s */
    double E;        /* J/mol */
    double dH;       /* J/mol (negative => exothermic) */
    double rho;      /* kg/m^3 */
    double Cp;       /* J/kg/K */
    double V;        /* m^3 */
    double tau;      /* s */
    double U;        /* W/m^2/K */
    double A;        /* m^2 */
    double C_Af;     /* mol/m^3 */
    double T_f;      /* K */
    double T_c;      /* K */
    int    adiabatic;/* 1 => omit cooling term */
} CSTRParams;

/* State vector: C_A and T */
typedef struct {
    double C_A;      /* mol/m^3 */
    double T;        /* K */
} State;

/* Integrator choices for dynamic simulation */
typedef enum {
    INTEGRATOR_EULER = 0,
    INTEGRATOR_RK2   = 1,
    INTEGRATOR_RK4   = 2,
    INTEGRATOR_RK45  = 3
} IntegratorType;

/* Options for dynamic simulation */
typedef struct {
    IntegratorType method;  /* which integration scheme to use */
    double t_final;         /* final time [s] */
    double dt_initial;      /* initial time step [s] */
    double dt_min;          /* minimum allowed time step [s] */
    double dt_max;          /* maximum allowed time step [s] */
    double rel_tol;         /* relative tolerance for adaptive schemes */
    double abs_tol;         /* absolute tolerance for adaptive schemes */
    int    max_steps;       /* max number of integration steps */
    int    output_every;    /* write every N steps (>=1) */
    int    adaptive;        /* 1 => adapt step (only makes sense for RK45) */
} SimOptions;

/* Structure for dimensionless groups / derived info */
typedef struct {
    double Da;      /* Damkohler number */
    double beta;    /* adiabatic temperature rise factor */
    double gamma;   /* dimensionless activation energy E/(R*T_f) */
} DimensionlessInfo;

/* ---------------------------------------------------------------------
 * Utility functions
 * ------------------------------------------------------------------ */

/* Clamp value to [min_val, max_val] */
static double clamp(double x, double min_val, double max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

/* Safe exponential to avoid overflow/underflow */
static double safe_exp(double x) {
    if (x > 700.0) x = 700.0;
    if (x < -700.0) x = -700.0;
    return exp(x);
}

/* Safe logarithm (never called with <=0 in principle) */
static double safe_log(double x) {
    if (x <= 0.0) x = VERY_SMALL;
    return log(x);
}

/* Simple squared norm of residual vector */
static double norm2_state(const State *s) {
    return sqrt(s->C_A * s->C_A + s->T * s->T);
}

/* Add two states: z = x + y */
static void state_add(const State *x, const State *y, State *z) {
    z->C_A = x->C_A + y->C_A;
    z->T   = x->T   + y->T;
}

/* Scale state: y = alpha * x */
static void state_scale(const State *x, double alpha, State *y) {
    y->C_A = alpha * x->C_A;
    y->T   = alpha * x->T;
}

/* z = x + alpha * y */
static void state_axpy(const State *x, double alpha, const State *y, State *z) {
    z->C_A = x->C_A + alpha * y->C_A;
    z->T   = x->T   + alpha * y->T;
}

/* ---------------------------------------------------------------------
 * Model: reaction rate and ODE right-hand side
 * ------------------------------------------------------------------ */

/* Reaction rate r_A = k0 * exp(-E/(R*T)) * C_A */
static double reaction_rate(const CSTRParams *p, const State *s) {
    double T = s->T;
    if (T < 1.0) {
        /* Prevent division by zero / crazy exponent at T ~ 0 K */
        T = 1.0;
    }
    double k = p->k0 * safe_exp(-p->E / (R_gas * T));
    double rA = k * s->C_A;
    if (!isfinite(rA)) {
        /* Fail-safe clamp */
        if (rA > 0.0) rA = VERY_LARGE;
        else          rA = -VERY_LARGE;
    }
    return rA;
}

/*
 * cstr_rhs:
 *   Compute time derivatives d(state)/dt = f(state, params)
 */
static void cstr_rhs(const CSTRParams *p, const State *s, State *dydt) {
    double C_A = s->C_A;
    double T   = s->T;

    /* enforce basic physical bounds to avoid non-physical growth */
    if (C_A < 0.0) C_A = 0.0;
    if (T   < 150.0) T = 150.0; /* just a safety lower bound */

    double rA = reaction_rate(p, s);

    /* Material balance on A */
    double dCAdt = (p->C_Af - C_A) / p->tau - rA;

    /* Heat generation term (K/s) */
    double heat_gen = (-p->dH / (p->rho * p->Cp)) * rA;

    /* Heat removal term (cooling) */
    double heat_rem = 0.0;
    if (!p->adiabatic) {
        heat_rem = (p->U * p->A) / (p->rho * p->Cp * p->V) * (p->T_c - T);
    }

    /* Energy balance */
    double dTdt = (p->T_f - T) / p->tau + heat_gen + heat_rem;

    dydt->C_A = dCAdt;
    dydt->T   = dTdt;
}

/*
 * Steady-state residual:
 *   At steady state, dC_A/dt = 0, dT/dt = 0.
 *   So F(x) = f(x) = RHS evaluated at x.
 */
static void cstr_steady_residual(const CSTRParams *p, const State *x, State *res) {
    cstr_rhs(p, x, res);
}

/* ---------------------------------------------------------------------
 * Jacobian & linear algebra for Newton-Raphson
 * ------------------------------------------------------------------ */

/*
 * Compute numerical Jacobian of steady-state residual:
 *   J_ij = dF_i/dx_j
 * where x = [C_A, T].
 *
 * We use central finite differences.
 */
static void numerical_jacobian(const CSTRParams *p, const State *x, double J[2][2]) {
    const double eps = 1e-6;
    State x_plus, x_minus, r_plus, r_minus;

    /* dF/dC_A */
    x_plus  = *x;
    x_minus = *x;
    x_plus.C_A  += eps;
    x_minus.C_A -= eps;
    cstr_steady_residual(p, &x_plus,  &r_plus);
    cstr_steady_residual(p, &x_minus, &r_minus);

    J[0][0] = (r_plus.C_A - r_minus.C_A) / (2.0 * eps);
    J[1][0] = (r_plus.T   - r_minus.T)   / (2.0 * eps);

    /* dF/dT */
    x_plus  = *x;
    x_minus = *x;
    x_plus.T  += eps;
    x_minus.T -= eps;
    cstr_steady_residual(p, &x_plus,  &r_plus);
    cstr_steady_residual(p, &x_minus, &r_minus);

    J[0][1] = (r_plus.C_A - r_minus.C_A) / (2.0 * eps);
    J[1][1] = (r_plus.T   - r_minus.T)   / (2.0 * eps);
}

/*
 * Solve 2x2 linear system: J * dx = -F
 *
 * Returns 1 on success, 0 if Jacobian is singular.
 */
static int solve_2x2(const double J[2][2], const State *F, State *dx) {
    double a = J[0][0];
    double b = J[0][1];
    double c = J[1][0];
    double d = J[1][1];

    double det = a * d - b * c;
    if (fabs(det) < 1e-20) {
        return 0; /* nearly singular */
    }
    double inv_det = 1.0 / det;

    dx->C_A = (-d * F->C_A + b * F->T) * inv_det;
    dx->T   = ( c * F->C_A - a * F->T) * inv_det;
    return 1;
}

/* ---------------------------------------------------------------------
 * Newton-Raphson for steady state
 * ------------------------------------------------------------------ */

static int newton_steady_state(const CSTRParams *p,
                               State *x,
                               int max_iter,
                               double tol,
                               int verbose)
{
    int iter;
    for (iter = 0; iter < max_iter; ++iter) {
        State F;
        cstr_steady_residual(p, x, &F);

        double normF = norm2_state(&F);
        if (verbose) {
            printf("  Iter %2d: C_A = %.10g, T = %.10g, |F| = %.3e\n",
                   iter, x->C_A, x->T, normF);
        }
        if (normF < tol) {
            return 1; /* converged */
        }

        /* Compute Jacobian */
        double J[2][2];
        numerical_jacobian(p, x, J);

        State dx;
        if (!solve_2x2((const double (*)[2])J, &F, &dx)) {
            if (verbose) {
                printf("  Jacobian is singular or ill-conditioned.\n");
            }
            return 0;
        }

        /* Backtracking line search for global convergence */
        double alpha = 1.0;
        const double c1 = 1e-4;  /* Armijo constant */
        State x_trial, F_trial;

        /* Compute directional derivative approx: F^T * dx */


        int ls_ok = 0;
        for (int ls_iter = 0; ls_iter < 15; ++ls_iter) {
            x_trial.C_A = x->C_A - alpha * dx.C_A;
            x_trial.T   = x->T   - alpha * dx.T;

            /* Enforce simple physical bounds */
            if (x_trial.C_A < 0.0)    x_trial.C_A = 1e-9;
            if (x_trial.T   < 200.0)  x_trial.T   = 200.0;

            cstr_steady_residual(p, &x_trial, &F_trial);
            double normF_trial = norm2_state(&F_trial);

            if (normF_trial <= (1.0 - c1 * alpha) * normF || normF_trial < normF) {
                *x = x_trial;
                ls_ok = 1;
                break;
            }

            alpha *= 0.5;
        }

        if (!ls_ok) {
            if (verbose) {
                printf("  Line search failed to reduce residual; aborting.\n");
            }
            return 0;
        }
    }

    if (verbose) {
        printf("  Newton did not converge in %d iterations.\n", max_iter);
    }
    return 0;
}

/* ---------------------------------------------------------------------
 * Stability analysis at steady state
 * ------------------------------------------------------------------ */

/*
 * Analyze local stability of a steady-state solution x_ss by computing
 * eigenvalues of the 2x2 Jacobian J.
 *
 * For 2x2:
 *   tr(J) = J11 + J22
 *   det(J) = J11*J22 - J12*J21
 *   Eigenvalues are roots of lambda^2 - tr*lambda + det = 0.
 */
static void analyze_steady_state_stability(const CSTRParams *p, const State *x_ss) {
    double J[2][2];
    numerical_jacobian(p, x_ss, J);

    double tr  = J[0][0] + J[1][1];
    double det = J[0][0] * J[1][1] - J[0][1] * J[1][0];

    double disc = tr * tr - 4.0 * det;

    printf("\nStability analysis at steady state:\n");
    printf("  Jacobian J = [ [%.6e, %.6e],\n"
           "                 [%.6e, %.6e] ]\n",
           J[0][0], J[0][1], J[1][0], J[1][1]);
    printf("  Trace(J)      = %.6e\n", tr);
    printf("  Determinant(J)= %.6e\n", det);

    if (disc >= 0) {
        double sqrt_disc = sqrt(disc);
        double lambda1 = 0.5 * (tr + sqrt_disc);
        double lambda2 = 0.5 * (tr - sqrt_disc);
        printf("  Eigenvalues are real:\n");
        printf("    lambda1 = %.6e\n", lambda1);
        printf("    lambda2 = %.6e\n", lambda2);

        printf("  Classification: ");
        if (lambda1 < 0.0 && lambda2 < 0.0) {
            printf("Asymptotically stable node (both negative).\n");
        } else if (lambda1 > 0.0 && lambda2 > 0.0) {
            printf("Unstable node (both positive).\n");
        } else if (lambda1 * lambda2 < 0.0) {
            printf("Saddle point (one positive, one negative).\n");
        } else if (fabs(lambda1) < 1e-9 && fabs(lambda2) < 1e-9) {
            printf("Approximately marginal (both near zero).\n");
        } else {
            printf("Mixed / borderline case.\n");
        }
    } else {
        double real_part = 0.5 * tr;
        double imag_part = 0.5 * sqrt(-disc);
        printf("  Eigenvalues are complex conjugates:\n");
        printf("    lambda = %.6e +/- %.6e i\n", real_part, imag_part);

        printf("  Classification: ");
        if (real_part < 0.0) {
            printf("Stable focus (damped oscillations).\n");
        } else if (real_part > 0.0) {
            printf("Unstable focus (growing oscillations).\n");
        } else {
            printf("Center / marginally stable (purely imaginary).\n");
        }
    }
}

/* ---------------------------------------------------------------------
 * Time-stepping (Euler, RK2, RK4, RK45)
 * ------------------------------------------------------------------ */

/* Single explicit Euler step */
static void euler_step(const CSTRParams *p, const State *y, double dt, State *yout) {
    State f;
    cstr_rhs(p, y, &f);
    yout->C_A = y->C_A + dt * f.C_A;
    yout->T   = y->T   + dt * f.T;
}

/* Heun's method (RK2) */
static void rk2_step(const CSTRParams *p, const State *y, double dt, State *yout) {
    State k1, k2, y_tmp;

    cstr_rhs(p, y, &k1);
    y_tmp.C_A = y->C_A + dt * k1.C_A;
    y_tmp.T   = y->T   + dt * k1.T;

    cstr_rhs(p, &y_tmp, &k2);

    yout->C_A = y->C_A + dt * 0.5 * (k1.C_A + k2.C_A);
    yout->T   = y->T   + dt * 0.5 * (k1.T   + k2.T);
}

/* Classic RK4 step */
static void rk4_step(const CSTRParams *p, const State *y, double dt, State *yout) {
    State k1, k2, k3, k4, yt;

    cstr_rhs(p, y, &k1);

    yt.C_A = y->C_A + 0.5 * dt * k1.C_A;
    yt.T   = y->T   + 0.5 * dt * k1.T;
    cstr_rhs(p, &yt, &k2);

    yt.C_A = y->C_A + 0.5 * dt * k2.C_A;
    yt.T   = y->T   + 0.5 * dt * k2.T;
    cstr_rhs(p, &yt, &k3);

    yt.C_A = y->C_A + dt * k3.C_A;
    yt.T   = y->T   + dt * k3.T;
    cstr_rhs(p, &yt, &k4);

    yout->C_A = y->C_A + dt * (k1.C_A + 2*k2.C_A + 2*k3.C_A + k4.C_A) / 6.0;
    yout->T   = y->T   + dt * (k1.T   + 2*k2.T   + 2*k3.T   + k4.T)   / 6.0;
}

/*
 * RK45 step (Fehlberg-like embedded pair) with error estimate.
 * This is a single step; the caller adjusts dt adaptively based on err.
 *
 * References: classic Runge-Kutta-Fehlberg 4(5) method.
 */
static void rk45_step(const CSTRParams *p,
                      const State *y,
                      double t,
                      double dt,
                      State *y4,      /* 4th-order estimate */
                      State *y5,      /* 5th-order estimate */
                      State *err_est) /* local error estimate */
{
    (void)t; /* t not used explicitly in autonomous system but kept for clarity */

    /* Fehlberg coefficients */
    const double a2 = 1.0/4.0;
    const double a3 = 3.0/8.0;
    const double a4 = 12.0/13.0;
    const double a5 = 1.0;
    const double a6 = 1.0/2.0;

    const double b21 = 1.0/4.0;

    const double b31 = 3.0/32.0;
    const double b32 = 9.0/32.0;

    const double b41 = 1932.0/2197.0;
    const double b42 = -7200.0/2197.0;
    const double b43 = 7296.0/2197.0;

    const double b51 = 439.0/216.0;
    const double b52 = -8.0;
    const double b53 = 3680.0/513.0;
    const double b54 = -845.0/4104.0;

    const double b61 = -8.0/27.0;
    const double b62 = 2.0;
    const double b63 = -3544.0/2565.0;
    const double b64 = 1859.0/4104.0;
    const double b65 = -11.0/40.0;

    const double c1 = 16.0/135.0;
    const double c3 = 6656.0/12825.0;
    const double c4 = 28561.0/56430.0;
    const double c5 = -9.0/50.0;
    const double c6 = 2.0/55.0;

    const double c1_star = 25.0/216.0;
    const double c3_star = 1408.0/2565.0;
    const double c4_star = 2197.0/4104.0;
    const double c5_star = -1.0/5.0;

    State k1, k2, k3, k4, k5, k6;
    State y_temp;

    /* k1 */
    cstr_rhs(p, y, &k1);

    /* k2 */
    y_temp.C_A = y->C_A + dt * b21 * k1.C_A;
    y_temp.T   = y->T   + dt * b21 * k1.T;
    cstr_rhs(p, &y_temp, &k2);

    /* k3 */
    y_temp.C_A = y->C_A + dt * (b31 * k1.C_A + b32 * k2.C_A);
    y_temp.T   = y->T   + dt * (b31 * k1.T   + b32 * k2.T);
    cstr_rhs(p, &y_temp, &k3);

    /* k4 */
    y_temp.C_A = y->C_A + dt * (b41 * k1.C_A + b42 * k2.C_A + b43 * k3.C_A);
    y_temp.T   = y->T   + dt * (b41 * k1.T   + b42 * k2.T   + b43 * k3.T);
    cstr_rhs(p, &y_temp, &k4);

    /* k5 */
    y_temp.C_A = y->C_A + dt * (b51 * k1.C_A + b52 * k2.C_A + b53 * k3.C_A + b54 * k4.C_A);
    y_temp.T   = y->T   + dt * (b51 * k1.T   + b52 * k2.T   + b53 * k3.T   + b54 * k4.T);
    cstr_rhs(p, &y_temp, &k5);

    /* k6 */
    y_temp.C_A = y->C_A + dt * (b61 * k1.C_A + b62 * k2.C_A + b63 * k3.C_A + b64 * k4.C_A + b65 * k5.C_A);
    y_temp.T   = y->T   + dt * (b61 * k1.T   + b62 * k2.T   + b63 * k3.T   + b64 * k4.T   + b65 * k5.T);
    cstr_rhs(p, &y_temp, &k6);

    /* 5th-order solution */
    y5->C_A = y->C_A + dt * (c1 * k1.C_A + c3 * k3.C_A + c4 * k4.C_A + c5 * k5.C_A + c6 * k6.C_A);
    y5->T   = y->T   + dt * (c1 * k1.T   + c3 * k3.T   + c4 * k4.T   + c5 * k5.T   + c6 * k6.T);

    /* 4th-order solution (star) */
    y4->C_A = y->C_A + dt * (c1_star * k1.C_A + c3_star * k3.C_A + c4_star * k4.C_A + c5_star * k5.C_A);
    y4->T   = y->T   + dt * (c1_star * k1.T   + c3_star * k3.T   + c4_star * k4.T   + c5_star * k5.T);

    /* Local error estimate: y5 - y4 */
    err_est->C_A = y5->C_A - y4->C_A;
    err_est->T   = y5->T   - y4->T;
}

/* Compute weighted RMS norm of local error */
static double error_norm(const State *y, const State *err, double rel_tol, double abs_tol) {
    double sc_CA = abs_tol + rel_tol * fabs(y->C_A);
    double sc_T  = abs_tol + rel_tol * fabs(y->T);

    double e_CA = err->C_A / sc_CA;
    double e_T  = err->T   / sc_T;

    double rms = sqrt(0.5 * (e_CA * e_CA + e_T * e_T));
    return rms;
}

/* ---------------------------------------------------------------------
 * Dynamic simulation: simple version (fixed RK4) as in original
 * ------------------------------------------------------------------ */

static void simulate_dynamics_basic(const CSTRParams *p,
                                    const State *y0,
                                    double t_final,
                                    double dt,
                                    const char *outfile)
{
    FILE *fp = NULL;
    if (outfile && strlen(outfile) > 0) {
        fp = fopen(outfile, "w");
        if (!fp) {
            perror("Failed to open output file");
        }
    }

    printf("\n[Basic dynamic simulation: RK4, fixed step]\n");
    printf("  t_final = %.4g s, dt = %.4g s\n", t_final, dt);
    printf("  %-12s %-16s %-16s %-16s\n",
           "time[s]", "C_A[mol/m^3]", "T[K]", "r_A[mol/m^3/s]");

    if (fp) {
        fprintf(fp, "# time_s,C_A_mol_m3,T_K,rA_mol_m3_s\n");
    }

    State y = *y0;
    double t = 0.0;

    while (t <= t_final + 1e-12) {
        double rA = reaction_rate(p, &y);
        printf("  %-12.4f %-16.8g %-16.8g %-16.8g\n", t, y.C_A, y.T, rA);
        if (fp) {
            fprintf(fp, "%.8f,%.12g,%.12g,%.12g\n", t, y.C_A, y.T, rA);
        }

        State y_next;
        rk4_step(p, &y, dt, &y_next);
        y = y_next;
        t += dt;
    }

    if (fp) {
        fclose(fp);
        printf("  Results written to '%s' (CSV).\n", outfile);
    }
}

/* ---------------------------------------------------------------------
 * Dynamic simulation: advanced version with multiple integrators
 * ------------------------------------------------------------------ */

static void simulate_dynamics_advanced(const CSTRParams *p,
                                       const State *y0,
                                       const SimOptions *opt,
                                       const char *outfile)
{
    FILE *fp = NULL;
    if (outfile && strlen(outfile) > 0 && strcmp(outfile, "-") != 0) {
        fp = fopen(outfile, "w");
        if (!fp) {
            perror("Failed to open advanced output file");
        }
    }

    const char *method_name = "Unknown";
    switch (opt->method) {
        case INTEGRATOR_EULER: method_name = "Euler (1st-order)"; break;
        case INTEGRATOR_RK2:   method_name = "Heun / RK2 (2nd-order)"; break;
        case INTEGRATOR_RK4:   method_name = "Classic RK4 (4th-order)"; break;
        case INTEGRATOR_RK45:  method_name = "Adaptive RK45 (4/5th-order)"; break;
        default: break;
    }

    printf("\n[Advanced dynamic simulation]\n");
    printf("  Integrator     : %s\n", method_name);
    printf("  t_final        : %.6g s\n", opt->t_final);
    printf("  dt_initial     : %.6g s\n", opt->dt_initial);
    printf("  dt_min         : %.6g s\n", opt->dt_min);
    printf("  dt_max         : %.6g s\n", opt->dt_max);
    printf("  rel_tol        : %.3e\n", opt->rel_tol);
    printf("  abs_tol        : %.3e\n", opt->abs_tol);
    printf("  max_steps      : %d\n", opt->max_steps);
    printf("  adaptive       : %s\n", opt->adaptive ? "yes" : "no");

    printf("\n  %-12s %-16s %-16s %-16s %-10s\n",
           "time[s]", "C_A[mol/m^3]", "T[K]", "r_A[mol/m^3/s]", "dt[s]");

    if (fp) {
        fprintf(fp, "# time_s,C_A_mol_m3,T_K,rA_mol_m3_s,dt_s\n");
    }

    State y = *y0;
    double t = 0.0;
    double dt = opt->dt_initial;
    if (dt <= 0.0) dt = (opt->t_final > 0.0) ? opt->t_final / 100.0 : 1.0;

    int step = 0;
    int printed_header = 0;

    while (t < opt->t_final && step < opt->max_steps) {
        if (dt < opt->dt_min) dt = opt->dt_min;
        if (dt > opt->dt_max) dt = opt->dt_max;

        if (t + dt > opt->t_final) {
            dt = opt->t_final - t;
        }

        State y_new;
        double err = 0.0;

        if (opt->method == INTEGRATOR_RK45 && opt->adaptive) {
            State y4, y5, err_est;
            rk45_step(p, &y, t, dt, &y4, &y5, &err_est);
            err = error_norm(&y5, &err_est, opt->rel_tol, opt->abs_tol);

            if (err <= 1.0) {
                /* Accept step, use higher order solution y5 */
                y_new = y5;
                t += dt;
                ++step;

                if (step % opt->output_every == 0 || !printed_header) {
                    double rA = reaction_rate(p, &y_new);
                    printf("  %-12.6g %-16.10g %-16.10g %-16.10g %-10.3g\n",
                           t, y_new.C_A, y_new.T, rA, dt);
                    if (fp) {
                        fprintf(fp, "%.12g,%.12g,%.12g,%.12g,%.12g\n",
                                t, y_new.C_A, y_new.T, rA, dt);
                    }
                    printed_header = 1;
                }

                y = y_new;

                /* Adapt step size: increase if error was small */
                double safety = 0.9;
                double factor = safety * pow(err + 1e-16, -0.25);
                factor = clamp(factor, 0.2, 5.0);
                dt = dt * factor;
            } else {
                /* Reject step, shrink dt */
                double safety = 0.9;
                double factor = safety * pow(err + 1e-16, -0.25);
                factor = clamp(factor, 0.1, 0.5);
                dt = dt * factor;
                continue;
            }
        } else {
            /* Fixed-step integrators */
            switch (opt->method) {
                case INTEGRATOR_EULER:
                    euler_step(p, &y, dt, &y_new);
                    break;
                case INTEGRATOR_RK2:
                    rk2_step(p, &y, dt, &y_new);
                    break;
                case INTEGRATOR_RK4:
                default:
                    rk4_step(p, &y, dt, &y_new);
                    break;
            }

            t += dt;
            ++step;
            if (step % opt->output_every == 0 || !printed_header) {
                double rA = reaction_rate(p, &y_new);
                printf("  %-12.6g %-16.10g %-16.10g %-16.10g %-10.3g\n",
                       t, y_new.C_A, y_new.T, rA, dt);
                if (fp) {
                    fprintf(fp, "%.12g,%.12g,%.12g,%.12g,%.12g\n",
                            t, y_new.C_A, y_new.T, rA, dt);
                }
                printed_header = 1;
            }
            y = y_new;
        }
    }

    if (step >= opt->max_steps) {
        printf("\nWarning: reached maximum number of steps (%d) before t_final.\n",
               opt->max_steps);
    }

    if (fp) {
        fclose(fp);
        printf("  Advanced simulation results written to '%s'.\n", outfile);
    }
}

/* ---------------------------------------------------------------------
 * Steady-state sweep over feed temperature
 * ------------------------------------------------------------------ */

static void sweep_feed_temperature(CSTRParams *base_params,
                                   double T_start, double T_end, int n_steps,
                                   double C_A_guess, double T_guess,
                                   const char *outfile)
{
    FILE *fp = fopen(outfile, "w");
    if (!fp) {
        perror("Failed to open sweep output file");
        return;
    }

    fprintf(fp, "# T_f[K],C_A_ss[mol/m^3],T_ss[K],conversion[-],rA_ss[mol/m^3_s]\n");

    CSTRParams p = *base_params;
    State x;
    x.C_A = C_A_guess;
    x.T   = T_guess;

    printf("\n[Sweep over feed temperature T_f]\n");
    printf("  T_start = %.3f K, T_end = %.3f K, n_steps = %d\n",
           T_start, T_end, n_steps);

    for (int i = 0; i <= n_steps; ++i) {
        double alpha = (double)i / (double)n_steps;
        p.T_f = T_start + alpha * (T_end - T_start);

        State x_init = x;
        int ok = newton_steady_state(&p, &x, 80, 1e-10, 0);
        if (!ok) {
            fprintf(fp, "%.8g,NaN,NaN,NaN,NaN\n", p.T_f);
            x = x_init;
        } else {
            double conversion = 1.0 - x.C_A / p.C_Af;
            double rA = reaction_rate(&p, &x);
            fprintf(fp, "%.8g,%.12g,%.12g,%.12g,%.12g\n",
                    p.T_f, x.C_A, x.T, conversion, rA);
        }
    }

    fclose(fp);
    printf("  Sweep results written to '%s' (CSV).\n", outfile);
}

/* ---------------------------------------------------------------------
 * Parameter sensitivity analysis
 * ------------------------------------------------------------------ */

typedef enum {
    PARAM_K0 = 1,
    PARAM_E,
    PARAM_DH,
    PARAM_RHO,
    PARAM_CP,
    PARAM_V,
    PARAM_TAU,
    PARAM_U,
    PARAM_A,
    PARAM_CAF,
    PARAM_TF,
    PARAM_TC
} ParamID;

static const char *param_name_from_id(ParamID id) {
    switch (id) {
        case PARAM_K0:  return "k0";
        case PARAM_E:   return "E";
        case PARAM_DH:  return "dH";
        case PARAM_RHO: return "rho";
        case PARAM_CP:  return "Cp";
        case PARAM_V:   return "V";
        case PARAM_TAU: return "tau";
        case PARAM_U:   return "U";
        case PARAM_A:   return "A";
        case PARAM_CAF: return "C_Af";
        case PARAM_TF:  return "T_f";
        case PARAM_TC:  return "T_c";
        default:        return "unknown";
    }
}

static double *param_pointer(CSTRParams *p, ParamID id) {
    switch (id) {
        case PARAM_K0:  return &p->k0;
        case PARAM_E:   return &p->E;
        case PARAM_DH:  return &p->dH;
        case PARAM_RHO: return &p->rho;
        case PARAM_CP:  return &p->Cp;
        case PARAM_V:   return &p->V;
        case PARAM_TAU: return &p->tau;
        case PARAM_U:   return &p->U;
        case PARAM_A:   return &p->A;
        case PARAM_CAF: return &p->C_Af;
        case PARAM_TF:  return &p->T_f;
        case PARAM_TC:  return &p->T_c;
        default:        return NULL;
    }
}

/*
 * Perform sensitivity analysis of steady-state conversion with respect
 * to a selected parameter using symmetric finite differences.
 */
static void steady_state_sensitivity(CSTRParams *base,
                                     ParamID param_id,
                                     double rel_delta,
                                     const State *guess)
{
    CSTRParams p_plus  = *base;
    CSTRParams p_minus = *base;

    double *param_plus  = param_pointer(&p_plus,  param_id);
    double *param_minus = param_pointer(&p_minus, param_id);

    if (!param_plus || !param_minus) {
        printf("Invalid parameter id.\n");
        return;
    }

    double base_value = *param_plus;
    if (fabs(base_value) < 1e-16) {
        printf("Cannot perform relative perturbation on parameter with value ~ 0.\n");
        return;
    }

    double delta = rel_delta * fabs(base_value);
    *param_plus  = base_value + delta;
    *param_minus = base_value - delta;

    State x_plus  = *guess;
    State x_minus = *guess;

    int ok_plus  = newton_steady_state(&p_plus,  &x_plus,  80, 1e-10, 0);
    int ok_minus = newton_steady_state(&p_minus, &x_minus, 80, 1e-10, 0);

    if (!ok_plus || !ok_minus) {
        printf("Newton failed for perturbed parameters (plus=%d, minus=%d).\n",
               ok_plus, ok_minus);
        return;
    }

    double conv_base = 1.0 - guess->C_A / base->C_Af;
    double conv_plus = 1.0 - x_plus.C_A  / p_plus.C_Af;
    double conv_minus= 1.0 - x_minus.C_A / p_minus.C_Af;

    double dconv_dparam = (conv_plus - conv_minus) / (2.0 * delta);

    printf("\nSensitivity of steady-state conversion to parameter '%s':\n",
           param_name_from_id(param_id));
    printf("  Base parameter value   : %.10g\n", base_value);
    printf("  Perturbation delta     : %.10g (%.3g%%)\n", delta, rel_delta*100.0);
    printf("  Base conversion        : %.10g\n", conv_base);
    printf("  Conversion (plus)      : %.10g\n", conv_plus);
    printf("  Conversion (minus)     : %.10g\n", conv_minus);
    printf("  d(conversion)/d(%s)    : %.10g\n",
           param_name_from_id(param_id), dconv_dparam);
}

/* ---------------------------------------------------------------------
 * Dimensionless groups
 * ------------------------------------------------------------------ */

static void compute_dimensionless_groups(const CSTRParams *p,
                                         DimensionlessInfo *info)
{
    /*
     * These are heuristic / educational dimensionless groups. For a
     * rigorous derivation, one would nondimensionalize the governing
     * equations carefully. Here we define:
     *
     * Da    ~ k0 * exp(-E/(R*T_f)) * tau
     * beta  ~ (-dH * C_Af) / (rho * Cp * T_f)
     * gamma ~ E / (R * T_f)
     */

    double T_ref = p->T_f;
    if (T_ref < 1.0) T_ref = 300.0;

    double k_ref = p->k0 * safe_exp(-p->E / (R_gas * T_ref));
    info->Da    = k_ref * p->tau;
    info->beta  = (-p->dH * p->C_Af) / (p->rho * p->Cp * T_ref);
    info->gamma = p->E / (R_gas * T_ref);
}

static void print_dimensionless_groups(const CSTRParams *p) {
    DimensionlessInfo info;
    compute_dimensionless_groups(p, &info);

    printf("\nApproximate dimensionless groups:\n");
    printf("  Damkohler number Da   ~ %.6g\n", info.Da);
    printf("  Beta (heat release)   ~ %.6g\n", info.beta);
    printf("  Gamma (activation)    ~ %.6g\n", info.gamma);
}

/* ---------------------------------------------------------------------
 * Default parameters and presets
 * ------------------------------------------------------------------ */

static void default_params(CSTRParams *p) {
    p->k0   = 1.0e6;      /* 1/s */
    p->E    = 8.0e4;      /* J/mol */
    p->dH   = -5.0e4;     /* J/mol (exothermic) */
    p->rho  = 1000.0;     /* kg/m^3 */
    p->Cp   = 4180.0;     /* J/kg/K */
    p->V    = 1.0;        /* m^3 */
    p->tau  = 100.0;      /* s */
    p->U    = 500.0;      /* W/m^2/K */
    p->A    = 10.0;       /* m^2 */
    p->C_Af = 2000.0;     /* mol/m^3 */
    p->T_f  = 350.0;      /* K */
    p->T_c  = 300.0;      /* K */
    p->adiabatic = 0;
}

/* A few presets to quickly explore different behaviors */
static void preset_adiabatic_hot(CSTRParams *p) {
    default_params(p);
    p->adiabatic = 1;
    p->T_f       = 380.0;
    p->C_Af      = 2500.0;
}

static void preset_near_isothermal(CSTRParams *p) {
    default_params(p);
    p->U   = 3000.0;
    p->A   = 20.0;
    p->T_c = 340.0;
}

static void preset_mild_reaction(CSTRParams *p) {
    default_params(p);
    p->k0 = 2.5e4;
    p->E  = 6.0e4;
    p->dH = -3.0e4;
}

static void preset_strongly_exothermic(CSTRParams *p) {
    default_params(p);
    p->dH  = -8.0e4;
    p->C_Af= 3000.0;
    p->T_f = 360.0;
}

static void print_params(const CSTRParams *p) {
    printf("\nCurrent CSTR parameters:\n");
    printf("  k0   = %.6g 1/s\n",  p->k0);
    printf("  E    = %.6g J/mol\n", p->E);
    printf("  dH   = %.6g J/mol (negative = exothermic)\n", p->dH);
    printf("  rho  = %.6g kg/m^3\n", p->rho);
    printf("  Cp   = %.6g J/kg/K\n", p->Cp);
    printf("  V    = %.6g m^3\n", p->V);
    printf("  tau  = %.6g s\n", p->tau);
    printf("  U    = %.6g W/m^2/K\n", p->U);
    printf("  A    = %.6g m^2\n", p->A);
    printf("  C_Af = %.6g mol/m^3\n", p->C_Af);
    printf("  T_f  = %.6g K\n", p->T_f);
    printf("  T_c  = %.6g K\n", p->T_c);
    printf("  Adiabatic: %s\n", p->adiabatic ? "yes" : "no");

    print_dimensionless_groups(p);
}

/* ---------------------------------------------------------------------
 * Parameter editing and preset menu
 * ------------------------------------------------------------------ */

static void print_preset_menu(void) {
    printf("\nParameter preset menu:\n");
    printf(" 1) Default moderate exotherm\n");
    printf(" 2) Adiabatic, hotter feed (runaway-prone)\n");
    printf(" 3) Near-isothermal (strong cooling)\n");
    printf(" 4) Mild reaction (lower k0, E, |dH|)\n");
    printf(" 5) Strongly exothermic, higher feed conc\n");
    printf(" 0) Cancel\n");
    printf("Select preset: ");
}

static void apply_preset(CSTRParams *p) {
    print_preset_menu();
    int choice;
    if (scanf("%d", &choice) != 1) {
        fprintf(stderr, "Invalid input.\n");
        return;
    }
    switch (choice) {
        case 1: default_params(p);            break;
        case 2: preset_adiabatic_hot(p);      break;
        case 3: preset_near_isothermal(p);    break;
        case 4: preset_mild_reaction(p);      break;
        case 5: preset_strongly_exothermic(p);break;
        case 0: default:                      break;
    }
}

/*
 * Simple text-file configuration:
 *
 * Each line: <name> <value>
 *   e.g., k0 1.0e6
 *         E  8.0e4
 *         adiabatic 1
 *
 * Names recognized: k0, E, dH, rho, Cp, V, tau, U, A, C_Af, T_f, T_c, adiabatic
 */

static void save_params_to_file(const CSTRParams *p, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open parameter file for writing");
        return;
    }

    fprintf(fp, "# CSTR parameter file\n");
    fprintf(fp, "k0 % .16e\n", p->k0);
    fprintf(fp, "E % .16e\n", p->E);
    fprintf(fp, "dH % .16e\n", p->dH);
    fprintf(fp, "rho % .16e\n", p->rho);
    fprintf(fp, "Cp % .16e\n", p->Cp);
    fprintf(fp, "V % .16e\n", p->V);
    fprintf(fp, "tau % .16e\n", p->tau);
    fprintf(fp, "U % .16e\n", p->U);
    fprintf(fp, "A % .16e\n", p->A);
    fprintf(fp, "C_Af % .16e\n", p->C_Af);
    fprintf(fp, "T_f % .16e\n", p->T_f);
    fprintf(fp, "T_c % .16e\n", p->T_c);
    fprintf(fp, "adiabatic %d\n", p->adiabatic);

    fclose(fp);
    printf("Parameters saved to '%s'.\n", filename);
}

static int fpeek(FILE *fp);

static void load_params_from_file(CSTRParams *p, const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open parameter file for reading");
        return;
    }

    char name[64];
    double val;
    int ival;

    while (!feof(fp)) {
        int c = fpeek(fp);
        if (c == '#') {
            /* Comment line, skip */
            int ch;
            while ((ch = fgetc(fp)) != EOF && ch != '\n') { /* nop */ }
            continue;
        }

        if (fscanf(fp, "%63s", name) != 1) {
            break;
        }

        if (strcmp(name, "adiabatic") == 0) {
            if (fscanf(fp, "%d", &ival) == 1) {
                p->adiabatic = ival ? 1 : 0;
            }
        } else {
            if (fscanf(fp, "%lf", &val) != 1) {
                break;
            }
            if      (strcmp(name, "k0")   == 0) p->k0   = val;
            else if (strcmp(name, "E")    == 0) p->E    = val;
            else if (strcmp(name, "dH")   == 0) p->dH   = val;
            else if (strcmp(name, "rho")  == 0) p->rho  = val;
            else if (strcmp(name, "Cp")   == 0) p->Cp   = val;
            else if (strcmp(name, "V")    == 0) p->V    = val;
            else if (strcmp(name, "tau")  == 0) p->tau  = val;
            else if (strcmp(name, "U")    == 0) p->U    = val;
            else if (strcmp(name, "A")    == 0) p->A    = val;
            else if (strcmp(name, "C_Af") == 0) p->C_Af = val;
            else if (strcmp(name, "T_f")  == 0) p->T_f  = val;
            else if (strcmp(name, "T_c")  == 0) p->T_c  = val;
        }
    }

    fclose(fp);
    printf("Parameters loaded from '%s'.\n", filename);
}

/* Helper: peek next non-whitespace char from a FILE without consuming it */
static int fpeek(FILE *fp) {
    int c;
    do {
        c = fgetc(fp);
        if (c == EOF) return EOF;
    } while (isspace(c));
    ungetc(c, fp);
    return c;
}

static void edit_params(CSTRParams *p) {
    int choice = -1;
    while (choice != 0) {
        print_params(p);
        printf("\nParameter editing menu:\n");
        printf(" 1) k0   (1/s)\n");
        printf(" 2) E    (J/mol)\n");
        printf(" 3) dH   (J/mol)\n");
        printf(" 4) rho  (kg/m^3)\n");
        printf(" 5) Cp   (J/kg/K)\n");
        printf(" 6) V    (m^3)\n");
        printf(" 7) tau  (s)\n");
        printf(" 8) U    (W/m^2/K)\n");
        printf(" 9) A    (m^2)\n");
        printf("10) C_Af (mol/m^3)\n");
        printf("11) T_f  (K)\n");
        printf("12) T_c  (K)\n");
        printf("13) Toggle adiabatic\n");
        printf("14) Apply preset scenario\n");
        printf("15) Save parameters to file\n");
        printf("16) Load parameters from file\n");
        printf(" 0) Done\n> ");

        if (scanf("%d", &choice) != 1) {
            fprintf(stderr, "Invalid input. Exiting parameter editor.\n");
            return;
        }

        double val;
        char fname[256];
        switch (choice) {
            case 1:  printf("Enter new k0: ");  if (scanf("%lf", &val)==1) p->k0=val; break;
            case 2:  printf("Enter new E: ");   if (scanf("%lf", &val)==1) p->E=val;  break;
            case 3:  printf("Enter new dH: ");  if (scanf("%lf", &val)==1) p->dH=val; break;
            case 4:  printf("Enter new rho: "); if (scanf("%lf", &val)==1) p->rho=val;break;
            case 5:  printf("Enter new Cp: ");  if (scanf("%lf", &val)==1) p->Cp=val; break;
            case 6:  printf("Enter new V: ");   if (scanf("%lf", &val)==1) p->V=val;  break;
            case 7:  printf("Enter new tau: "); if (scanf("%lf", &val)==1) p->tau=val;break;
            case 8:  printf("Enter new U: ");   if (scanf("%lf", &val)==1) p->U=val;  break;
            case 9:  printf("Enter new A: ");   if (scanf("%lf", &val)==1) p->A=val;  break;
            case 10: printf("Enter new C_Af: ");if (scanf("%lf", &val)==1) p->C_Af=val;break;
            case 11: printf("Enter new T_f: "); if (scanf("%lf", &val)==1) p->T_f=val;break;
            case 12: printf("Enter new T_c: "); if (scanf("%lf", &val)==1) p->T_c=val;break;
            case 13: p->adiabatic = !p->adiabatic; break;
            case 14: apply_preset(p); break;
            case 15:
                printf("Enter filename to save parameters: ");
                if (scanf("%255s", fname)==1) {
                    save_params_to_file(p, fname);
                }
                break;
            case 16:
                printf("Enter filename to load parameters: ");
                if (scanf("%255s", fname)==1) {
                    load_params_from_file(p, fname);
                }
                break;
            case 0: default: break;
        }
    }
}

/* ---------------------------------------------------------------------
 * Menus
 * ------------------------------------------------------------------ */

static void print_main_menu(void) {
    printf("\n================== ADVANCED CSTR REACTOR SIMULATOR ==================\n");
    printf(" 1) Show current parameters and dimensionless groups\n");
    printf(" 2) Edit parameters / load presets / config files\n");
    printf(" 3) Solve for steady state (Newton-Raphson)\n");
    printf(" 4) Analyze stability at a steady state\n");
    printf(" 5) Basic dynamic simulation (fixed-step RK4)\n");
    printf(" 6) Advanced dynamic simulation (multi-integrator, adaptive)\n");
    printf(" 7) Sweep feed temperature and compute steady states\n");
    printf(" 8) Steady-state sensitivity to a chosen parameter\n");
    printf(" 0) Exit\n");
    printf("=====================================================================\n");
    printf("Enter choice: ");
}

static void print_integrator_menu(void) {
    printf("\nIntegrator options:\n");
    printf(" 0) Euler (1st order)\n");
    printf(" 1) Heun / RK2 (2nd order)\n");
    printf(" 2) Classic RK4 (4th order)\n");
    printf(" 3) Adaptive RK45 (4/5th order)\n");
    printf("Select method: ");
}

static void print_param_sensitivity_menu(void) {
    printf("\nSelect parameter for sensitivity analysis:\n");
    printf(" 1) k0\n");
    printf(" 2) E\n");
    printf(" 3) dH\n");
    printf(" 4) rho\n");
    printf(" 5) Cp\n");
    printf(" 6) V\n");
    printf(" 7) tau\n");
    printf(" 8) U\n");
    printf(" 9) A\n");
    printf("10) C_Af\n");
    printf("11) T_f\n");
    printf("12) T_c\n");
    printf(" 0) Cancel\n");
    printf("Select parameter: ");
}

/* ---------------------------------------------------------------------
 * MAIN
 * ------------------------------------------------------------------ */

int main(void) {
    CSTRParams params;
    default_params(&params);

    int running = 1;
    while (running) {
        print_main_menu();
        int choice;
        if (scanf("%d", &choice) != 1) {
            fprintf(stderr, "Invalid input; exiting.\n");
            break;
        }

        if (choice == 0) {
            running = 0;
        }
        else if (choice == 1) {
            print_params(&params);
        }
        else if (choice == 2) {
            edit_params(&params);
        }
        else if (choice == 3) {
            /* Steady-state solve */
            State guess;
            printf("Enter initial guess for C_A (mol/m^3): ");
            if (scanf("%lf", &guess.C_A) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter initial guess for T (K): ");
            if (scanf("%lf", &guess.T) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            int verbose;
            printf("Verbose Newton output? (1=yes,0=no): ");
            if (scanf("%d", &verbose) != 1) verbose = 0;

            State x = guess;
            int ok = newton_steady_state(&params, &x, 80, 1e-10, verbose);
            if (!ok) {
                printf("\nNewton solver failed to converge from this initial guess.\n");
            } else {
                double conversion = 1.0 - x.C_A / params.C_Af;
                double rA_ss = reaction_rate(&params, &x);
                printf("\nSteady-state solution:\n");
                printf("  C_A = %.12g mol/m^3\n", x.C_A);
                printf("  T   = %.12g K\n", x.T);
                printf("  Conversion = %.12g (-)\n", conversion);
                printf("  r_A = %.12g mol/m^3/s\n", rA_ss);
            }
        }
        else if (choice == 4) {
            /* Stability analysis */
            State guess;
            printf("Enter steady-state point (or good guess) C_A (mol/m^3): ");
            if (scanf("%lf", &guess.C_A) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter steady-state point (or good guess) T (K): ");
            if (scanf("%lf", &guess.T) != 1) { fprintf(stderr, "Bad input.\n"); continue; }

            State x = guess;
            int ok = newton_steady_state(&params, &x, 80, 1e-10, 0);
            if (!ok) {
                printf("Failed to find steady state near given guess.\n");
            } else {
                printf("Using steady state: C_A = %.12g, T = %.12g\n", x.C_A, x.T);
                analyze_steady_state_stability(&params, &x);
            }
        }
        else if (choice == 5) {
            /* Basic dynamic simulation, original RK4 */
            State y0;
            double t_final, dt;
            char fname[256];

            printf("Enter initial C_A (mol/m^3): ");
            if (scanf("%lf", &y0.C_A) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter initial T (K): ");
            if (scanf("%lf", &y0.T) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter final time (s): ");
            if (scanf("%lf", &t_final) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter time step dt (s): ");
            if (scanf("%lf", &dt) != 1 || dt <= 0.0) { fprintf(stderr, "Bad dt.\n"); continue; }
            printf("Enter output CSV filename (or '-' for no file): ");
            if (scanf("%255s", fname) != 1) { fprintf(stderr, "Bad filename.\n"); continue; }

            if (strcmp(fname, "-") == 0) {
                simulate_dynamics_basic(&params, &y0, t_final, dt, NULL);
            } else {
                simulate_dynamics_basic(&params, &y0, t_final, dt, fname);
            }
        }
        else if (choice == 6) {
            /* Advanced dynamic simulation */
            State y0;
            SimOptions opt;
            char fname[256];

            printf("Enter initial C_A (mol/m^3): ");
            if (scanf("%lf", &y0.C_A) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter initial T (K): ");
            if (scanf("%lf", &y0.T) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter final time (s): ");
            if (scanf("%lf", &opt.t_final) != 1) { fprintf(stderr, "Bad input.\n"); continue; }

            print_integrator_menu();
            int method;
            if (scanf("%d", &method) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            if (method < 0 || method > 3) method = 2; /* default RK4 */

            opt.method = (IntegratorType)method;

            printf("Enter initial time step dt_initial (s): ");
            if (scanf("%lf", &opt.dt_initial) != 1 || opt.dt_initial <= 0.0) {
                opt.dt_initial = opt.t_final / 100.0;
            }
            printf("Enter minimum time step dt_min (s): ");
            if (scanf("%lf", &opt.dt_min) != 1 || opt.dt_min <= 0.0) {
                opt.dt_min = opt.dt_initial * 1e-3;
            }
            printf("Enter maximum time step dt_max (s): ");
            if (scanf("%lf", &opt.dt_max) != 1 || opt.dt_max <= opt.dt_min) {
                opt.dt_max = opt.dt_initial * 10.0;
            }

            printf("Enter relative tolerance (e.g. 1e-4): ");
            if (scanf("%lf", &opt.rel_tol) != 1 || opt.rel_tol <= 0.0) {
                opt.rel_tol = 1e-4;
            }
            printf("Enter absolute tolerance (e.g. 1e-8): ");
            if (scanf("%lf", &opt.abs_tol) != 1 || opt.abs_tol <= 0.0) {
                opt.abs_tol = 1e-8;
            }
            printf("Enter max number of steps (e.g. 100000): ");
            if (scanf("%d", &opt.max_steps) != 1 || opt.max_steps <= 0) {
                opt.max_steps = 100000;
            }
            printf("Enter output frequency (print every N steps, N>=1): ");
            if (scanf("%d", &opt.output_every) != 1 || opt.output_every <= 0) {
                opt.output_every = 1;
            }
            printf("Use adaptive step size (1=yes,0=no)? ");
            if (scanf("%d", &opt.adaptive) != 1) opt.adaptive = 0;

            printf("Enter output CSV filename (or '-' for no file): ");
            if (scanf("%255s", fname) != 1) { fprintf(stderr, "Bad filename.\n"); continue; }

            simulate_dynamics_advanced(&params, &y0, &opt, fname);
        }
        else if (choice == 7) {
            /* Temperature sweep */
            double T_start, T_end;
            int n_steps;
            State guess;
            char fname[256];

            printf("Enter start feed temperature T_start (K): ");
            if (scanf("%lf", &T_start) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter end feed temperature T_end (K): ");
            if (scanf("%lf", &T_end) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter number of steps (e.g., 100): ");
            if (scanf("%d", &n_steps) != 1 || n_steps <= 0) {
                fprintf(stderr, "Bad n_steps.\n"); continue;
            }
            printf("Enter initial guess C_A (mol/m^3): ");
            if (scanf("%lf", &guess.C_A) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter initial guess T (K): ");
            if (scanf("%lf", &guess.T) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter sweep output CSV filename: ");
            if (scanf("%255s", fname) != 1) { fprintf(stderr, "Bad filename.\n"); continue; }

            sweep_feed_temperature(&params, T_start, T_end, n_steps,
                                   guess.C_A, guess.T, fname);
        }
        else if (choice == 8) {
            /* Parameter sensitivity */
            print_param_sensitivity_menu();
            int pid;
            if (scanf("%d", &pid) != 1 || pid <= 0 || pid > 12) {
                printf("Cancelled or invalid.\n");
                continue;
            }
            ParamID param_id = (ParamID)pid;
            double rel_delta;
            printf("Enter relative perturbation size (e.g. 0.01 for 1%%): ");
            if (scanf("%lf", &rel_delta) != 1 || rel_delta <= 0.0) {
                rel_delta = 0.01;
            }

            State guess;
            printf("Enter initial guess C_A for steady state (mol/m^3): ");
            if (scanf("%lf", &guess.C_A) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter initial guess T for steady state (K): ");
            if (scanf("%lf", &guess.T) != 1) { fprintf(stderr, "Bad input.\n"); continue; }

            State x = guess;
            int ok = newton_steady_state(&params, &x, 80, 1e-10, 0);
            if (!ok) {
                printf("Could not converge to steady state from this guess.\n");
                continue;
            }
            steady_state_sensitivity(&params, param_id, rel_delta, &x);
        }
        else {
            printf("Unknown choice.\n");
        }
    }

    printf("Exiting advanced CSTR simulator.\n");
    return 0;
}
