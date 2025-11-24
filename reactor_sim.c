#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
 * Nonlinear CSTR Simulator (single file, pure C)
 *
 * Models an adiabatic or non-adiabatic continuous stirred-tank reactor
 * with a single irreversible exothermic reaction: A -> B
 *
 * Governing equations (dynamic):
 *
 *   dC_A/dt = (C_Af - C_A)/tau - r_A
 *   dT/dt   = (T_f - T)/tau + (-dH/(rho*Cp))*r_A - (U*A/(rho*Cp*V))*(T - T_c)
 *   r_A     = k0 * exp(-E/(R*T)) * C_A
 *
 * This program can:
 *   - Solve for steady states via Newton-Raphson
 *   - Simulate dynamic behavior with 4th-order Runge-Kutta
 *   - Sweep a parameter (feed temperature) and trace steady states
 *
 * The goal is to demonstrate fairly sophisticated numerical methods,
 * clear structure, and engineering-style documentation in a single C file.
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Physical constants */
const double R_gas = 8.314462618; /* J/mol/K */

/* Structure holding all process/kinetic parameters */
typedef struct {
    double k0;      /* pre-exponential factor, 1/s */
    double E;       /* activation energy, J/mol */
    double dH;      /* reaction enthalpy (negative for exothermic), J/mol */
    double rho;     /* density, kg/m^3 */
    double Cp;      /* heat capacity, J/kg/K */
    double V;       /* reactor volume, m^3 */
    double tau;     /* residence time, s (V / volumetric_flow) */
    double U;       /* heat transfer coefficient, W/m^2/K */
    double A;       /* heat transfer area, m^2 */
    double C_Af;    /* feed concentration, mol/m^3 */
    double T_f;     /* feed temperature, K */
    double T_c;     /* coolant temperature, K */
    int    adiabatic; /* 1 = adiabatic (no cooling term), 0 = with cooling */
} CSTRParams;

/* State vector: [C_A, T] */
typedef struct {
    double C_A;
    double T;
} State;

/* Utility: safe exponential to avoid underflow/overflow */
static double safe_exp(double x) {
    /* clamp argument to reasonable range */
    if (x > 700.0) x = 700.0;
    if (x < -700.0) x = -700.0;
    return exp(x);
}

/* Reaction rate r_A = k0 * exp(-E/(R*T)) * C_A */
static double reaction_rate(const CSTRParams *p, const State *s) {
    double k = p->k0 * safe_exp(-p->E / (R_gas * s->T));
    return k * s->C_A;
}

/* Right-hand side of ODEs: d(state)/dt = f(state, params) */
static void cstr_rhs(const CSTRParams *p, const State *s, State *dydt) {
    double rA = reaction_rate(p, s);
    double C_A = s->C_A;
    double T = s->T;

    double dCAdt = (p->C_Af - C_A) / p->tau - rA;

    double heat_gen = (-p->dH / (p->rho * p->Cp)) * rA;  /* K/s */
    double heat_rem = 0.0;
    if (!p->adiabatic) {
        heat_rem = (p->U * p->A) / (p->rho * p->Cp * p->V) * (p->T_c - T);
    }
    double dTdt = (p->T_f - T) / p->tau + heat_gen + heat_rem;

    dydt->C_A = dCAdt;
    dydt->T = dTdt;
}

/* Compute residuals for steady-state equations f(x) = 0 */
static void cstr_steady_residual(const CSTRParams *p, const State *x, State *res) {
    State rhs;
    cstr_rhs(p, x, &rhs);
    /* At steady state, dC_A/dt = 0, dT/dt = 0 */
    res->C_A = rhs.C_A;
    res->T = rhs.T;
}

/* Numerical Jacobian using central differences */
static void numerical_jacobian(const CSTRParams *p, const State *x, double J[2][2]) {
    const double eps = 1e-6;
    State x_plus, x_minus, r_plus, r_minus;

    /* dF/dC_A */
    x_plus = *x;
    x_minus = *x;
    x_plus.C_A += eps;
    x_minus.C_A -= eps;
    cstr_steady_residual(p, &x_plus, &r_plus);
    cstr_steady_residual(p, &x_minus, &r_minus);
    J[0][0] = (r_plus.C_A - r_minus.C_A) / (2 * eps);
    J[1][0] = (r_plus.T - r_minus.T) / (2 * eps);

    /* dF/dT */
    x_plus = *x;
    x_minus = *x;
    x_plus.T += eps;
    x_minus.T -= eps;
    cstr_steady_residual(p, &x_plus, &r_plus);
    cstr_steady_residual(p, &x_minus, &r_minus);
    J[0][1] = (r_plus.C_A - r_minus.C_A) / (2 * eps);
    J[1][1] = (r_plus.T - r_minus.T) / (2 * eps);
}

/* Solve 2x2 linear system J * dx = -F using explicit formula */
static int solve_2x2(const double J[2][2], const State *F, State *dx) {
    double a = J[0][0];
    double b = J[0][1];
    double c = J[1][0];
    double d = J[1][1];

    double det = a * d - b * c;
    if (fabs(det) < 1e-14) {
        return 0; /* singular */
    }
    double inv_det = 1.0 / det;

    /* dx = -J^{-1} F */
    dx->C_A = (-d * F->C_A + b * F->T) * inv_det;
    dx->T   = ( c * F->C_A - a * F->T) * inv_det;
    return 1;
}

/* Newton-Raphson solver for steady state */
static int newton_steady_state(const CSTRParams *p, State *x, int max_iter, double tol, int verbose) {
    int iter;
    for (iter = 0; iter < max_iter; ++iter) {
        State F;
        cstr_steady_residual(p, x, &F);
        double normF = sqrt(F.C_A * F.C_A + F.T * F.T);
        if (verbose) {
            printf("  Iter %2d: C_A = %.6g mol/m^3, T = %.6g K, |F| = %.3e\n",
                   iter, x->C_A, x->T, normF);
        }
        if (normF < tol) {
            return 1; /* converged */
        }

        double J[2][2];
        numerical_jacobian(p, x, J);
        State dx;
        if (!solve_2x2((const double (*)[2])J, &F, &dx)) {
            if (verbose) {
                printf("  Jacobian is singular; aborting Newton.\n");
            }
            return 0;
        }

        /* Simple damping line search */
        double alpha = 1.0;
        State x_trial;
        State F_trial;
        int ls_iter;
        for (ls_iter = 0; ls_iter < 10; ++ls_iter) {
            x_trial.C_A = x->C_A - alpha * dx.C_A;
            x_trial.T   = x->T   - alpha * dx.T;
            /* enforce physical bounds */
            if (x_trial.C_A < 0.0) x_trial.C_A = 1e-9;
            if (x_trial.T < 200.0) x_trial.T = 200.0;

            cstr_steady_residual(p, &x_trial, &F_trial);
            double norm_trial = sqrt(F_trial.C_A * F_trial.C_A + F_trial.T * F_trial.T);
            if (norm_trial < normF) {
                *x = x_trial;
                break;
            }
            alpha *= 0.5;
        }
        if (ls_iter == 10) {
            if (verbose) {
                printf("  Line search failed to reduce residual; aborting Newton.\n");
            }
            return 0;
        }
    }
    if (verbose) {
        printf("  Newton did not converge within %d iterations.\n", max_iter);
    }
    return 0;
}

/* 4th-order Runge-Kutta integration for a single step */
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

/* Simulate dynamics over [0, t_final] with fixed step size */
static void simulate_dynamics(const CSTRParams *p, const State *y0, double t_final, double dt, const char *outfile) {
    FILE *fp = NULL;
    if (outfile && strlen(outfile) > 0) {
        fp = fopen(outfile, "w");
        if (!fp) {
            perror("Failed to open output file");
        }
    }

    printf("\nDynamic simulation: t_final = %.2f s, dt = %.4f s\n", t_final, dt);
    printf("  %-12s %-16s %-16s %-16s\n", "time[s]", "C_A[mol/m^3]", "T[K]", "r_A[mol/m^3/s]");

    if (fp) {
        fprintf(fp, "# time_s,C_A_mol_m3,T_K,rA_mol_m3_s\n");
    }

    State y = *y0;
    double t = 0.0;
    while (t <= t_final + 1e-12) {
        double rA = reaction_rate(p, &y);
        printf("  %-12.4f %-16.6g %-16.6g %-16.6g\n", t, y.C_A, y.T, rA);
        if (fp) {
            fprintf(fp, "%.8f,%.8g,%.8g,%.8g\n", t, y.C_A, y.T, rA);
        }

        State ynext;
        rk4_step(p, &y, dt, &ynext);
        y = ynext;
        t += dt;
    }

    if (fp) {
        fclose(fp);
        printf("\nResults written to '%s' (CSV).\n", outfile);
    }
}

/* Sweep feed temperature and compute steady states, write to CSV */
static void sweep_feed_temperature(CSTRParams *base_params,
                                   double T_start, double T_end, int n_steps,
                                   double C_A_guess, double T_guess,
                                   const char *outfile) {
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

    for (int i = 0; i <= n_steps; ++i) {
        double alpha = (double)i / (double)n_steps;
        p.T_f = T_start + alpha * (T_end - T_start);

        State x_init = x; /* continuation seed */
        int ok = newton_steady_state(&p, &x, 50, 1e-10, 0);
        if (!ok) {
            fprintf(fp, "%.8g,NaN,NaN,NaN,NaN\n", p.T_f);
            x = x_init; /* revert seed */
        } else {
            double conversion = 1.0 - x.C_A / p.C_Af;
            double rA = reaction_rate(&p, &x);
            fprintf(fp, "%.8g,%.8g,%.8g,%.8g,%.8g\n",
                    p.T_f, x.C_A, x.T, conversion, rA);
        }
    }

    fclose(fp);
    printf("\nSweep results written to '%s' (CSV).\n", outfile);
}

/* Default parameter set representing a modestly exothermic reaction */
static void default_params(CSTRParams *p) {
    p->k0   = 1.0e6;      /* 1/s */
    p->E    = 8.0e4;      /* J/mol */
    p->dH   = -5.0e4;     /* J/mol (exothermic) */
    p->rho  = 1000.0;     /* kg/m^3 */
    p->Cp   = 4180.0;     /* J/kg/K (water-like) */
    p->V    = 1.0;        /* m^3 */
    p->tau  = 100.0;      /* s */
    p->U    = 500.0;      /* W/m^2/K */
    p->A    = 10.0;       /* m^2 */
    p->C_Af = 2000.0;     /* mol/m^3 */
    p->T_f  = 350.0;      /* K */
    p->T_c  = 300.0;      /* K */
    p->adiabatic = 0;
}

static void print_params(const CSTRParams *p) {
    printf("\nCurrent CSTR parameters:\n");
    printf("  k0   = %.3g 1/s\n", p->k0);
    printf("  E    = %.3g J/mol\n", p->E);
    printf("  dH   = %.3g J/mol (negative = exothermic)\n", p->dH);
    printf("  rho  = %.3g kg/m^3\n", p->rho);
    printf("  Cp   = %.3g J/kg/K\n", p->Cp);
    printf("  V    = %.3g m^3\n", p->V);
    printf("  tau  = %.3g s\n", p->tau);
    printf("  U    = %.3g W/m^2/K\n", p->U);
    printf("  A    = %.3g m^2\n", p->A);
    printf("  C_Af = %.3g mol/m^3\n", p->C_Af);
    printf("  T_f  = %.3g K\n", p->T_f);
    printf("  T_c  = %.3g K\n", p->T_c);
    printf("  Adiabatic: %s\n", p->adiabatic ? "yes" : "no");
}

static void edit_params(CSTRParams *p) {
    int choice = -1;
    while (choice != 0) {
        print_params(p);
        printf("\nParameter editing menu (enter number to change, 0 to exit):\n");
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
        printf(" 0) Done\n> ");
        if (scanf("%d", &choice) != 1) {
            fprintf(stderr, "Invalid input. Exiting parameter editor.\n");
            return;
        }
        double val;
        switch (choice) {
            case 1: printf("Enter new k0: "); if (scanf("%lf", &val)==1) p->k0=val; break;
            case 2: printf("Enter new E: "); if (scanf("%lf", &val)==1) p->E=val; break;
            case 3: printf("Enter new dH: "); if (scanf("%lf", &val)==1) p->dH=val; break;
            case 4: printf("Enter new rho: "); if (scanf("%lf", &val)==1) p->rho=val; break;
            case 5: printf("Enter new Cp: "); if (scanf("%lf", &val)==1) p->Cp=val; break;
            case 6: printf("Enter new V: "); if (scanf("%lf", &val)==1) p->V=val; break;
            case 7: printf("Enter new tau: "); if (scanf("%lf", &val)==1) p->tau=val; break;
            case 8: printf("Enter new U: "); if (scanf("%lf", &val)==1) p->U=val; break;
            case 9: printf("Enter new A: "); if (scanf("%lf", &val)==1) p->A=val; break;
            case 10: printf("Enter new C_Af: "); if (scanf("%lf", &val)==1) p->C_Af=val; break;
            case 11: printf("Enter new T_f: "); if (scanf("%lf", &val)==1) p->T_f=val; break;
            case 12: printf("Enter new T_c: "); if (scanf("%lf", &val)==1) p->T_c=val; break;
            case 13: p->adiabatic = !p->adiabatic; break;
            case 0: default: break;
        }
    }
}

static void menu() {
    printf("\n================ CSTR Nonlinear Reactor Simulator ================\n");
    printf(" 1) Show current parameters\n");
    printf(" 2) Edit parameters\n");
    printf(" 3) Solve for steady state (Newton-Raphson)\n");
    printf(" 4) Dynamic simulation (Runge-Kutta)\n");
    printf(" 5) Sweep feed temperature and compute steady states\n");
    printf(" 0) Exit\n");
    printf("================================================================\n");
    printf("Enter choice: ");
}

int main(void) {
    CSTRParams params;
    default_params(&params);

    int running = 1;
    while (running) {
        menu();
        int choice;
        if (scanf("%d", &choice) != 1) {
            fprintf(stderr, "Invalid input; exiting.\n");
            break;
        }

        if (choice == 0) {
            running = 0;
        } else if (choice == 1) {
            print_params(&params);
        } else if (choice == 2) {
            edit_params(&params);
        } else if (choice == 3) {
            State guess;
            printf("Enter initial guess for C_A (mol/m^3): ");
            if (scanf("%lf", &guess.C_A) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter initial guess for T (K): ");
            if (scanf("%lf", &guess.T) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            int verbose;
            printf("Verbose Newton output? (1=yes,0=no): ");
            if (scanf("%d", &verbose) != 1) verbose = 0;

            State x = guess;
            int ok = newton_steady_state(&params, &x, 50, 1e-10, verbose);
            if (!ok) {
                printf("\nNewton solver failed to converge from this initial guess.\n");
            } else {
                double conversion = 1.0 - x.C_A / params.C_Af;
                double rA_ss = reaction_rate(&params, &x);
                printf("\nSteady-state solution:\n");
                printf("  C_A = %.6g mol/m^3\n", x.C_A);
                printf("  T   = %.6g K\n", x.T);
                printf("  Conversion = %.6g (-)\n", conversion);
                printf("  r_A = %.6g mol/m^3/s\n", rA_ss);
            }
        } else if (choice == 4) {
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
                simulate_dynamics(&params, &y0, t_final, dt, NULL);
            } else {
                simulate_dynamics(&params, &y0, t_final, dt, fname);
            }
        } else if (choice == 5) {
            double T_start, T_end;
            int n_steps;
            State guess;
            char fname[256];

            printf("Enter start feed temperature T_start (K): ");
            if (scanf("%lf", &T_start) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter end feed temperature T_end (K): ");
            if (scanf("%lf", &T_end) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter number of steps (e.g., 100): ");
            if (scanf("%d", &n_steps) != 1 || n_steps <= 0) { fprintf(stderr, "Bad n_steps.\n"); continue; }
            printf("Enter initial guess C_A (mol/m^3): ");
            if (scanf("%lf", &guess.C_A) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter initial guess T (K): ");
            if (scanf("%lf", &guess.T) != 1) { fprintf(stderr, "Bad input.\n"); continue; }
            printf("Enter sweep output CSV filename: ");
            if (scanf("%255s", fname) != 1) { fprintf(stderr, "Bad filename.\n"); continue; }

            sweep_feed_temperature(&params, T_start, T_end, n_steps,
                                   guess.C_A, guess.T, fname);
        } else {
            printf("Unknown choice.\n");
        }
    }

    printf("Exiting CSTR simulator.\n");
    return 0;
}
