/*
 * Standalone verification harness: the intEOSsolid() Berman/Vinet branches
 * and the CP_BERMAN/CP_SAXENA general-case reference-state formulas,
 * copied VERBATIM (variable names, formulas, comments trimmed only) from
 * MAGMA sources/gibbs.c lines 449-534 (intEOSsolid) and 2474-2535 (the
 * generic Cp/H/S branch), compiled standalone so we can cross-check the
 * Python translation in melts_vec against the real C arithmetic without
 * needing the rest of the MELTS library (silmin.h structs, threading,
 * XML I/O, etc.) or a working alphaMELTS binary.
 *
 * Build:  gcc -O0 -lm -o verify_eos verify_eos.c
 * Run:    ./verify_eos > verify_eos_output.tsv
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#define SQUARE(x) ((x)*(x))
#define CUBE(x)   ((x)*(x)*(x))
#define QUARTIC(x) ((x)*(x)*(x)*(x))

static const double tr = 298.15;
static const double pr = 1.0;

/* ---- intEOSsolid(), verbatim from gibbs.c lines 449-533 ---------------- */
typedef struct { int type; /* 0=BERMAN, 1=VINET */
                 double v0, v1, v2, v3, v4;           /* Berman */
                 double alpha, K, Kp;                  /* Vinet  */
} EosParams;

static void intEOSsolid(EosParams *phase, double t, double p, double *g,
                        double *h, double *s, double *cp, double *dcpdt, double *v, double *dvdt,
                        double *dvdp, double *d2vdt2, double *d2vdtdp, double *d2vdp2)
{
    if (phase->type == 0) {  /* EOS_BERMAN */
        double v0 = phase->v0;
        double v1 = phase->v1;
        double v2 = phase->v2;
        double v3 = phase->v3;
        double v4 = phase->v4;

        *g       += v0*((v1/2.0-v2)*(p*p-pr*pr)+v2*(p*p*p-pr*pr*pr)/3.0 + (1.0-v1+v2+v3*(t-tr)+v4*SQUARE(t-tr))*(p-pr));
        *h       += v0*((v1/2.0-v2)*(p*p-pr*pr)+v2*(p*p*p-pr*pr*pr)/3.0 + (1.0-v1+v2+v3*(t-tr)+v4*SQUARE(t-tr))*(p-pr)) - t*v0*(v3 + 2.0*(t-tr)*v4)*(p-pr);
        *s       += -v0*(v3 + 2.0*(t-tr)*v4)*(p-pr);
        *cp      += -t*v0*2.0*v4*(p-pr);
        *dcpdt   += -v0*2.0*v4*(p-pr);
        *v       += v0*(1.0 + (p-pr)*v1 + (p-pr)*(p-pr)*v2 + (t-tr)*v3 + (t-tr)*(t-tr)*v4);
        *dvdt    += v0*(v3 + 2.0*(t-tr)*v4);
        *dvdp    += v0*(v1 + 2.0*(p-pr)*v2);
        *d2vdt2  += v0*2.0*v4;
        *d2vdtdp += 0.0;
        *d2vdp2  += v0*2.0*v2;
    } else if (phase->type == 1) {  /* EOS_VINET */
        double v0    = phase->v0;
        double alpha = phase->alpha;
        double K     = phase->K;
        double Kp    = phase->Kp;
        double eta   = 3.0*(Kp-1.0)/2.0;
        double x     = 1.0;
        double x0    = 1.0;
        double fn, dfn, a, dxdt, dxdp, d2xdt2, d2xdtdp, d2xdp2, dx0dt, d2x0dt2;
        int iter;

        iter = 0;
        do {
            fn = x*x*(p/10000.0) - 3.0*K*(1.0-x)*exp(eta*(1.0-x)) - x*x*alpha*K*(t-tr);
            dfn = 2.0*x*(p/10000.0) + 3.0*K*(1.0+eta*(1.0-x))*exp(eta*(1.0-x)) - 2.0*x*alpha*K*(t-tr);
            x = x - fn/dfn;
            iter++;
        } while ((iter < 500) && (fn*fn > DBL_EPSILON));
        dxdt    = -(1.0/3.0)*x*x*x*alpha*K/(K*exp(eta*(1.0-x))*(-2.0+x-eta*x+eta*x*x));
        d2xdt2  = -dxdt*(
                         2.0*(p/10000.0)*dxdt-6.0*K*eta*dxdt*exp(eta*(1.0-x))-3.0*K*eta*eta*dxdt*exp(eta*(1.0-x))+3.0*K*eta*eta*x*dxdt*exp(eta*(1.0-x))
                         -4.0*alpha*K*x-2.0*alpha*K*dxdt*(t-tr))/(2.0*(p/10000.0)*x+3.0*K*exp(eta*(1.0-x))+3.0*K*eta*exp(eta*(1.0-x))
                                                                  -3.0*K*eta*x*exp(eta*(1.0-x))-2.0*alpha*K*x*(t-tr));
        dxdp    = -(1.0/3.0)*x*x*x/(K*exp(eta*(1.0-x))*(2.0-x+eta*x-eta*x*x));
        d2xdtdp = -(2.0*x*dxdt+2.0*(p/10000.0)*dxdp*dxdt-2.0*alpha*K*dxdp*dxdt*(t-tr)-3.0*K*eta*eta*dxdt*dxdp*exp(eta*(1.0-x))
                    -6.0*K*dxdt*eta*dxdp*exp(eta*(1.0-x))-2.0*alpha*K*x*dxdp+3.0*K*eta*eta*dxdt*dxdp*exp(eta*(1.0-x))*x)/
        (2.0*(p/10000.0)*x+3.0*K*exp(eta*(1.0-x))+3.0*K*eta*exp(eta*(1.0-x))-3.0*K*eta*exp(eta*(1.0-x))*x-2.0*alpha*K*x*(t-tr));
        d2xdp2  = - dxdp*dxdp*((-6.0+2.0*x-4.0*eta*x+2.0*eta*x*x-eta*eta*x*x+eta*eta*x*x*x)*exp(eta*(1.0-x)))/(x*exp(eta*(1.0-x))*(2.0-x+eta*x-eta*x*x));

        iter = 0;
        do {
            fn = x0*x0*(pr/10000.0) - 3.0*K*(1.0-x0)*exp(eta*(1.0-x0)) - x0*x0*alpha*K*(t-tr);
            dfn = 2.0*x0*(pr/10000.0) + 3.0*K*(1.0+eta*(1.0-x0))*exp(eta*(1.0-x0)) - 2.0*x0*alpha*K*(t-tr);
            x0 = x0 - fn/dfn;
            iter++;
        } while ((iter < 500) && (fn*fn > DBL_EPSILON));
        dx0dt    = -(1.0/3.0)*x0*x0*x0*alpha*K/(K*exp(eta*(1.0-x0))*(-2.0+x0-eta*x0+eta*x0*x0));
        d2x0dt2  = -dx0dt*(
                           2.0*(pr/10000.0)*dx0dt-6.0*K*eta*dx0dt*exp(eta*(1.0-x0))-3.0*K*eta*eta*dx0dt*exp(eta*(1.0-x0))+3.0*K*eta*eta*x0*dx0dt*exp(eta*(1.0-x0))
                           -4.0*alpha*K*x0-2.0*alpha*K*dx0dt*(t-tr))/(2.0*(pr/10000.0)*x0+3.0*K*exp(eta*(1.0-x0))+3.0*K*eta*exp(eta*(1.0-x0))
                                                                      -3.0*K*eta*x0*exp(eta*(1.0-x0))-2.0*alpha*K*x0*(t-tr));

        a  = (9.0*v0*K/(eta*eta))*(1.0 - eta*(1.0-x))*exp(eta*(1.0-x));
        a += v0*(t-tr)*K*alpha*(x*x*x - 1.0) - 9.0*v0*K/(eta*eta);

        a -= (9.0*v0*K/(eta*eta))*(1.0 - eta*(1.0-x0))*exp(eta*(1.0-x0));
        a -= v0*(t-tr)*K*alpha*(x0*x0*x0 - 1.0) - 9.0*v0*K/(eta*eta);

        *g       += -a*10000.0 + p*v0*x*x*x - pr*v0*x0*x0*x0;
        *h       += -a*10000.0 + p*v0*x*x*x - pr*v0*x0*x0*x0 + 10000.0*t*alpha*K*v0*(x*x*x-x0*x0*x0);
        *s       += 10000.0*alpha*K*v0*(x*x*x-x0*x0*x0);
        *cp      += 10000.0*t*alpha*K*v0*3.0*(x*x*dxdt - x0*x0*dx0dt);
        *dcpdt   += 10000.0*alpha*K*v0*3.0*(x*x*dxdt - x0*x0*dx0dt)
        + 10000.0*t*alpha*K*v0*3.0*(2.0*x*dxdt*dxdt + x*x*d2xdt2 - 2.0*x0*dx0dt*dx0dt - x0*x0*d2x0dt2);
        *v       += v0*x*x*x;
        *dvdt    += 3.0*v0*x*x*dxdt;
        *dvdp    += 3.0*v0*x*x*dxdp/10000.0;
        *d2vdt2  += 3.0*v0*(2.0*x*dxdt*dxdt + x*x*d2xdt2);
        *d2vdtdp += 3.0*v0*(2.0*x*dxdt*dxdp + x*x*d2xdtdp)/10000.0;
        *d2vdp2  += 3.0*v0*(2.0*x*dxdp*dxdp + x*x*d2xdp2)/(10000.0*10000.0);
    }
}

/* ---- CP_BERMAN / CP_SAXENA general branch, verbatim from gibbs.c
        lines 2474-2535 (condensed to just the two Cp branches + gs=hs-t*ss) --*/
static void solidGibbs(const char *cp_type, double href, double sref,
                        double k0, double k1, double k2, double k3,
                        double cp_t, double cp_h, double l1, double l2,
                        EosParams *eos, double t, double p,
                        double *G, double *H, double *S, double *Cp, double *dCpdT, double *V,
                        double *dVdT, double *dVdP, double *d2VdT2, double *d2VdTdP, double *d2VdP2)
{
    double hs = href, ss = sref, cps = 0.0, dcpsdt = 0.0, gs;
    double vs = 0.0, dvsdt = 0.0, dvsdp = 0.0, d2vsdt2 = 0.0, d2vsdtdp = 0.0, d2vsdp2 = 0.0;

    if (strcmp(cp_type, "CP_BERMAN") == 0) {
        cps = k0 + k1/sqrt(t) + k2/SQUARE(t) + k3/CUBE(t);
        dcpsdt = - 0.5*k1/pow(t, (double) 1.5) - 2.0*k2/CUBE(t) - 3.0*k3/QUARTIC(t);

        hs = hs + k0*(t-tr) + 2.0*k1*(sqrt(t)-sqrt(tr))
        - k2*(1.0/t-1.0/tr) - 0.5*k3*(1.0/(t*t)-1.0/(tr*tr));
        ss = ss + k0*log(t/tr)
        - 2.0*k1*(1.0/sqrt(t)-1.0/sqrt(tr))
        - 0.5*k2*(1.0/(t*t)-1.0/(tr*tr))
        - (1.0/3.0)*k3*(1.0/(t*t*t)-1.0/(tr*tr*tr));
        if (cp_t != 0.0) {
            if (t > cp_t) {
                hs = hs + cp_h
                + 0.5*l1*l1*(cp_t*cp_t-tr*tr)
                + (2.0/3.0)*l1*l2*(cp_t*cp_t*cp_t-tr*tr*tr)
                + 0.25*l2*l2*(cp_t*cp_t*cp_t*cp_t-tr*tr*tr*tr);
                ss = ss + cp_h/cp_t
                + l1*l1*(cp_t-tr)
                + l1*l2*(cp_t*cp_t-tr*tr)
                + (1.0/3.0)*l2*l2*(cp_t*cp_t*cp_t-tr*tr*tr);
            } else {
                hs = hs + 0.5*l1*l1*(t*t-tr*tr)
                + (2.0/3.0)*l1*l2*(t*t*t-tr*tr*tr)
                + 0.25*l2*l2*(t*t*t*t-tr*tr*tr*tr);
                ss = ss + l1*l1*(t-tr)
                + l1*l2*(t*t-tr*tr)
                + (1.0/3.0)*l2*l2*(t*t*t-tr*tr*tr);
                cps += t*SQUARE(l1+l2*t);
                dcpsdt += SQUARE(l1+l2*t) + t*2.0*(l1+l2*t)*l2;
            }
        }
    } else if (strcmp(cp_type, "CP_SAXENA") == 0) {
        double aCp=k0, bCp=k1, cCp=k2, dCp=k3, eCp=cp_t, gCp=cp_h, hCp=l1;  /* repurpose args */
        cps    = aCp + bCp*t + cCp/SQUARE(t) + dCp*SQUARE(t) + eCp/CUBE(t) + gCp/sqrt(t) + hCp/t;
        hs     = hs + aCp*(t-tr) + (bCp/2.0)*(SQUARE(t)-SQUARE(tr)) - cCp*(1.0/t-1.0/tr) + (dCp/3.0)*(CUBE(t)-CUBE(tr))
        - (eCp/2.0)*(1.0/SQUARE(t)-1.0/SQUARE(tr)) + 2.0*gCp*(sqrt(t)-sqrt(tr)) + hCp*log(t/tr);
        ss     = ss + aCp*log(t/tr) + bCp*(t-tr) - (cCp/2.0)*(1.0/SQUARE(t)-1.0/SQUARE(tr)) + (dCp/2.0)*(SQUARE(t)-SQUARE(tr))
        - (eCp/3.0)*(1.0/CUBE(t)-1.0/CUBE(tr)) - 2.0*gCp*(1.0/sqrt(t)-1.0/sqrt(tr)) - hCp*(1.0/t-1.0/tr);
        dcpsdt = bCp - 2.0*cCp/CUBE(t) + 2.0*dCp*t - 3.0*eCp/QUARTIC(t) - (gCp/2.0)/pow(t, (double) 3.0/2.0) - hCp/SQUARE(t);
    }

    gs = hs - t*ss;
    intEOSsolid(eos, t, p, &gs, &hs, &ss, &cps, &dcpsdt, &vs, &dvsdt, &dvsdp, &d2vsdt2, &d2vsdtdp, &d2vsdp2);

    *G = gs; *H = hs; *S = ss; *Cp = cps; *dCpdT = dcpsdt;
    *V = vs; *dVdT = dvsdt; *dVdP = dvsdp; *d2VdT2 = d2vsdt2; *d2VdTdP = d2vsdtdp; *d2VdP2 = d2vsdp2;
}

int main(void) {
    /* Endmembers: label, href, sref, k0..k3, Berman v1..v4 OR Vinet alpha/K/Kp */
    /* Values below were regenerated programmatically from
     * MELTS_Parameters/sol_struct_data.json (meltsSolids table) rather than
     * hand-typed, to eliminate transcription error as a variable in this
     * verification. */
    struct { const char *label; double href, sref, k0,k1,k2,k3;
             int eos_is_vinet; double e1,e2,e3,e4; } tab[] = {
        {"forsterite", -2174420.0, 94.01, 238.64, -2001.3, 0.0, -116240000.0,
            0, -7.91e-07, 1.351e-12, 2.9464e-05, 8.8633e-09},
        {"fayalite_berman", -1479360.0, 150.93, 248.93, -1923.9, 0.0, -139100000.0,
            0, -7.3e-07, 0.0, 2.6546e-05, 7.9482e-09},
        {"fayalite_vinet", -1479360.0, 150.93, 248.93, -1923.9, 0.0, -139100000.0,
            1, 2.9447e-05, 127.955797, 5.016956, 0.0},
        {"diopside", -3200583.0, 142.5, 305.41, -1604.9, -7166000.0, 921840000.0,
            0, -8.72e-07, 1.707e-12, 2.7795e-05, 8.3082e-09},
    };
    int ntab = sizeof(tab)/sizeof(tab[0]);

    double Ts[] = {1000.0, 1500.0, 2000.0};
    double Ps[] = {1.0, 10000.0, 50000.0};

    printf("label\tT\tP\tG\tH\tS\tCp\tV\tdVdT\tdVdP\n");
    for (int i = 0; i < ntab; i++) {
        for (int b = 0; b < 3; b++) {
            double t = Ts[b], p = Ps[b];
            EosParams eos;
            memset(&eos, 0, sizeof(eos));
            eos.v0 = 0;  /* set below */
            double G,H,S,Cp,dCpdT,V,dVdT,dVdP,d2VdT2,d2VdTdP,d2VdP2;

            /* v0 needs to come from the same table used in Python (sol_struct_data.json) */
            double v0;
            if (strcmp(tab[i].label, "forsterite")==0) v0 = 4.366;
            else if (strncmp(tab[i].label, "fayalite", 8)==0) v0 = 4.630;
            else if (strcmp(tab[i].label, "diopside")==0) v0 = 6.620;
            else v0 = 0.0;

            eos.v0 = v0;
            if (tab[i].eos_is_vinet) {
                eos.type = 1; eos.alpha = tab[i].e1; eos.K = tab[i].e2; eos.Kp = tab[i].e3;
            } else {
                eos.type = 0; eos.v1 = tab[i].e1; eos.v2 = tab[i].e2; eos.v3 = tab[i].e3; eos.v4 = tab[i].e4;
            }

            solidGibbs("CP_BERMAN", tab[i].href, tab[i].sref,
                       tab[i].k0, tab[i].k1, tab[i].k2, tab[i].k3,
                       0.0, 0.0, 0.0, 0.0,
                       &eos, t, p,
                       &G,&H,&S,&Cp,&dCpdT,&V,&dVdT,&dVdP,&d2VdT2,&d2VdTdP,&d2VdP2);

            printf("%s\t%.1f\t%.1f\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\n",
                   tab[i].label, t, p, G, H, S, Cp, V, dVdT, dVdP);
        }
    }
    return 0;
}
