/* Standalone C harness verifying melts_vec.liquid_eos.kress_component
 * against formulas transcribed verbatim from MAGMA's sources/gibbs.c
 * (the generic "all other components, MELTS, pMELTS, XMELTS" liquid
 * branch, ~lines 1278-1348, t > tglass sub-branch only -- see
 * liquid_eos.py's module docstring for why that's the only live path
 * for meltsLiquid).
 *
 * Three components are checked: SiO2 and Al2O3 (plain Berman solid
 * reference, no lambda transition) and KAlSiO4 (which DOES have a
 * nonzero order-disorder lambda correction on its solid reference phase,
 * with Tt = 800.15 K < Tfusion = 2023.15 K -- this is the case that an
 * earlier draft of liquid_eos.py silently dropped by not passing
 * cp_t/cp_h/l1/l2 through to berman_ref_state; this harness exists
 * specifically to catch that class of bug).
 *
 * Build & run:
 *   gcc -O0 -std=c99 -o verify_liquid verify_liquid.c -lm && ./verify_liquid
 */
#include <stdio.h>
#include <math.h>

#define SQUARE(x) ((x)*(x))
#define CUBE(x)   ((x)*(x)*(x))
#define QUARTIC(x) ((x)*(x)*(x)*(x))

static const double R   = 8.3143;
static const double Tr  = 298.15;
static const double Pr  = 1.0;
static const double Trl = 1673.0;

typedef struct {
    const char *label;
    double ref_h, ref_s, k0, k1, k2, k3, cp_t, cp_h, l1, l2;
    double v_liq, dvdt, dvdp, d2vdtp, d2vdp2;
    double t_fusion, s_fusion, cp_liquid;
} Comp;

/* Full CP_BERMAN solid reference-state H,S at (T, Pr), with the optional
 * lambda transition -- verbatim from gibbs.c's CP_BERMAN branch
 * (already validated bit-for-bit against melts_vec.thermal in the prior
 * session's benchmark_test.py; reproduced here so this harness is
 * self-contained). */
static void berman_hs(double T, double h_ref, double s_ref,
                       double k0, double k1, double k2, double k3,
                       double cp_t, double cp_h, double l1, double l2,
                       double *H, double *S) {
    double h = h_ref + k0*(T-Tr) + 2.0*k1*(sqrt(T)-sqrt(Tr))
               - k2*(1.0/T - 1.0/Tr) - 0.5*k3*(1.0/SQUARE(T) - 1.0/SQUARE(Tr));
    double s = s_ref + k0*log(T/Tr)
               - 2.0*k1*(1.0/sqrt(T) - 1.0/sqrt(Tr))
               - 0.5*k2*(1.0/SQUARE(T) - 1.0/SQUARE(Tr))
               - (1.0/3.0)*k3*(1.0/CUBE(T) - 1.0/CUBE(Tr));
    if (cp_t != 0.0) {
        if (T > cp_t) {
            h += cp_h + 0.5*l1*l1*(cp_t*cp_t - Tr*Tr)
                 + (2.0/3.0)*l1*l2*(CUBE(cp_t) - CUBE(Tr))
                 + 0.25*l2*l2*(QUARTIC(cp_t) - QUARTIC(Tr));
            s += cp_h/cp_t + l1*l1*(cp_t-Tr) + l1*l2*(cp_t*cp_t-Tr*Tr)
                 + (1.0/3.0)*l2*l2*(CUBE(cp_t)-CUBE(Tr));
        } else {
            h += 0.5*l1*l1*(T*T-Tr*Tr) + (2.0/3.0)*l1*l2*(CUBE(T)-CUBE(Tr))
                 + 0.25*l2*l2*(QUARTIC(T)-QUARTIC(Tr));
            s += l1*l1*(T-Tr) + l1*l2*(T*T-Tr*Tr) + (1.0/3.0)*l2*l2*(CUBE(T)-CUBE(Tr));
        }
    }
    *H = h; *S = s;
}

static void liquid_props(const Comp *c, double T, double P,
                          double *G, double *H, double *S, double *Cp,
                          double *V, double *dVdT, double *dVdP) {
    /* fusion state: gibbs(tfus, pr, ...) on the SOLID reference phase */
    double fus_h, fus_s;
    berman_hs(c->t_fusion, c->ref_h, c->ref_s, c->k0, c->k1, c->k2, c->k3,
              c->cp_t, c->cp_h, c->l1, c->l2, &fus_h, &fus_s);
    fus_h += c->t_fusion * c->s_fusion;
    fus_s += c->s_fusion;

    /* t > tglass branch (tglass = 0 for all meltsLiquid) */
    double hl = fus_h + c->cp_liquid*(T - c->t_fusion);
    double sl = fus_s + c->cp_liquid*log(T/c->t_fusion);
    double cpl = c->cp_liquid;

    double gl = hl - T*sl
        + (c->v_liq + c->dvdt*(T-Trl))*(P-Pr)
        + 0.5*(c->dvdp + (T-Trl)*c->d2vdtp)*(P*P-Pr*Pr)
        - (c->dvdp + (T-Trl)*c->d2vdtp)*Pr*(P-Pr)
        + c->d2vdp2*( (CUBE(P)-CUBE(Pr))/6.0 - Pr*(P*P-Pr*Pr)/2.0 + Pr*Pr*(P-Pr)/2.0 );
    hl += (c->v_liq + c->dvdt*(T-Trl))*(P-Pr)
        + 0.5*(c->dvdp + (T-Trl)*c->d2vdtp)*(P*P-Pr*Pr)
        - (c->dvdp + (T-Trl)*c->d2vdtp)*Pr*(P-Pr)
        + c->d2vdp2*( (CUBE(P)-CUBE(Pr))/6.0 - Pr*(P*P-Pr*Pr)/2.0 + Pr*Pr*(P-Pr)/2.0 )
        - T*(c->dvdt*(P-Pr) + 0.5*c->d2vdtp*(P-Pr)*(P-Pr));
    sl += -(c->dvdt*(P-Pr) + 0.5*c->d2vdtp*(P-Pr)*(P-Pr));

    double vl = c->v_liq + c->dvdt*(T-Trl)
        + (c->dvdp + c->d2vdtp*(T-Trl))*(P-Pr)
        + c->d2vdp2*(0.5*P*P - Pr*(P-Pr));
    double dvldt = c->dvdt + c->d2vdtp*(P-Pr);
    double dvldp = c->dvdp + c->d2vdtp*(T-Trl) + c->d2vdp2*(P-Pr);

    *G = gl; *H = hl; *S = sl; *Cp = cpl; *V = vl; *dVdT = dvldt; *dVdP = dvldp;
}

int main(void) {
    Comp SiO2    = {"SiO2",    -906377.0, 46.029, 83.51, -374.7, -2455400.0, 280070000.0, 0,0,0,0,
                     2.69, 0.0, -1.89e-05, 1.3e-08, 3.6e-10, 1999.0, 4.46, 82.6};
    Comp Al2O3   = {"Al2O3",   -1675700.0, 50.82, 155.02, -828.4, -3861400.0, 409080000.0, 0,0,0,0,
                     3.711, 0.000262, -2.26e-05, 2.7e-08, 4.0e-10, 2319.65, 48.61, 170.3};
    Comp KAlSiO4 = {"KAlSiO4", -2111813.55, 133.9653, 186.0, 0.0, -13106700.0, 2138930000.0,
                     800.15, 1154.0, -0.07096454, 0.00021682,
                     6.8375, 0.0007265, -6.395e-05, -4.6e-08, 1.21e-09, 2023.15, 24.5, 217.0};

    Comp comps[3] = {SiO2, Al2O3, KAlSiO4};
    double Ts[2] = {1673.0, 1200.0};
    double Ps[2] = {1.0, 10000.0};

    printf("label\tT\tP\tG\tH\tS\tCp\tV\tdVdT\tdVdP\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            double T = Ts[j], P = Ps[j];
            double G,H,S,Cp,V,dVdT,dVdP;
            liquid_props(&comps[i], T, P, &G,&H,&S,&Cp,&V,&dVdT,&dVdP);
            printf("%s\t%.4f\t%.4f\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\n",
                   comps[i].label, T, P, G, H, S, Cp, V, dVdT, dVdP);
        }
    }
    return 0;
}
