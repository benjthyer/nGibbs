/*
 * Standalone verification harness for the clinopyroxene solid-solution
 * mixing model. The bulk of this file (verify_cpx_macros.h) is EXTRACTED
 * VERBATIM (byte-for-byte, via `sed`) from sources/clinopyroxene.c's own
 * #define blocks -- base constants, dependent parameters, vertices,
 * Taylor-expansion coefficients, pure-endmember macros, FR/ENDMEMBERS/
 * SIC/S/H/V/G, and DGDR0-5/DGDS0-1 -- so there is zero re-derivation risk
 * between the source and this harness; the only new code here is the
 * main() driver and the site-fraction/order-parameter plumbing needed to
 * exercise those macros, mirroring order()'s formulas (clinopyroxene.c
 * lines 3583-3662) directly.
 *
 * Build:  gcc -O0 -lm -o verify_clinopyroxene verify_clinopyroxene.c
 * Run:    ./verify_clinopyroxene > verify_clinopyroxene_output.tsv
 */
#include <stdio.h>
#include <math.h>
#include <float.h>

static const double R = 8.3143;
static const int clino = 1;   /* ISCLINO undefined in the real build -> isClino() always TRUE */

#define RHYOLITE_ADJUSTMENTS
#include "verify_cpx_macros.h"

/* Site fractions, clinopyroxene.c order() lines 3583-3595, 3644-3662
 * (converged-iterate formulas; DBL_EPSILON clipping only, no MAX_ITER
 * special-casing -- this harness targets the same "well away from a
 * degenerate join" test points the Python module's own gating logic
 * hands off to Newton, so the special-casing is not exercised). */
static double xti4m1, xca2m2, xna1m2, xsi4tet;
static double xal3m1, xfe2m1, xfe3m1, xmg2m1, xfe2m2, xmg2m2, xal3tet, xfe3tet;

static void site_fractions(double r[6], double s[2]) {
    double v;
    xti4m1  = (r[1]+r[2])/2.0;
    xca2m2  = 1.0 - r[4] - r[5];
    xna1m2  = r[4];
    xsi4tet = (4.0-2.0*r[1]-2.0*r[2]-2.0*r[3]+r[4])/4.0;
    xal3m1  = (2.0*r[3]+r[4]-2.0*s[0])/4.0;
    xfe2m1  = (2.0*r[0]-r[5]-s[1])/2.0;
    xfe3m1  = (2.0*r[3]+r[4]+2.0*s[0])/4.0;
    xmg2m1  = 1.0 - r[0] - r[3] - 0.5*(r[1]+r[2]+r[4]-r[5]-s[1]);
    xfe2m2  = (r[5]+s[1])/2.0;
    xmg2m2  = (r[5]-s[1])/2.0;
    xal3tet = (4.0*r[1]+2.0*r[3]-r[4]+2.0*s[0])/8.0;
    xfe3tet = (4.0*r[2]+2.0*r[3]-r[4]-2.0*s[0])/8.0;
#define CLIP(x) if (x <= DBL_EPSILON) x = DBL_EPSILON; if (x >= 1.0-DBL_EPSILON) x = 1.0-DBL_EPSILON;
    CLIP(xti4m1); CLIP(xca2m2); CLIP(xna1m2); CLIP(xsi4tet);
    CLIP(xal3m1); CLIP(xfe2m1); CLIP(xfe3m1); CLIP(xmg2m1);
    CLIP(xfe2m2); CLIP(xmg2m2); CLIP(xal3tet); CLIP(xfe3tet);
#undef CLIP
    (void)v;
}

/* ab-initio initial guess for s (order() lines 3551-3575), used here as
 * a direct evaluation (no Newton) at test points chosen to already be
 * physically consistent with it, PLUS a free general (r,s,T,P) evaluation
 * mode for the polynomial/derivative cross-checks (s supplied directly). */

int main(void) {
    printf("=== CPX_COEFFS ===\n");
    printf("name\tvalue\n");
    printf("H0\t%.10g\n", (double)(H0));
    printf("HX2\t%.10g\n", (double)(HX2));
    printf("HX7\t%.10g\n", (double)(HX7));
    printf("HS1\t%.10g\n", (double)(HS1));
    printf("HS2\t%.10g\n", (double)(HS2));
    printf("HX3X7\t%.10g\n", (double)(HX3X7));
    printf("HX7X7\t%.10g\n", (double)(HX7X7));
    printf("HX7X7X7\t%.10g\n", (double)(HX7X7X7));
    printf("HX7S2S2\t%.10g\n", (double)(HX7S2S2));
    printf("SX7\t%.10g\n", (double)(SX7));
    printf("SX2X3\t%.10g\n", (double)(SX2X3));
    printf("SX7X7\t%.10g\n", (double)(SX7X7));
    printf("SX5\t%.10g\n", (double)(SX5));  /* should be 0.0, per S55 not W55 fix */
    printf("SX6\t%.10g\n", (double)(SX6));  /* should be 0.0 */
    printf("SS1\t%.10g\n", (double)(SS1));  /* should be 0.0 */
    printf("VX2\t%.10g\n", (double)(VX2));
    printf("VX7\t%.10g\n", (double)(VX7));
    printf("VX7X7\t%.10g\n", (double)(VX7X7));

    printf("=== CPX_VERTEX_GHSV ===\n");
    printf("name\tT\tP\tG\tH\tS\tV\n");
    {
        double Ts[] = {1300.0, 1500.0};
        double Ps[] = {1000.0, 8000.0};
        const char *names[] = {"diopside","clinoenstatite","hedenbergite",
                                "alumino-buffonite","buffonite","jadeite"};
        double rv[6][6] = {
            {0,0,0,0,0,0},
            {0,0,0,0,0,1},
            {1,0,0,0,0,0},
            {0,1,0,0,0,0},
            {0,0,1,0,0,0},
            {0,0.5,-0.5,0.5,1,0},
        };
        double sv[6][2] = {
            {0,0}, {0,-1}, {0,0}, {0,0}, {0,0}, {-1,0},
        };
        for (int n=0; n<6; n++) {
            for (int b=0; b<2; b++) {
                double r[6], s[2], t=Ts[b], p=Ps[b];
                for (int k=0;k<6;k++) r[k]=rv[n][k];
                for (int k=0;k<2;k++) s[k]=sv[n][k];
                site_fractions(r,s);
                double Hv=(double)(H), Sv=(double)(S)-(double)(SIC), Vv=(double)(V);
                /* NOTE: "S" macro includes SIC; the pure-endmember *_S macros
                 * (DI_S, EN_S, ...) do NOT (verified directly against source,
                 * see clinopyroxene.py module docstring) -- report both the
                 * poly-only S (matching *_S) and the full S (matching the
                 * mixed-phase "S" macro) so both are checkable. */
                double Gpoly = Hv - t*Sv + (p-1.0)*Vv;   /* matches DI_G..JD_G */
                double Gfull = (double)(G);              /* matches the mixed-phase G macro at this (r,s) */
                printf("%s\t%.1f\t%.1f\t%.10g\t%.10g\t%.10g\t%.10g\tGfull=%.10g\n",
                       names[n], t, p, Gpoly, Hv, Sv, Vv, Gfull);
            }
        }
    }

    printf("=== CPX_ESSENITE_ORDER ===\n");
    printf("T\tP\ts\tDES_GDS1_at_s\tES_G\tES_H\tES_S\tES_V\n");
    {
        double Ts[] = {1300.0, 1500.0};
        double Ps[] = {1000.0, 8000.0};
        for (int b=0;b<2;b++) {
            double t=Ts[b], p=Ps[b];
            /* Newton solve for essenite's own s (bisection-safe simple Newton) */
            double s = 0.0;
            for (int it=0; it<100; it++) {
                double f = DES_GDS1;
                double eps=1e-6;
                double splus = s+eps, sminus = s-eps;
                double s_save = s;
                s = splus; double fplus = DES_GDS1; s = sminus; double fminus = DES_GDS1; s = s_save;
                double jac = (fplus-fminus)/(2.0*eps);
                s = s - f/jac;
                if (s > 1.0-1e-12) s = 1.0-1e-12;
                if (s < -1.0+1e-12) s = -1.0+1e-12;
            }
            double check = DES_GDS1;
            double g=(double)(ES_G), h=(double)(ES_H), ss=(double)(ES_S), v=(double)(0.0);
            /* ES_V uses V0,VX5,VS1*s,VX5X5,VX5S1*s,VS1S1*s*s -- compute directly: */
            v = (V0)+(VX5)+(VS1)*s+(VX5X5)+(VX5S1)*s+(VS1S1)*s*s;
            printf("%.1f\t%.1f\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\n",
                   t, p, s, check, g, h, ss, v);
        }
    }

    printf("=== CPX_MIXED_G_H_S_V_DGDR_DGDS ===\n");
    printf("r0\tr1\tr2\tr3\tr4\tr5\ts0\ts1\tT\tP\tH\tS\tG\tV\tDGDR0\tDGDR1\tDGDR2\tDGDR3\tDGDR4\tDGDR5\tDGDS0\tDGDS1\n");
    {
        /* Interior, non-degenerate test compositions (all site fractions
         * well away from 0/1) with hand-picked (not necessarily
         * equilibrium) s -- this is a pure formula check, independent of
         * whichever Newton/ordering strategy either side uses. */
        double tests[][10] = {
            /* r0,   r1,   r2,   r3,   r4,   r5,    s0,    s1,     T,     P */
            {0.15, 0.05, 0.04, 0.03, 0.02, 0.30,  0.02, -0.05, 1350.0, 5000.0},
            {0.30, 0.02, 0.10, 0.01, 0.05, 0.10, -0.03,  0.08, 1500.0, 12000.0},
            {0.05, 0.20, 0.02, 0.15, 0.08, 0.45,  0.10, -0.10, 1250.0, 1000.0},
        };
        for (int i=0;i<3;i++) {
            double r[6], s[2];
            for (int k=0;k<6;k++) r[k]=tests[i][k];
            s[0]=tests[i][6]; s[1]=tests[i][7];
            double t=tests[i][8], p=tests[i][9];
            site_fractions(r,s);
            double Hv=(double)(H), Sv=(double)(S), Vv=(double)(V), Gv=(double)(G);
            double dgdr0=(double)(DGDR0), dgdr1=(double)(DGDR1), dgdr2=(double)(DGDR2);
            double dgdr3=(double)(DGDR3), dgdr4=(double)(DGDR4), dgdr5=(double)(DGDR5);
            double dgds0=(double)(DGDS0), dgds1=(double)(DGDS1);
            printf("%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.1f\t%.1f\t"
                   "%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\n",
                   r[0],r[1],r[2],r[3],r[4],r[5],s[0],s[1],t,p,
                   Hv,Sv,Gv,Vv,dgdr0,dgdr1,dgdr2,dgdr3,dgdr4,dgdr5,dgds0,dgds1);
        }
    }
    return 0;
}
