/*
 * Standalone verification harness for the feldspar and olivine
 * solid-solution mixing models, mirroring verify_eos.c's approach:
 * formulas copied verbatim from MAGMA sources/feldspar.c and
 * sources/olivine.c, compiled standalone, cross-checked against the
 * Python melts_vec.feldspar / melts_vec.olivine translations.
 *
 * Build:  gcc -O0 -lm -o verify_solutions verify_solutions.c
 * Run:    ./verify_solutions > verify_solutions_output.tsv
 */
#include <stdio.h>
#include <math.h>

static const double R = 8.3143;

/* ======================= FELDSPAR ======================= */

static const double whabor = 18810.0, wsabor = 10.3, wvabor = 0.4602;
static const double whorab = 27320.0, wsorab = 10.3, wvorab = 0.3264;
static const double whaban = 7924.0,  whanab = 0.0;
static const double whoran = 40317.0, whanor = 38974.0, wvanor = -0.1037;
static const double whabanor = 12545.0, wvabanor = -1.095;

static void feldspar_gmix(double xab, double xan, double t, double p,
                           double *G, double *H, double *S, double *V) {
    double xor = 1.0 - xab - xan;
    double Sconf = -R*(xab*log(xab) + xan*log(xan) + xor*log(xor));
    double s = Sconf + wsabor*xab*xor*(xor+xan/2.0) + wsorab*xab*xor*(xab+xan/2.0);
    double h = whaban*xab*xan*(xan+xor/2.0) + whanab*xab*xan*(xab+xor/2.0)
             + whabor*xab*xor*(xor+xan/2.0) + whorab*xab*xor*(xab+xan/2.0)
             + whanor*xan*xor*(xor+xab/2.0) + whoran*xan*xor*(xan+xab/2.0)
             + whabanor*xab*xan*xor;
    double v = wvabor*xab*xor*(xor+xan/2.0) + wvorab*xab*xor*(xab+xan/2.0)
             + wvanor*xan*xor*(xor+xab/2.0) + wvabanor*xab*xan*xor;
    double g = h - t*s + (p-1.0)*v;
    *G = g; *H = h; *S = s; *V = v;
}

/* ======================= OLIVINE ======================= */
/* Base parameters verbatim from olivine.c lines 121-217 */
#define HEXMGMN  15.80e3
#define HEXMGFE  00.00e3
#define HEXMGCO -15.00e3
#define HEXMGNI -19.75e3
#define HEXMNFE -11.80e3
#define HEXMNCO  00.00e3
#define HEXMNNI  00.00e3
#define HEXFECO  00.00e3
#define HEXFENI -20.00e3
#define HEXCONI   0.00e3

#define VEXMGMN  0.0
#define VEXMGFE  0.0
#define VEXMGCO  0.0
#define VEXMGNI  0.0
#define VEXMNFE  0.0
#define VEXMNCO  0.0
#define VEXMNNI  0.0
#define VEXFECO  0.0
#define VEXFENI  0.0
#define VEXCONI  0.0

#define HXMGMN    8.75e3
#define HXMGFE   10.15e3
#define HXMGCO   03.00e3
#define HXMGNI    2.20e3
#define HXMNFE    0.50e3
#define HXMNCO    0.00e3
#define HXMNNI   00.00e3
#define HXFECO   03.00e3
#define HXFENI   10.00e3
#define HXCONI    0.00e3

#define VXMGMN   00.00
#define VXMGFE   00.015
#define VXMGCO   00.00
#define VXMGNI   00.000
#define VXMNFE   00.00
#define VXMNCO   00.00
#define VXMNNI   00.00
#define VXFECO   00.00
#define VXFENI   00.000
#define VXCONI   00.00

#define WH1MGMN    6.625e3
#define WH2MGMN    6.625e3
#define WH1MGFE    5.075e3
#define WH2MGFE    5.075e3
#define WH1MGCO    1.50e3
#define WH2MGCO    1.50e3
#define WH1MGNI    -.600e3
#define WH2MGNI    2.800e3
#define WH1MNFE    1.75e3
#define WH2MNFE    1.75e3
#define WH1MNCO    0.00e3
#define WH2MNCO    0.00e3
#define WH1MNNI    0.00e3
#define WH2MNNI    0.00e3
#define WH1FECO    1.50e3
#define WH2FECO    1.50e3
#define WH1FENI    5.000e3
#define WH2FENI    5.000e3
#define WH1CONI    0.00e3
#define WH2CONI    0.00e3

#define WV1MGMN   0.0
#define WV2MGMN   0.0
#define WV1MGFE   0.0
#define WV2MGFE   0.0
#define WV1MGCO   0.0
#define WV2MGCO   0.0
#define WV1MGNI   0.0
#define WV2MGNI   0.0
#define WV1MNFE   0.0
#define WV2MNFE   0.0
#define WV1MNCO   0.0
#define WV2MNCO   0.0
#define WV1MNNI   0.0
#define WV2MNNI   0.0
#define WV1FECO   0.0
#define WV2FECO   0.0
#define WV1FENI   0.0
#define WV2FENI   0.0
#define WV1CONI   0.0
#define WV2CONI   0.0

#define WH2CAMG   34.50e3
#define WH2CAMN   16.00e3
#define WH2CAFE   21.90e3
#define WH2CACO   30.00e3
#define WH2CANI   40.00e3

#define WV2CAMG   00.35
#define WV2CAMN   00.00
#define WV2CAFE   00.00
#define WV2CACO   00.00
#define WV2CANI   00.00

#define F_MN     09.50e3
#define F_FE     09.50e3
#define F_CO     00.00
#define F_NI     00.00

/* p is a function parameter in every macro below, matching olivine.c */
#define GEXMGMN  (HEXMGMN) + (p-1.0)*(VEXMGMN)
#define GEXMGFE  (HEXMGFE) + (p-1.0)*(VEXMGFE)
#define GEXMGCO  (HEXMGCO) + (p-1.0)*(VEXMGCO)
#define GEXMGNI  (HEXMGNI) + (p-1.0)*(VEXMGNI)
#define GEXMNFE  (HEXMNFE) + (p-1.0)*(VEXMNFE)
#define GEXMNCO  (HEXMNCO) + (p-1.0)*(VEXMNCO)
#define GEXMNNI  (HEXMNNI) + (p-1.0)*(VEXMNNI)
#define GEXFECO  (HEXFECO) + (p-1.0)*(VEXFECO)
#define GEXFENI  (HEXFENI) + (p-1.0)*(VEXFENI)
#define GEXCONI  (HEXCONI) + (p-1.0)*(VEXCONI)

#define GXMGMN  (HXMGMN) + (p-1.0)*(VXMGMN)
#define GXMGFE  (HXMGFE) + (p-1.0)*(VXMGFE)
#define GXMGCO  (HXMGCO) + (p-1.0)*(VXMGCO)
#define GXMGNI  (HXMGNI) + (p-1.0)*(VXMGNI)
#define GXMNFE  (HXMNFE) + (p-1.0)*(VXMNFE)
#define GXMNCO  (HXMNCO) + (p-1.0)*(VXMNCO)
#define GXMNNI  (HXMNNI) + (p-1.0)*(VXMNNI)
#define GXFECO  (HXFECO) + (p-1.0)*(VXFECO)
#define GXFENI  (HXFENI) + (p-1.0)*(VXFENI)
#define GXCONI  (HXCONI) + (p-1.0)*(VXCONI)

#define W1MGMN  (WH1MGMN) + (p-1.0)*(WV1MGMN)
#define W2MGMN  (WH2MGMN) + (p-1.0)*(WV2MGMN)
#define W1MGFE  (WH1MGFE) + (p-1.0)*(WV1MGFE)
#define W2MGFE  (WH2MGFE) + (p-1.0)*(WV2MGFE)
#define W1MGCO  (WH1MGCO) + (p-1.0)*(WV1MGCO)
#define W2MGCO  (WH2MGCO) + (p-1.0)*(WV2MGCO)
#define W1MGNI  (WH1MGNI) + (p-1.0)*(WV1MGNI)
#define W2MGNI  (WH2MGNI) + (p-1.0)*(WV2MGNI)
#define W1MNFE  (WH1MNFE) + (p-1.0)*(WV1MNFE)
#define W2MNFE  (WH2MNFE) + (p-1.0)*(WV2MNFE)
#define W1MNCO  (WH1MNCO) + (p-1.0)*(WV1MNCO)
#define W2MNCO  (WH2MNCO) + (p-1.0)*(WV2MNCO)
#define W1MNNI  (WH1MNNI) + (p-1.0)*(WV1MNNI)
#define W2MNNI  (WH2MNNI) + (p-1.0)*(WV2MNNI)
#define W1FECO  (WH1FECO) + (p-1.0)*(WV1FECO)
#define W2FECO  (WH2FECO) + (p-1.0)*(WV2FECO)
#define W1FENI  (WH1FENI) + (p-1.0)*(WV1FENI)
#define W2FENI  (WH2FENI) + (p-1.0)*(WV2FENI)
#define W1CONI  (WH1CONI) + (p-1.0)*(WV1CONI)
#define W2CONI  (WH2CONI) + (p-1.0)*(WV2CONI)

#define W2CAMG  (WH2CAMG) + (p-1.0)*(WV2CAMG)
#define W2CAMN  (WH2CAMN) + (p-1.0)*(WV2CAMN)
#define W2CAFE  (WH2CAFE) + (p-1.0)*(WV2CAFE)
#define W2CACO  (WH2CACO) + (p-1.0)*(WV2CACO)
#define W2CANI  (WH2CANI) + (p-1.0)*(WV2CANI)

#define G0  0.25*(     ((GXMNFE)+(W1MNFE)+(W2MNFE)) \
                                            +((GXMNCO)+(W1MNCO)+(W2MNCO)) \
                                            +((GXMNNI)+(W1MNNI)+(W2MNNI)) \
                                            +((GXFECO)+(W1FECO)+(W2FECO)) \
                                            +((GXFENI)+(W1FENI)+(W2FENI)) \
                                            +((GXCONI)+(W1CONI)+(W2CONI)) \
                                    -2.0*((GXMGMN)+(W1MGMN)+(W2MGMN)) \
                                    -2.0*((GXMGFE)+(W1MGFE)+(W2MGFE)) \
                                    -2.0*((GXMGCO)+(W1MGCO)+(W2MGCO)) \
                                    -2.0*((GXMGNI)+(W1MGNI)+(W2MGNI)))
#define GR1 0.25*(-3.0*((GXMGMN)+(W1MGMN)+(W2MGMN)) \
                                            -((GXMGFE)+(W1MGFE)+(W2MGFE)) \
                                            -((GXMGCO)+(W1MGCO)+(W2MGCO)) \
                                            -((GXMGNI)+(W1MGNI)+(W2MGNI)) \
                                            +((GXMNFE)+(W1MNFE)+(W2MNFE)) \
                                            +((GXMNCO)+(W1MNCO)+(W2MNCO)) \
                                            +((GXMNNI)+(W1MNNI)+(W2MNNI)))
#define GR2  0.25*(   -((GXMGMN)+(W1MGMN)+(W2MGMN)) \
                                    -3.0*((GXMGFE)+(W1MGFE)+(W2MGFE)) \
                                            -((GXMGCO)+(W1MGCO)+(W2MGCO)) \
                                            -((GXMGNI)+(W1MGNI)+(W2MGNI)) \
                                            +((GXMNFE)+(W1MNFE)+(W2MNFE)) \
                                            +((GXFECO)+(W1FECO)+(W2FECO)) \
                                            +((GXFENI)+(W1FENI)+(W2FENI)))
#define GR3  0.25*(   -((GXMGMN)+(W1MGMN)+(W2MGMN)) \
                                            -((GXMGFE)+(W1MGFE)+(W2MGFE)) \
                                    -3.0*((GXMGCO)+(W1MGCO)+(W2MGCO)) \
                                            -((GXMGNI)+(W1MGNI)+(W2MGNI)) \
                                            +((GXMNCO)+(W1MNCO)+(W2MNCO)) \
                                            +((GXFECO)+(W1FECO)+(W2FECO)) \
                                            +((GXCONI)+(W1CONI)+(W2CONI)))
#define GR4 0.25*(    -((GXMGMN)+(W1MGMN)+(W2MGMN)) \
                                            -((GXMGFE)+(W1MGFE)+(W2MGFE)) \
                                            -((GXMGCO)+(W1MGCO)+(W2MGCO)) \
                                    -3.0*((GXMGNI)+(W1MGNI)+(W2MGNI)) \
                                            +((GXMNNI)+(W1MNNI)+(W2MNNI)) \
                                            +((GXFENI)+(W1FENI)+(W2FENI)) \
                                            +((GXCONI)+(W1CONI)+(W2CONI)))
#define GR5        -(W2CAMG) \
                                    -0.25*(((F_MN)+(GEXMGMN)+(GXMGMN) \
                                                -2.0*(W2CAMN)+2.0*(W2MGMN)) \
                                                +((F_FE)+(GEXMGFE)+(GXMGFE) \
                                                -2.0*(W2CAFE)+2.0*(W2MGFE)) \
                                                +((F_CO)+(GEXMGCO)+(GXMGCO) \
                                                -2.0*(W2CACO)+2.0*(W2MGCO)) \
                                                +((F_NI)+(GEXMGNI)+(GXMGNI) \
                                                -2.0*(W2CANI)+2.0*(W2MGNI)))
#define GR1R5             -0.25*((F_MN)+(GEXMGMN)+(GXMGMN) \
                   +2.0*(W2CAMG)-2.0*(W2CAMN)+2.0*(W2MGMN))
#define GR2R5             -0.25*((F_FE)+(GEXMGFE)+(GXMGFE) \
                   +2.0*(W2CAMG)-2.0*(W2CAFE)+2.0*(W2MGFE))
#define GR3R5             -0.25*((F_CO)+(GEXMGCO)+(GXMGCO) \
                   +2.0*(W2CAMG)-2.0*(W2CACO)+2.0*(W2MGCO))
#define GS1   0.25*(( (GEXMGMN)+3.0*(W1MGMN)-3.0*(W2MGMN)) \
                                                        -((GEXMGFE)-(W1MGFE)+(W2MGFE)) \
                                                        -((GEXMGCO)-(W1MGCO)+(W2MGCO)) \
                                                        -((GEXMGNI)-(W1MGNI)+(W2MGNI)) \
                                                        +((GEXMNFE)-(W1MNFE)+(W2MNFE)) \
                                                        +((GEXMNCO)-(W1MNCO)+(W2MNCO)) \
                                                        +((GEXMNNI)-(W1MNNI)+(W2MNNI)))
#define GS2           0.25*((-(GEXMGMN)+(W1MGMN)-(W2MGMN)) \
                                        +((GEXMGFE)+3.0*(W1MGFE)-3.0*(W2MGFE)) \
                                                        -((GEXMGCO)-(W1MGCO)+(W2MGCO)) \
                                                        -((GEXMGNI)-(W1MGNI)+(W2MGNI)) \
                                                        -((GEXMNFE)+(W1MNFE)-(W2MNFE)) \
                                                        +((GEXFECO)-(W1FECO)+(W2FECO)) \
                                                        +((GEXFENI)-(W1FENI)+(W2FENI)))
#define GS3           0.25*(( (GEXMGMN)-(W1MGMN)+(W2MGMN)) \
                                                        +((GEXMGFE)-(W1MGFE)+(W2MGFE)) \
                                        -((GEXMGCO)+3.0*(W1MGCO)-3.0*(W2MGCO)) \
                                                        +((GEXMGNI)-(W1MGNI)+(W2MGNI)) \
                                                        +((GEXMNCO)+(W1MNCO)-(W2MNCO)) \
                                                        +((GEXFECO)+(W1FECO)-(W2FECO)) \
                                                        -((GEXCONI)-(W1CONI)+(W2CONI)))
#define GS4           0.25*(( (GEXMGMN)-(W1MGMN)+(W2MGMN)) \
                                                        +((GEXMGFE)-(W1MGFE)+(W2MGFE)) \
                                                        +((GEXMGCO)-(W1MGCO)+(W2MGCO)) \
                                        -((GEXMGNI)+3.0*(W1MGNI)-3.0*(W2MGNI)) \
                                                        +((GEXMNNI)+(W1MNNI)-(W2MNNI)) \
                                                        +((GEXFENI)+(W1FENI)-(W2FENI)) \
                                                        +((GEXCONI)+(W1CONI)-(W2CONI)))

/* Remaining quadratic G coefficients, verbatim olivine.c lines 359-459 */
#define GR1R1         -0.25*(  (GXMGMN)+(W1MGMN)+(W2MGMN))
#define GR1R2          0.25*(( (GXMNFE)+(W1MNFE)+(W2MNFE)) \
                             -((GXMGMN)+(W1MGMN)+(W2MGMN)) \
                             -((GXMGFE)+(W1MGFE)+(W2MGFE)))
#define GR1R3          0.25*(( (GXMNCO)+(W1MNCO)+(W2MNCO)) \
                             -((GXMGMN)+(W1MGMN)+(W2MGMN)) \
                             -((GXMGCO)+(W1MGCO)+(W2MGCO)))
#define GR1R4          0.25*(( (GXMNNI)+(W1MNNI)+(W2MNNI)) \
                             -((GXMGMN)+(W1MGMN)+(W2MGMN)) \
                             -((GXMGNI)+(W1MGNI)+(W2MGNI)))
#define GR1S1                       0.5*((W1MGMN)-(W2MGMN))
#define GR1S2         0.25*((-(GEXMGMN)-(W1MGMN)+(W2MGMN)) \
                                                        +((GEXMGFE)+(W1MGFE)-(W2MGFE)) \
                                                        -((GEXMNFE)+(W1MNFE)-(W2MNFE)))
#define GR1S3         0.25*(( (GEXMGMN)-(W1MGMN)+(W2MGMN)) \
                                                        -((GEXMGCO)+(W1MGCO)-(W2MGCO)) \
                                                        +((GEXMNCO)+(W1MNCO)-(W2MNCO)))
#define GR1S4         0.25*(( (GEXMGMN)-(W1MGMN)+(W2MGMN)) \
                                                        -((GEXMGNI)+(W1MGNI)-(W2MGNI)) \
                                                        +((GEXMNNI)+(W1MNNI)-(W2MNNI)))
#define GR2R2         -0.25*(( (GXMGFE)+(W1MGFE)+(W2MGFE)))
#define GR2R3          0.25*(( (GXFECO)+(W1FECO)+(W2FECO)) \
                             -((GXMGFE)+(W1MGFE)+(W2MGFE)) \
                             -((GXMGCO)+(W1MGCO)+(W2MGCO)))
#define GR2R4          0.25*(( (GXFENI)+(W1FENI)+(W2FENI)) \
                             -((GXMGFE)+(W1MGFE)+(W2MGFE)) \
                             -((GXMGNI)+(W1MGNI)+(W2MGNI)))
#define GR2S1         0.25*(( (GEXMGMN)+(W1MGMN)-(W2MGMN)) \
                                                        -((GEXMGFE)-(W1MGFE)+(W2MGFE)) \
                                                        +((GEXMNFE)-(W1MNFE)+(W2MNFE)))
#define GR2S2                       0.5*((W1MGFE)-(W2MGFE))
#define GR2S3         0.25*(( (GEXMGFE)-(W1MGFE)+(W2MGFE)) \
                                                        -((GEXMGCO)+(W1MGCO)-(W2MGCO)) \
                                                        +((GEXFECO)+(W1FECO)-(W2FECO)))
#define GR2S4         0.25*(( (GEXMGFE)-(W1MGFE)+(W2MGFE)) \
                                                        -((GEXMGNI)+(W1MGNI)-(W2MGNI)) \
                                                        +((GEXFENI)+(W1FENI)-(W2FENI)))
#define GR3R3          -0.25*(  (GXMGCO)+(W1MGCO)+(W2MGCO))
#define GR3R4          0.25*(( (GXCONI)+(W1CONI)+(W2CONI)) \
                             -((GXMGCO)+(W1MGCO)+(W2MGCO)) \
                             -((GXMGNI)+(W1MGNI)+(W2MGNI)))
#define GR3S1         0.25*(( (GEXMGMN)+(W1MGMN)-(W2MGMN)) \
                                                        -((GEXMGCO)-(W1MGCO)+(W2MGCO)) \
                                                        +((GEXMNCO)-(W1MNCO)+(W2MNCO)))
#define GR3S2         0.25*(( (GEXMGFE)+(W1MGFE)-(W2MGFE)) \
                                                        -((GEXMGCO)-(W1MGCO)+(W2MGCO)) \
                                                        +((GEXFECO)-(W1FECO)+(W2FECO)))
#define GR3S3                      0.5*(-(W1MGCO)+(W2MGCO))
#define GR3S4          0.25*(( (GEXMGCO)+(W1MGCO)-(W2MGCO)) \
                             -((GEXMGNI)-(W1MGNI)+(W2MGNI)) \
                             +((GEXCONI)-(W1CONI)+(W2CONI)))
#define GR4R4           -0.25*(  (GXMGNI)+(W1MGNI)+(W2MGNI))
#define GR4R5              -0.25*((F_NI)+(GEXMGNI)+(GXMGNI) \
                                        +2.0*(W2CAMG)-2.0*(W2CANI)+2.0*(W2MGNI))
#define GR4S1          0.25*(( (GEXMGMN)+(W1MGMN)-(W2MGMN)) \
                             -((GEXMGNI)-(W1MGNI)+(W2MGNI)) \
                             +((GEXMNNI)-(W1MNNI)+(W2MNNI)))
#define GR4S2          0.25*(( (GEXMGFE)+(W1MGFE)-(W2MGFE)) \
                             -((GEXMGNI)-(W1MGNI)+(W2MGNI)) \
                             +((GEXFENI)-(W1FENI)+(W2FENI)))
#define GR4S3          0.25*((-(GEXMGCO)+(W1MGCO)-(W2MGCO)) \
                             +((GEXMGNI)-(W1MGNI)+(W2MGNI)) \
                             -((GEXCONI)-(W1CONI)+(W2CONI)))
#define GR4S4                       0.5*(-(W1MGNI)+(W2MGNI))
#define GR5R5        -1.0*(W2CAMG)
#define GR5S1                0.25*((F_MN)+(GEXMGMN)+(GXMGMN) \
                     -2.0*(W2CAMG)+2.0*(W2CAMN)-2.0*(W2MGMN))
#define GR5S2                0.25*((F_FE)+(GEXMGFE)+(GXMGFE) \
                     -2.0*(W2CAMG)+2.0*(W2CAFE)-2.0*(W2MGFE))
#define GR5S3               -0.25*((F_CO)+(GEXMGCO)+(GXMGCO) \
                     -2.0*(W2CAMG)+2.0*(W2CACO)-2.0*(W2MGCO))
#define GR5S4               -0.25*((F_NI)+(GEXMGNI)+(GXMGNI) \
                     -2.0*(W2CAMG)+2.0*(W2CANI)-2.0*(W2MGNI))
#define GS1S1               0.25*((GXMGMN)-(W1MGMN)-(W2MGMN))
#define GS1S2            0.25*(( (GXMGMN)-(W1MGMN)-(W2MGMN)) \
                               +((GXMGFE)-(W1MGFE)-(W2MGFE)) \
                               -((GXMNFE)-(W1MNFE)-(W2MNFE)))
#define GS1S3            0.25*(-((GXMGMN)-(W1MGMN)-(W2MGMN)) \
                               -((GXMGCO)-(W1MGCO)-(W2MGCO)) \
                               +((GXMNCO)-(W1MNCO)-(W2MNCO)))
#define GS1S4            0.25*(-((GXMGMN)-(W1MGMN)-(W2MGMN)) \
                               -((GXMGNI)-(W1MGNI)-(W2MGNI)) \
                               +((GXMNNI)-(W1MNNI)-(W2MNNI)))
#define GS2S2             0.25*( (GXMGFE)-(W1MGFE)-(W2MGFE))
#define GS2S3            0.25*(-((GXMGFE)-(W1MGFE)-(W2MGFE)) \
                               -((GXMGCO)-(W1MGCO)-(W2MGCO)) \
                               +((GXFECO)-(W1FECO)-(W2FECO)))
#define GS2S4            0.25*(-((GXMGFE)-(W1MGFE)-(W2MGFE)) \
                               -((GXMGNI)-(W1MGNI)-(W2MGNI)) \
                               +((GXFENI)-(W1FENI)-(W2FENI)))
#define GS3S3             0.25*( (GXMGCO)-(W1MGCO)-(W2MGCO))
#define GS3S4            0.25*( ((GXMGCO)-(W1MGCO)-(W2MGCO)) \
                               +((GXMGNI)-(W1MGNI)-(W2MGNI)) \
                               -((GXCONI)-(W1CONI)-(W2CONI)))
#define GS4S4             0.25*( (GXMGNI)-(W1MGNI)-(W2MGNI))

static double H_of(double r0,double r1,double r2,double r3,double r4,
                    double s0,double s1,double s2,double s3, double p) {
    return (G0) +
        (GR1)*r0 + (GR2)*r1 + (GR3)*r2 + (GR4)*r3 + (GR5)*r4 +
        (GS1)*s0 + (GS2)*s1 + (GS3)*s2 + (GS4)*s3 +
        (GR1R1)*r0*r0 + (GR1R2)*r0*r1 + (GR1R3)*r0*r2 +
        (GR1R4)*r0*r3 + (GR1R5)*r0*r4 + (GR1S1)*r0*s0 +
        (GR1S2)*r0*s1 + (GR1S3)*r0*s2 + (GR1S4)*r0*s3 +
        (GR2R2)*r1*r1 + (GR2R3)*r1*r2 + (GR2R4)*r1*r3 +
        (GR2R5)*r1*r4 + (GR2S1)*r1*s0 + (GR2S2)*r1*s1 +
        (GR2S3)*r1*s2 + (GR2S4)*r1*s3 + (GR3R3)*r2*r2 +
        (GR3R4)*r2*r3 + (GR3R5)*r2*r4 + (GR3S1)*r2*s0 +
        (GR3S2)*r2*s1 + (GR3S3)*r2*s2 + (GR3S4)*r2*s3 +
        (GR4R4)*r3*r3 + (GR4R5)*r3*r4 + (GR4S1)*r3*s0 +
        (GR4S2)*r3*s1 + (GR4S3)*r3*s2 + (GR4S4)*r3*s3 +
        (GR5R5)*r4*r4 + (GR5S1)*r4*s0 + (GR5S2)*r4*s1 +
        (GR5S3)*r4*s2 + (GR5S4)*r4*s3 + (GS1S1)*s0*s0 +
        (GS1S2)*s0*s1 + (GS1S3)*s0*s2 + (GS1S4)*s0*s3 +
        (GS2S2)*s1*s1 + (GS2S3)*s1*s2 + (GS2S4)*s1*s3 +
        (GS3S3)*s2*s2 + (GS3S4)*s2*s3 + (GS4S4)*s3*s3;
}

static void site_fracs(double r0,double r1,double r2,double r3,double r4,
                        double s0,double s1,double s2,double s3,
                        double *xm1mn,double *xm2mn,double *xm1fe,double *xm2fe,
                        double *xm1co,double *xm2co,double *xm1ni,double *xm2ni,
                        double *xm2ca,double *xm1mg,double *xm2mg) {
    *xm1mn = (r0-s0+1.0)/2.0;
    *xm2mn = (r0+s0+1.0)/2.0;
    *xm1fe = (r1-s1+1.0)/2.0;
    *xm2fe = (r1+s1+1.0)/2.0;
    *xm1co = (r2+s2+1.0)/2.0;
    *xm2co = (r2-s2+1.0)/2.0;
    *xm1ni = (r3+s3+1.0)/2.0;
    *xm2ni = (r3-s3+1.0)/2.0;
    *xm2ca = r4;
    *xm1mg = 1.0 - *xm1mn - *xm1fe - *xm1co - *xm1ni;
    *xm2mg = 1.0 - *xm2mn - *xm2fe - *xm2co - *xm2ni - *xm2ca;
}

static double S_of(double xm1mn,double xm2mn,double xm1fe,double xm2fe,
                    double xm1co,double xm2co,double xm1ni,double xm2ni,
                    double xm2ca,double xm1mg,double xm2mg) {
    return -R*( xm1mn*log(xm1mn) + xm2mn*log(xm2mn) + xm1fe*log(xm1fe)+
                xm2fe*log(xm2fe) + xm1co*log(xm1co) + xm2co*log(xm2co)+
                xm1ni*log(xm1ni) + xm2ni*log(xm2ni) + xm2ca*log(xm2ca)+
                xm1mg*log(xm1mg) + xm2mg*log(xm2mg));
}

#define DGDR0M(r0,r1,r2,r3,r4,s0,s1,s2,s3,t,xm1mn,xm2mn,xm1mg,xm2mg) \
    ((GR1) + 2.0*(GR1R1)*r0 + \
     (GR1R2)*r1 + (GR1R3)*r2 + (GR1R4)*r3 + (GR1R5)*r4 + \
     (GR1S1)*s0 + (GR1S2)*s1 + (GR1S3)*s2 + (GR1S4)*s3 + \
     0.5*R*t*(log(xm1mn*xm2mn/xm1mg/xm2mg)))

#define DGDS0M(r0,r1,r2,r3,r4,s0,s1,s2,s3,t,xm2mn,xm1mn,xm1mg,xm2mg) \
    ((GS1) + 2.0*(GS1S1)*s0 + \
     (GR1S1)*r0 + (GR2S1)*r1 + (GR3S1)*r2 + (GR4S1)*r3 + \
     (GR5S1)*r4 + (GS1S2)*s1 + (GS1S3)*s2 + (GS1S4)*s3 + \
     0.5*R*t*(log(xm2mn*xm1mg/xm1mn/xm2mg)))

int main(void) {
    printf("=== FELDSPAR ===\n");
    printf("xab\txan\tT\tP\tG\tH\tS\tV\n");
    {
        double comps[][2] = {{0.6,0.3},{0.9,0.05},{0.2,0.7},{0.33,0.33}};
        double Ts[] = {900.0, 1200.0};
        double Ps[] = {1.0, 5000.0};
        for (int i=0;i<4;i++) {
            for (int b=0;b<2;b++) {
                double G,H,S,V;
                feldspar_gmix(comps[i][0], comps[i][1], Ts[b], Ps[b], &G,&H,&S,&V);
                printf("%.6g\t%.6g\t%.1f\t%.1f\t%.10g\t%.10g\t%.10g\t%.10g\n",
                       comps[i][0], comps[i][1], Ts[b], Ps[b], G,H,S,V);
            }
        }
    }

    printf("=== OLIVINE_COEFFS ===\n");
    printf("P\tG0\tGR1\tGR2\tGR3\tGR4\tGR5\tGS1\tGS2\tGS3\tGS4\tGR1R5\tGR2R5\tGR3R5\n");
    {
        double Ps[] = {1.0, 10000.0, 30000.0};
        for (int i=0;i<3;i++) {
            double p = Ps[i];
            printf("%.1f\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\n",
                   p, (double)(G0), (double)(GR1), (double)(GR2), (double)(GR3), (double)(GR4), (double)(GR5),
                   (double)(GS1), (double)(GS2), (double)(GS3), (double)(GS4),
                   (double)(GR1R5), (double)(GR2R5), (double)(GR3R5));
        }
    }

    printf("=== OLIVINE_H_S_DGDR_DGDS ===\n");
    printf("r0\tr1\tr2\tr3\tr4\ts0\ts1\ts2\ts3\tT\tP\tH\tS\tG\tDGDR0\tDGDS0\n");
    {
        /* Mixed composition matching the Python smoke test, plus the
         * ordering parameters returned by the Python Newton solve there
         * (pasted in from that run) so this is an apples-to-apples
         * check of the H/S/DGDR0/DGDS0 *formulas* themselves, not of
         * whether the Newton solves agree (a separate, already-strong
         * check via the pure-forsterite limit in olivine.py's own
         * module tests). */
        double tests[][11] = {
            /* r0,r1,r2,r3,r4, s0,s1,s2,s3, T,P */
            {-0.998, -0.80, -0.998, -0.998, 0.01,  0.00115527206,-0.0000603898399,0.00104057230,0.00136465727, 1400.0, 10000.0},
            {-0.998, -0.40, -0.998, -0.998, 0.01,  0.000996295403,0.0000181380011,0.000745383563,0.00121539113, 1600.0, 30000.0},
        };
        for (int i=0;i<2;i++) {
            double r0=tests[i][0], r1=tests[i][1], r2=tests[i][2], r3=tests[i][3], r4=tests[i][4];
            double s0=tests[i][5], s1=tests[i][6], s2=tests[i][7], s3=tests[i][8];
            double t=tests[i][9], p=tests[i][10];
            double xm1mn,xm2mn,xm1fe,xm2fe,xm1co,xm2co,xm1ni,xm2ni,xm2ca,xm1mg,xm2mg;
            site_fracs(r0,r1,r2,r3,r4,s0,s1,s2,s3,
                       &xm1mn,&xm2mn,&xm1fe,&xm2fe,&xm1co,&xm2co,&xm1ni,&xm2ni,&xm2ca,&xm1mg,&xm2mg);
            double H = H_of(r0,r1,r2,r3,r4,s0,s1,s2,s3,p);
            double S = S_of(xm1mn,xm2mn,xm1fe,xm2fe,xm1co,xm2co,xm1ni,xm2ni,xm2ca,xm1mg,xm2mg);
            double G = H - t*S;
            double dgdr0 = DGDR0M(r0,r1,r2,r3,r4,s0,s1,s2,s3,t,xm1mn,xm2mn,xm1mg,xm2mg);
            double dgds0 = DGDS0M(r0,r1,r2,r3,r4,s0,s1,s2,s3,t,xm2mn,xm1mn,xm1mg,xm2mg);
            printf("%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.1f\t%.1f\t%.10g\t%.10g\t%.10g\t%.10g\t%.10g\n",
                   r0,r1,r2,r3,r4,s0,s1,s2,s3,t,p,H,S,G,dgdr0,dgds0);
        }
    }
    return 0;
}
