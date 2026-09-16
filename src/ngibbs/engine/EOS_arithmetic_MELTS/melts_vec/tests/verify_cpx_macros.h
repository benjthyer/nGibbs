#define  R       8.3143
#define  NR      6       /* Six independent composition variables */
#define  NS      2       /* Two ordering parameters               */
#define  NA      7       /* Seven endmember compositions          */

#define  DH1      0.5573     * 1000.0 * 4.184 /* joules     */
#define  DS1      0.00033    * 1000.0 * 4.184 /* joules/K   */
#define  DV1      0.018               * 4.184 /* joules/bar */
#define  DH2      1.200      * 1000.0 * 4.184 /* joules     */
#define  DS2      0.000675   * 1000.0 * 4.184 /* joules/K   */
#define  DV2      0.0148              * 4.184 /* joules/bar */
#define  DH3      0.6991252  * 1000.0 * 4.184 /* joules     */
#define  DS3      0.0005926  * 1000.0 * 4.184 /* joules/K   */
#define  DV3      0.00865             * 4.184 /* joules/bar */
#define  DH4     -0.49       * 1000.0 * 4.184 /* joules     */
#define  DS4     -0.000132   * 1000.0 * 4.184 /* joules/K   */

#define cV0DI     6.620                       /* joules/bar */  /* reference */
#define cV0HD     6.7894                      /* joules/bar */  /* reference */
#define oV0HD     1.64620             * 4.184 /* joules/bar */
#define oV0EN     3.133 * 2.0                 /* joules/bar */  /* reference */
#define oV0FS     3.296 * 2.0                 /* joules/bar */  /* reference */

#define oH027    -3.30       * 1000.0 * 4.184 /* joules     */
#define oS027    -0.00055428 * 1000.0 * 4.184 /* joules/K   */
#define pH027    -1.97547480 * 1000.0 * 4.184 /* joules     */
#define pS027     0.00071134 * 1000.0 * 4.184 /* joules/K   */
#define pV027    -0.00920929          * 4.184 /* joules/bar */

#define oHEX     -1.87       * 1000.0 * 4.184 /* joules     */
#define oVEX     -0.029               * 4.184 /* joules/bar */
#define cHEX     -2.2        * 1000.0 * 4.184 /* joules     */
#define cVEX      0.01                * 4.184 /* joules/bar */
#define pHEX     -0.65       * 1000.0 * 4.184 /* joules     */
#define pVEX      0.0                 * 4.184 /* joules/bar */

#define oHX      -0.45       * 1000.0 * 4.184 /* joules     */
#define oVX       0.00675             * 4.184 /* joules/bar */
#define cHX      -0.10       * 1000.0 * 4.184 /* joules     */
#define cVX       0.0                 * 4.184 /* joules/bar */
#define pHX      -0.10       * 1000.0 * 4.184 /* joules     */
#define pVX       0.0                 * 4.184 /* joules/bar */

#define oWHFEMG   2.0        * 1000.0 * 4.184 /* joules     */  /* W Fe-Mg M2 */
#define oWVFEMG   0.003375            * 4.184 /* joules/bar */
#define cWHFEMG   1.15       * 1000.0 * 4.184 /* joules     */
#define cWVFEMG  -0.0035              * 4.184 /* joules/bar */

#define oWH12     2.0        * 1000.0 * 4.184 /* joules     */  /* W12 */
#define oWV12     0.003375            * 4.184 /* joules/bar */
#define cWH12     1.68       * 1000.0 * 4.184 /* joules     */
#define cWV12     0.0                 * 4.184 /* joules/bar */

#define oWHCAMG   7.56       * 1000.0 * 4.184 /* joules     */  /* W17 */
#define oWVCAMG   0.008               * 4.184 /* joules/bar */
#define cWHCAMG   6.72       * 1000.0 * 4.184 /* joules     */
#define cWVCAMG  -0.009               * 4.184 /* joules/bar */

#define oWHCAFE   4.12       * 1000.0 * 4.184 /* joules     */  /* W17U */
#define oWVCAFE   0.011               * 4.184 /* joules/bar */
#define cWHCAFE   4.515      * 1000.0 * 4.184 /* joules     */
#define cWVCAFE   0.003               * 4.184 /* joules/bar */

#define oDWHCAMG  -1.3       * 1000.0 * 4.184 /* joules     */  /* delta W17 */
#define oDWVCAMG   0.012              * 4.184 /* joules/bar */
#define cDWHCAMG  -0.7       * 1000.0 * 4.184 /* joules     */
#define cDWVCAMG   0.008              * 4.184 /* joules/bar */

#define oDWHCAFE  -1.1       * 1000.0 * 4.184 /* joules     */  /* delta W17U */
#define oDWVCAFE   0.005              * 4.184 /* joules/bar */
#define cDWHCAFE  -0.5375    * 1000.0 * 4.184 /* joules     */
#define cDWVCAFE   0.0025             * 4.184 /* joules/bar */

#define  W13     16.318     * 1000.0 /* joules     */
#define  W14     18.82800   * 1000.0 /* joules     */
#define  W15     20.920     * 1000.0 /* joules     */
#define  W15P    16.78900   * 1000.0 /* joules     */
#define  W16      0.00000   * 1000.0 /* joules     */
#define  W25      3.98176   * 1000.0 /* joules     */
#define  W25P     7.49005   * 1000.0 /* joules     */
#define  W26      0.00000   * 1000.0 /* joules     */
#define  W34     16.78900   * 1000.0 /* joules     */
#define  W35     12.02812   * 1000.0 /* joules     */
#define  W35P    47.38600   * 1000.0 /* joules     */
#define  W36     34.35062   * 1000.0 /* joules     */
#define  cW37    28.65091   * 1000.0 /* joules     */
#define  cW3U7U  34.90070   * 1000.0 /* joules     */
#define  W45     27.14312   * 1000.0 /* joules     */
#define  W45P    16.318     * 1000.0 /* joules     */
#define  W46     40.40662   * 1000.0 /* joules     */
#define  cW47    35.38104   * 1000.0 /* joules     */
#define  cW4U7U  28.54527   * 1000.0 /* joules     */
#define  cW55    35.670     * 1000.0 /* joules     */
#define  W56      0.00000   * 1000.0 /* joules     */
#define  cW57    33.25291   * 1000.0 /* joules     */
#define  cW57U   15.29737   * 1000.0 /* joules     */
#define  W5P6     0.00000   * 1000.0 /* joules     */
#define  cW5P7   36.24989   * 1000.0 /* joules     */
#define  cW5P7U   2.35321   * 1000.0 /* joules     */
#define  cW67     0.00000   * 1000.0 /* joules     */
#define  cW67U    0.00000   * 1000.0 /* joules     */
#define  oW37    28.65091   * 1000.0 /* joules     */
#define  oW3U7U  34.90070   * 1000.0 /* joules     */
#define  oW47    35.38104   * 1000.0 /* joules     */
#define  oW4U7U  28.54527   * 1000.0 /* joules     */
#define  oW55    35.670     * 1000.0 /* joules     */
#define  oW57    33.25291   * 1000.0 /* joules     */
#define  oW57U   15.29737   * 1000.0 /* joules     */
#define  oW5P7   36.24989   * 1000.0 /* joules     */
#define  oW5P7U   2.35321   * 1000.0 /* joules     */
#define  oW67     0.00000   * 1000.0 /* joules     */
#define  oW67U    0.00000   * 1000.0 /* joules     */

#define  H23     -2.71700   * 1000.0 /* joules     */
#define  S23      0.0                /* joules/K   */
#define  H24     -7.36648   * 1000.0 /* joules     */
#define  S24      0.0                /* joules/K   */
#define  cH55     9.48524   * 1000.0 /* joules     */
#define  S55      0.0                /* joules/K   */
#define  oH55     9.48524   * 1000.0 /* joules     */

#define  DcTOoH3    25.11924    * 1000.0 /* joules     */
#define  DcTOoS3    -2.00000             /* joules/K   */
#define  DcTOoV3    -0.05129             /* joules/bar */
#define  DcTOoH4    25.11924    * 1000.0 /* joules     */
#define  DcTOoS4    -2.00000             /* joules/K   */
#define  DcTOoV4    -0.05129             /* joules/bar */
#define  DcTOoH5    25.11924    * 1000.0 /* joules     */
#define  DcTOoS5    -2.00000             /* joules/K   */
#define  DcTOoV5    -0.05129             /* joules/bar */
#define  DcTOoH6    25.11924    * 1000.0 /* joules     */
#define  DcTOoS6    -2.00000             /* joules/K   */
#define  DcTOoV6    -0.05129             /* joules/bar */

#define  DcTOpH1    -0.9        * 1000.0 * 4.184 /* joules     */
#define  DcTOpS1    -0.0006920  * 1000.0 * 4.184 /* joules/K   */
#define  DcTOpV1     0.0                 * 4.184 /* joules/bar */
#define  DcTOpH2    -1.88       * 1000.0 * 4.184 /* joules     */
#define  DcTOpS2    -0.00156401 * 1000.0 * 4.184 /* joules/K   */
#define  DcTOpV2     0.0                 * 4.184 /* joules/bar */
#define  DcTOpH3    21.11924    * 1000.0 /* joules     */
#define  DcTOpS3    -2.00000             /* joules/K   */
#define  DcTOpV3    -0.05129             /* joules/bar */
#define  DcTOpH4    21.11924    * 1000.0 /* joules     */
#define  DcTOpS4    -2.00000             /* joules/K   */
#define  DcTOpV4    -0.05129             /* joules/bar */
#define  DcTOpH5    21.11924    * 1000.0 /* joules     */
#define  DcTOpS5    -2.00000             /* joules/K   */
#define  DcTOpV5    -0.05129             /* joules/bar */
#define  DcTOpHj    21.11924    * 1000.0 /* joules     */ /* Jadeite */
#define  DcTOpSj    -2.00000             /* joules/K   */ /* Jadeite */
#define  DcTOpVj    -0.05129             /* joules/bar */ /* Jadeite */
#define  DcTOpH7     0.0        * 1000.0 * 4.184 /* joules     */
#define  DcTOpS7     0.0        * 1000.0 * 4.184 /* joules/K   */
#define  DcTOpV7     0.0                 * 4.184 /* joules/bar */

/*
 *=============================================================================
 * Dependent parameters (DO NOT change definitions below this line:
 */

#define oV0DI    (cV0DI)+(DV1)
#define cV0EN    (oV0EN)+(DV2)
#define cV0FS    (oV0FS)+(DV3)

#define DV4      (cV0HD)-(oV0HD)

#define oV027    (oV0FS)-(oV0EN)+2.0*(oV0DI)-2.0*(oV0HD)
#define cV027    (cV0FS)-(cV0EN)+2.0*(cV0DI)-2.0*(cV0HD)
#define cH027    (oH027)+(DH3)-(DH2)-2.0*(DH1)-2.0*(DH4)
#define cS027    (oS027)+(DS3)-(DS2)-2.0*(DS1)-2.0*(DS4) 

#define  DG1     (DH1)-t*(DS1)+(p-1.0)*DV1
#define  DG2     (DH2)-t*(DS2)+(p-1.0)*DV2
#define  DG3     (DH3)-t*(DS3)+(p-1.0)*DV3
#define  DG4     (DH4)-t*(DS4)+(p-1.0)*DV4

#define cG027    (cH027)-t*(cS027)+(p-1.0)*(cV027)
#define pG027    (pH027)-t*(pS027)+(p-1.0)*(pV027)
#define oG027    (oH027)-t*(oS027)+(p-1.0)*(oV027)
#define  H027    (clino) ? (cH027) : (oH027)
#define  S027    (clino) ? (cS027) : (oS027)
#define  V027    (clino) ? (cV027) : (oV027)
#define  G027    (clino) ? (cG027) : (oG027)
#define dH027    (clino) ? (pH027)-(cH027) : 0.0
#define dS027    (clino) ? (pS027)-(cS027) : 0.0
#define dV027    (clino) ? (pV027)-(cV027) : 0.0
#define dG027    (clino) ? (pG027)-(cG027) : 0.0

#define cGEX     (cHEX)+(p-1.0)*(cVEX)
#define pGEX     (pHEX)+(p-1.0)*(pVEX)
#define oGEX     (oHEX)+(p-1.0)*(oVEX)
#define  HEX     (clino) ? (cHEX) : (oHEX)
#define  VEX     (clino) ? (cVEX) : (oVEX)
#define  GEX     (clino) ? (cGEX) : (oGEX)
#define dHEX     (clino) ? (pHEX)-(cHEX) : 0.0
#define dVEX     (clino) ? (pVEX)-(cVEX) : 0.0
#define dGEX     (clino) ? (pGEX)-(cGEX) : 0.0

#define cGX      (cHX)+(p-1.0)*(cVX)
#define pGX      (pHX)+(p-1.0)*(pVX)
#define oGX      (oHX)+(p-1.0)*(oVX)
#define  HX      (clino) ? (cHX) : (oHX)
#define  VX      (clino) ? (cVX) : (oVX)
#define  GX      (clino) ? (cGX) : (oGX)
#define dHX      (clino) ? (pHX)-(cHX) : 0.0
#define dVX      (clino) ? (pVX)-(cVX) : 0.0
#define dGX      (clino) ? (pGX)-(cGX) : 0.0

#define oWFEMG   (oWHFEMG)+(p-1.0)*(oWVFEMG)
#define cWFEMG   (cWHFEMG)+(p-1.0)*(cWVFEMG)
#define  WHFEMG  (clino) ? (cWHFEMG) : (oWHFEMG)
#define  WVFEMG  (clino) ? (cWVFEMG) : (oWVFEMG)
#define  WFEMG   (clino) ? (cWFEMG) : (oWFEMG)

#define oW12     (oWH12)+(p-1.0)*(oWV12)
#define cW12     (cWH12)+(p-1.0)*(cWV12)
#define  WH12    (clino) ? (cWH12) : (oWH12)
#define  WV12    (clino) ? (cWV12) : (oWV12)
#define  W12     (clino) ? (cW12) : (oW12)

#define oWCAMG   (oWHCAMG)+(p-1.0)*(oWVCAMG)
#define cWCAMG   (cWHCAMG)+(p-1.0)*(cWVCAMG)
#define  WHCAMG  (clino) ? (cWHCAMG) : (oWHCAMG)
#define  WVCAMG  (clino) ? (cWVCAMG) : (oWVCAMG)
#define  WCAMG   (clino) ? (cWCAMG) : (oWCAMG)

#define oDWCAMG   (oDWHCAMG)+(p-1.0)*(oDWVCAMG)
#define cDWCAMG   (cDWHCAMG)+(p-1.0)*(cDWVCAMG)
#define  DWHCAMG  (clino) ? (cDWHCAMG) : (oDWHCAMG)
#define  DWVCAMG  (clino) ? (cDWVCAMG) : (oDWVCAMG)
#define  DWCAMG   (clino) ? (cDWCAMG) : (oDWCAMG)

#define oWCAFE   (oWHCAFE)+(p-1.0)*(oWVCAFE)
#define cWCAFE   (cWHCAFE)+(p-1.0)*(cWVCAFE)
#define  WHCAFE  (clino) ? (cWHCAFE) : (oWHCAFE)
#define  WVCAFE  (clino) ? (cWVCAFE) : (oWVCAFE)
#define  WCAFE   (clino) ? (cWCAFE) : (oWCAFE)

#define oDWCAFE   (oDWHCAFE)+(p-1.0)*(oDWVCAFE)
#define cDWCAFE   (cDWHCAFE)+(p-1.0)*(cDWVCAFE)
#define  DWHCAFE  (clino) ? (cDWHCAFE) : (oDWHCAFE)
#define  DWVCAFE  (clino) ? (cDWVCAFE) : (oDWVCAFE)
#define  DWCAFE   (clino) ? (cDWCAFE) : (oDWCAFE)

#define  W37      (clino) ? (cW37)   : (oW37)
#define  W3U7U    (clino) ? (cW3U7U) : (oW3U7U)
#define  W47      (clino) ? (cW47)   : (oW47)
#define  W4U7U    (clino) ? (cW4U7U) : (oW4U7U)
#define  W55      (clino) ? (cW55)   : (oW55)
#define  W57      (clino) ? (cW57)   : (oW57)
#define  W57U     (clino) ? (cW57U)  : (oW57U)
#define  W5P7     (clino) ? (cW5P7)  : (oW5P7)
#define  W5P7U    (clino) ? (cW5P7U) : (oW5P7U)
#define  W67      (clino) ? (cW67)   : (oW67)
#define  W67U     (clino) ? (cW67U)  : (oW67U)
#define  H55      (clino) ? (cH55)   : (oH55)

#define  G23     (H23)-t*(S23)
#define  G24     (H24)-t*(S24)
#define  G55     (H55)-t*(S55)

#define  DcTOoG3 (DcTOoH3)-t*(DcTOoS3)+(p-1.0)*(DcTOoV3)
#define  DcTOoG4 (DcTOoH4)-t*(DcTOoS4)+(p-1.0)*(DcTOoV4)
#define  DcTOoG5 (DcTOoH5)-t*(DcTOoS5)+(p-1.0)*(DcTOoV5)
#define  DcTOoG6 (DcTOoH6)-t*(DcTOoS6)+(p-1.0)*(DcTOoV6)

#define  DcTOpG1 (DcTOpH1)-t*(DcTOpS1)+(p-1.0)*(DcTOpV1)
#define  DcTOpG2 (DcTOpH2)-t*(DcTOpS2)+(p-1.0)*(DcTOpV2)
#define  DcTOpG3 (DcTOpH3)-t*(DcTOpS3)+(p-1.0)*(DcTOpV3)
#define  DcTOpG4 (DcTOpH4)-t*(DcTOpS4)+(p-1.0)*(DcTOpV4)
#define  DcTOpG5 (DcTOpH5)-t*(DcTOpS5)+(p-1.0)*(DcTOpV5)
#define  DcTOpGj (DcTOpHj)-t*(DcTOpSj)+(p-1.0)*(DcTOpVj)
#define  DcTOpG7 (DcTOpH7)-t*(DcTOpS7)+(p-1.0)*(DcTOpV7)

/*
 * Vertices of composition space 
 */

         /* (Ca)(Mg)(Si)2 O6 */
#define  H1      (clino) ? 0.0 :  (DH1)
#define  S1      (clino) ? 0.0 :  (DS1)
#define  V1      (clino) ? 0.0 :  (DV1)
#define  G1      (clino) ? 0.0 :  (DH1)-t*(DS1)+(p-1.0)*(DV1)
         /* (Ca) (Fe2+) (Si)2 O6 */
#define  H2      (clino) ? 0.0 : -(DH4)
#define  S2      (clino) ? 0.0 : -(DS4)
#define  V2      (clino) ? 0.0 : -(DV4)
#define  G2      (clino) ? 0.0 : -(DH4)+t*(DS4)-(p-1.0)*(DV4)
         /* (Ca) (Ti,Mg) (Al,Si)2 O6 */
#define  H3      (clino) ? 0.0 : (DcTOoH3)
#define  S3      (clino) ? 0.0 : (DcTOoS3)
#define  V3      (clino) ? 0.0 : (DcTOoV3)
#define  G3      (clino) ? 0.0 : (DcTOoG3)
         /* (Ca) (Ti,Mg) (Fe3+,Si)2 O6 */
#define  H4      (clino) ? 0.0 : (DcTOoH4)
#define  S4      (clino) ? 0.0 : (DcTOoS4)
#define  V4      (clino) ? 0.0 : (DcTOoV4)
#define  G4      (clino) ? 0.0 : (DcTOoG4)
         /* (Ca) (Fe3+,Al) [(Fe3+,Al),Si]2 O6 */
#define  H5      (clino) ? 0.0 : (DcTOoH5)
#define  S5      (clino) ? 0.0 : (DcTOoS5)
#define  V5      (clino) ? 0.0 : (DcTOoV5)
#define  G5      (clino) ? 0.0 : (DcTOoG5)
         /* (Na) (Al) (Si)2 O6 */ 
#define  H6      (clino) ? 0.0 : (DcTOoH6)
#define  S6      (clino) ? 0.0 : (DcTOoS6)
#define  V6      (clino) ? 0.0 : (DcTOoV6)
#define  G6      (clino) ? 0.0 : (DcTOoG6)
         /* (Mg) (Mg) (Si)2 O6 */
#define  H7      (clino) ? 0.0 : -(DH2)
#define  S7      (clino) ? 0.0 : -(DS2)
#define  V7      (clino) ? 0.0 : -(DV2)
#define  G7      (clino) ? 0.0 : -(DH2)+t*(DS2)-(p-1.0)*(DV2)
         /* Note that ferrosilite (and DG3 terms) are dependent */

         /* (Ca)(Mg)(Si)2 O6 */
#define  pH1     (clino) ? (DcTOpH1) : 0.0
#define  pS1     (clino) ? (DcTOpS1) : 0.0
#define  pV1     (clino) ? (DcTOpV1) : 0.0
#define  pG1     (clino) ? (DcTOpG1) : 0.0
         /* (Ca) (Fe2+) (Si)2 O6 */
#define  pH2     (clino) ? (DcTOpH2) : 0.0
#define  pS2     (clino) ? (DcTOpS2) : 0.0
#define  pV2     (clino) ? (DcTOpV2) : 0.0
#define  pG2     (clino) ? (DcTOpG2) : 0.0
         /* (Ca) (Ti,Mg) (Al,Si)2 O6 */
#define  pH3     (clino) ? (DcTOpH3) : 0.0
#define  pS3     (clino) ? (DcTOpS3) : 0.0
#define  pV3     (clino) ? (DcTOpV3) : 0.0
#define  pG3     (clino) ? (DcTOpG3) : 0.0
         /* (Ca) (Ti,Mg) (Fe3+,Si)2 O6 */
#define  pH4     (clino) ? (DcTOpH4) : 0.0
#define  pS4     (clino) ? (DcTOpS4) : 0.0
#define  pV4     (clino) ? (DcTOpV4) : 0.0
#define  pG4     (clino) ? (DcTOpG4) : 0.0
         /* (Ca) (Fe3+,Al) [(Fe3+,Al),Si]2 O6 */
#define  pH5     (clino) ? (DcTOpH5) : 0.0
#define  pS5     (clino) ? (DcTOpS5) : 0.0
#define  pV5     (clino) ? (DcTOpV5) : 0.0
#define  pG5     (clino) ? (DcTOpG5) : 0.0
         /* (Na) (Al) (Si)2 O6 */ 
#define  pHj     (clino) ? (DcTOpHj) : 0.0
#define  pSj     (clino) ? (DcTOpSj) : 0.0
#define  pVj     (clino) ? (DcTOpVj) : 0.0
#define  pGj     (clino) ? (DcTOpGj) : 0.0
         /* (Mg) (Mg) (Si)2 O6 */
#define  pH7     (clino) ? (DcTOpH7) : 0.0
#define  pS7     (clino) ? (DcTOpS7) : 0.0
#define  pV7     (clino) ? (DcTOpV7) : 0.0
#define  pG7     (clino) ? (DcTOpG7) : 0.0

/*
 * Definitions of Taylor expansion coefficients in terms of solution
 * parameters. Independent variables are x2, x3, x4, x5, x6, x7, s1, s2
 */

#define H0    (H1)
#define HX2   (H2)-(H1) + (WH12)
#define HX3   (H3)-(H1) + (W13)
#define HX4   (H4)-(H1) + (W14)
#define HX5   (H5)-(H1) + 0.5*((W15)+(W15P)+(H55))
#define HX6   (H6) + 0.5*((H4)-(H1)-(H3)-(H5)) - 0.5*(W13) + 0.5*(W14) \
              + 0.25*(W15) - 0.75*(W15P) + (W16) - 0.75*(H55)
#define HX7   (H7)-(H1) + (pH1) + 0.5*((WHCAFE)+(WHCAMG)-(WH12)) \
              + 0.25*((HEX)+(HX)+(H027)) + 0.5*(DWHCAMG) - 1.5*(DWHCAFE)

#define HS1   0.5*((W15)-(W15P)-(H55))
#define HS2   0.5*((WHCAFE)-(WHCAMG)-(WH12)) + 0.25*((HEX)+(HX)+(H027)) \
              - 0.5*(DWHCAMG) - 1.5*(DWHCAFE)

#define HX2X2 -(WH12)
#define HX2X3 2.0*(H23) - 0.5*(WH12)
#define HX2X4 2.0*(H24) - 0.5*(WH12)
#define HX2X5 0.5*((W25)+(W25P)-(W15)-(W15P)) - (WH12)
#define HX2X6 (W26)-(W16) + 0.25*((W25)-(W15)) - 0.75*((W25P)-(W15P)) \
              + (H24) - (H23) - 0.5*(WH12)
#define HX2X7 (pH2)-(pH1) + (WH12) + 0.5*((H027)-(HEX)) - 2.0*(DWHCAMG) \
              + 2.0*(DWHCAFE)
#define HX2S1 0.5*((W25)-(W25P)-(W15)+(W15P))
#define HX2S2 (WH12) - 0.5*(HX) + 2.0*(DWHCAMG) + 2.0*(DWHCAFE)

#define HX3X3 -(W13)
#define HX3X4 (W34) - (W13) - (W14)
#define HX3X5 0.5*((W35)+(W35P)-(W15)-(W15P)) - (W13)
#define HX3X6 (W36)-(W16) + 0.25*((W35)-(W15)) + 0.5*((W34)-(W14)) \
              - 0.75*((W35P)-(W15P))
#define HX3X7 0.125*((H027)-(HEX)-(HX))+0.5*((W37)+(W3U7U)-(WHCAFE)-(WHCAMG)) \
              - 1.5*(H23) - (W13) + 0.25*(WH12) - (DWHCAMG) + (DWHCAFE) \
              + (pH3)-(pH1)
#define HX3S1 0.5*((W35)-(W35P)-(W15)+(W15P))
#define HX3S2 0.125*((H027)-(HEX)-(HX))+0.5*((W3U7U)-(W37)+(WHCAMG)-(WHCAFE)) \
              - 1.5*(H23) + 0.25*(WH12) + (DWHCAMG) + (DWHCAFE)

#define HX4X4 -(W14)
#define HX4X5 0.5*((W45)+(W45P)-(W15)-(W15P)) - (W14)
#define HX4X6 (W46)-(W16)-(W14) + 0.25*((W45)-(W15)) - 0.5*((W34)-(W13)) \
              - 0.75*((W45P)-(W15P))
#define HX4X7 0.125*((H027)-(HEX)-(HX))+0.5*((W47)+(W4U7U)-(WHCAFE)-(WHCAMG)) \
              - 1.5*(H24) - (W14) + 0.25*(WH12) - (DWHCAMG) + (DWHCAFE) \
              + (pH4)-(pH1)
#define HX4S1 0.5*((W45)-(W45P)-(W15)+(W15P))
#define HX4S2 0.125*((H027)-(HEX)-(HX))+0.5*((W4U7U)-(W47)+(WHCAMG)-(WHCAFE)) \
              - 1.5*(H24) + 0.25*(WH12) + (DWHCAMG) + (DWHCAFE)

#define HX5X5 0.25*(W55) - 0.5*((W15)+(W15P))
#define HX5X6 0.5*((W56)+(W5P6)-(W15)+(W15P)-2.0*(W16)) - 0.25*(W55) \
              + 0.25*((W45)+(W45P)-2.0*(W14)) - 0.25*((W35)+(W35P)-2.0*(W13))
#define HX5X7 0.25*((H027)-(HEX)-(HX)) + 0.25*((W57U)+(W5P7U)+(W57)+(W5P7)) \
              - 0.5*((WHCAFE)+(WHCAMG)+(W25)+(W25P)-(WH12)) - (DWHCAMG) \
              + 2.0*(DWHCAFE) + (pH5)-(pH1)
#define HX5S1 - 0.5*((W15)-(W15P))
#define HX5S2 0.25*((H027)-(HEX)-(HX)) + 0.25*((W57U)+(W5P7U)-(W57)-(W5P7)) \
              + 0.5*((W15)+(W15P)-(W25)-(W25P)+(WHCAMG)-(WHCAFE)+(WH12)) \
              + (DWHCAMG) + 2.0*(DWHCAFE)

#define HX6X6 0.25*(W56) - 0.75*(W5P6) + 0.5*((W46)-(W36)-(W16)) \
              - 0.1875*(W55) + 0.125*((W45)-(W35)-(W15)) \
              - 0.375*((W45P)-(W35P)-(W15P)) - 0.25*((W34)+(W14)-(W13))
#define HX6X7 0.5*((W67)+(W67U)) - 0.375*((W5P7)+(W5P7U)) \
              + 0.125*((W57)+(W57U)) + 0.25*((W47)+(W4U7U)) \
              - 0.25*((W37)+(W3U7U)) - (W26) - 0.25*(W25) + 0.75*(W25P) \
              - 0.5*((W14)-(W13)) + 0.25*(WH12) - 0.25*((WHCAFE)+(WHCAMG)) \
              - 0.125*((HEX)+(HX)-(H027)) - 0.75*((H24)-(H23)) \
              - 0.50*(DWHCAMG) + (DWHCAFE) + (pHj)-0.5*((pH1)+(pH3)-(pH4)+(pH5))
#define HX6S1 0.5*((W56)-(W5P6)) + 0.25*((W45)-(W45P)) - 0.25*((W35)-(W35P)) \
              - 0.25*((W15)-(W15P)) - 0.5*(W55)
#define HX6S2 0.5*((W67U)-(W67)) + 0.125*((W57U)-(W57)) \
              + 0.375*((W5P7)-(W5P7U)) + 0.25*((W4U7U)-(W47)) \
              + 0.25*((W37)-(W3U7U)) + (W16)-(W26) + 0.75*((W25P)-(W15P)) \
              + 0.25*((W15)-(W25)) + 0.25*(WH12) - 0.25*((WHCAFE)-(WHCAMG)) \
              - 0.125*((HEX)+(HX)-(H027)) - 0.75*((H24)-(H23)) \
              + 0.50*(DWHCAMG) + (DWHCAFE)

#define HX7X7 0.25*((HEX)-(H027)) - 0.5*((WHCAFE)+(WHCAMG)) \
              + 0.25*((WHFEMG)-(WH12)) + 0.75*(DWHCAMG) + 0.25*(DWHCAFE) \
              + (pH7)-(pH1)+0.25*((dH027)+(dHEX)+(dHX))
#define HX7S1 0.25*((W57U)+(W57)-(W5P7U)-(W5P7)) - 0.5*((W25)-(W25P))
#define HX7S2 0.5*((WHCAMG)-(WHCAFE)-(WH12)) + 0.25*((HEX)+(HX)-(H027)) \
              + 1.5*(DWHCAMG) - 1.5*(DWHCAFE) + 0.25*((dH027)+(dHEX)+(dHX))    

#define HS1S1 -0.25*(W55)
#define HS1S2 0.25*((W57U)+(W5P7)-(W5P7U)-(W57)) \
              + 0.5*((W25P)-(W25)+(W15)-(W15P))

#define HS2S2 0.25*((HX)-(WHFEMG)-(WH12)) - 2.25*(DWHCAMG) - 1.75*(DWHCAFE)

#define HX2X2X7 - 0.5*(DWHCAMG) + 0.5*(DWHCAFE)
#define HX2X2S2 0.5*(DWHCAMG) - 0.5*(DWHCAFE)
#define HX2X3X7 (DWHCAMG) - (DWHCAFE)
#define HX2X3S2 - (DWHCAMG) - (DWHCAFE)
#define HX2X4X7 (DWHCAMG) - (DWHCAFE)
#define HX2X4S2 - (DWHCAMG) - (DWHCAFE)
#define HX2X5X7 (DWHCAMG) - (DWHCAFE)
#define HX2X5S2 - (DWHCAMG) - (DWHCAFE)
#define HX2X6X7 0.5*(DWHCAMG) - 0.5*(DWHCAFE)
#define HX2X6S2 - 0.5*(DWHCAMG) - 0.5*(DWHCAFE)
#define HX2X7X7 2.75*(DWHCAMG) - 2.75*(DWHCAFE) + 0.5*((dH027)-(dHEX))
#define HX2X7S2 - 2.5*(DWHCAMG) - 1.5*(DWHCAFE) - 0.5*(dHX)
#define HX2S2S2 - 0.25*(DWHCAMG) + 0.25*(DWHCAFE)
#define HX3X3X7 0.5*(DWHCAMG)
#define HX3X3S2 - 0.5*(DWHCAMG)
#define HX3X4X7 (DWHCAMG) + 0.25*(DWHCAFE)
#define HX3X4S2 - (DWHCAMG) + 0.25*(DWHCAFE)
#define HX3X5X7 (DWHCAMG) - 0.25*(DWHCAFE)
#define HX3X5S2 - (DWHCAMG) - 0.25*(DWHCAFE)
#define HX3X6X7 0.5*(DWHCAMG)
#define HX3X6S2 - 0.5*(DWHCAMG)
#define HX3X7X7 0.25*(DWHCAMG) + 0.625*(DWHCAFE) + 0.125*((dH027)-(dHEX)-(dHX))
#define HX3X7S2 - 1.5*(DWHCAMG) + 1.5*(DWHCAFE) + 0.125*((dH027)-(dHEX)-(dHX))
#define HX3S2S2 1.25*(DWHCAMG) + 0.875*(DWHCAFE)
#define HX4X4X7 0.5*(DWHCAMG)
#define HX4X4S2 - 0.5*(DWHCAMG)
#define HX4X5X7 (DWHCAMG) - 0.25*(DWHCAFE)
#define HX4X5S2 - (DWHCAMG) - 0.25*(DWHCAFE)
#define HX4X6X7 0.5*(DWHCAMG) - 0.25*(DWHCAFE)
#define HX4X6S2 - 0.5*(DWHCAMG) - 0.25*(DWHCAFE)
#define HX4X7X7 0.25*(DWHCAMG) + 0.625*(DWHCAFE) + 0.125*((dH027)-(dHEX)-(dHX))
#define HX4X7S2 - 1.5*(DWHCAMG) + 1.5*(DWHCAFE) + 0.125*((dH027)-(dHEX)-(dHX))
#define HX4S2S2 1.25*(DWHCAMG) + 0.875*(DWHCAFE)
#define HX5X5X7 0.5*(DWHCAMG) - 0.5*(DWHCAFE)
#define HX5X5S2 - 0.5*(DWHCAMG) - 0.5*(DWHCAFE)
#define HX5X6X7 0.5*(DWHCAMG) - 0.5*(DWHCAFE)
#define HX5X6S2 - 0.5*(DWHCAMG) - 0.5*(DWHCAFE)
#define HX5X7X7 0.25*(DWHCAMG) - 0.25*(DWHCAFE) + 0.25*((dH027)-(dHEX)-(dHX))
#define HX5X7S2 - 1.5*(DWHCAMG) + 0.5*(DWHCAFE) + 0.25*((dH027)-(dHEX)-(dHX))
#define HX5S2S2 1.25*(DWHCAMG) + 0.75*(DWHCAFE)
#define HX6X6X7 0.125*(DWHCAMG) - 0.1875*(DWHCAFE)
#define HX6X6S2 - 0.125*(DWHCAMG) - 0.1875*(DWHCAFE)
#define HX6X7X7 0.125*(DWHCAMG) - 0.125*(DWHCAFE) + 0.125*((dH027)-(dHEX)-(dHX))
#define HX6X7S2 - 0.75*(DWHCAMG) + 0.25*(DWHCAFE) + 0.125*((dH027)-(dHEX)-(dHX))
#define HX6S2S2 0.625*(DWHCAMG) + 0.375*(DWHCAFE)
#define HX7X7X7 - 1.5*(DWHCAMG) + 1.5*(DWHCAFE) + 0.25*((dHEX)-(dH027))
#define HX7X7S2 - (DWHCAMG) + 3.0*(DWHCAFE) + 0.25*((dHEX)+(dHX)-(dH027))
#define HX7S2S2 2.5*(DWHCAMG) + 1.5*(DWHCAFE) + 0.25*(dHX)

#define S0    (S1)
#define SX2   (S2) - (S1)
#define SX3   (S3) - (S1)
#define SX4   (S4) - (S1)
#define SX5   (S5) - (S1) + 0.5*(S55)
#define SX6   (S6) + 0.5*((S4)-(S1)-(S3)-(S5)) - 0.75*(S55)
#define SX7   (S7) - (S1) + (pS1) + 0.25*(S027)

#define SS1   -0.5*(S55)
#ifndef RHYOLITE_ADJUSTMENTS
#define SS2   ((calculationMode == MODE_xMELTS) ? 0.25*(S027) : 0.0)
#else
#define SS2   0.0
#endif

#define SX2X2 0.0
#define SX2X3 2.0*(S23)
#define SX2X4 2.0*(S24)
#define SX2X5 0.0
#define SX2X6 (S24) - (S23)
#define SX2X7 (pS2)-(pS1) + 0.5*(S027)
#define SX2S1 0.0
#define SX2S2 0.0

#define SX3X3 0.0
#define SX3X4 0.0
#define SX3X5 0.0
#define SX3X6 0.0
#define SX3X7 0.125*(S027) - 1.5*(S23) + (pS3)-(pS1)
#define SX3S1 0.0
#define SX3S2 0.125*(S027) - 1.5*(S23)

#define SX4X4 0.0
#define SX4X5 0.0
#define SX4X6 0.0
#define SX4X7 0.125*(S027) - 1.5*(S24) + (pS4)-(pS1)
#define SX4S1 0.0
#define SX4S2 0.125*(S027) - 1.5*(S24)

#define SX5X5 0.0
#define SX5X6 0.0
#define SX5X7 0.25*(S027) + (pS5)-(pS1) 
#define SX5S1 0.0
#define SX5S2 0.25*(S027) 

#define SX6X6 0.0
#define SX6X7 0.125*(S027) - 0.75*((S24)-(S23)) \
              + (pSj)-0.5*((pS1)+(pS3)-(pS4)+(pS5))
#define SX6S1 0.0
#define SX6S2 0.125*(S027) - 0.75*((S24)-(S23))

#define SX7X7 -0.25*(S027) + (pS7)-(pS1)+0.25*(dS027)
#define SX7S1 0.0
#define SX7S2 -0.25*(S027) + 0.25*(dS027)

#define SS1S1 0.0
#define SS1S2 0.0

#define SS2S2 0.0

#define SX2X7X7 0.5*(dS027)
#define SX3X7X7 0.125*(dS027)
#define SX3X7S2 0.125*(dS027)
#define SX4X7X7 0.125*(dS027)
#define SX4X7S2 0.125*(dS027)
#define SX5X7X7 0.25*(dS027)
#define SX5X7S2 0.25*(dS027)
#define SX6X7X7 0.125*(dS027)
#define SX6X7S2 0.125*(dS027)
#define SX7X7X7 -0.25*(dS027)
#define SX7X7S2 -0.25*(dS027)

#define V0    (V1)
#define VX2   (V2)-(V1) + (WV12)
#define VX3   (V3)-(V1)
#define VX4   (V4)-(V1)
#define VX5   (V5)-(V1)
#define VX6   (V6) + 0.5*((V4)-(V1)-(V3)-(V5))
#define VX7   (V7)-(V1) + (pV1) + 0.5*((WVCAFE)+(WVCAMG)-(WV12)) \
              + 0.25*((VEX)+(VX)+(V027)) + 0.5*(DWVCAMG) - 1.5*(DWVCAFE)

#define VS1   0.0
#define VS2   0.5*((WVCAFE)-(WVCAMG)-(WV12)) + 0.25*((VEX)+(VX)+(V027)) \
              - 0.5*(DWVCAMG) - 1.5*(DWVCAFE)

#define VX2X2 -(WV12)
#define VX2X3 -0.5*(WV12)
#define VX2X4 -0.5*(WV12)
#define VX2X5 -(WV12)
#define VX2X6 -0.5*(WV12)
#define VX2X7 (pV2)-(pV1) + (WV12) + 0.5*((V027)-(VEX)) - 2.0*(DWVCAMG) \
              + 2.0*(DWVCAFE)
#define VX2S1 0.0
#define VX2S2 (WV12) - 0.5*(VX) + 2.0*(DWVCAMG) + 2.0*(DWVCAFE)

#define VX3X3 0.0
#define VX3X4 0.0
#define VX3X5 0.0
#define VX3X6 0.0
#define VX3X7 0.125*((V027)-(VEX)-(VX)) - 0.5*((WVCAFE)+(WVCAMG)) \
              + 0.25*(WV12) - (DWVCAMG) + (DWVCAFE) + (pV3)-(pV1)
#define VX3S1 0.0
#define VX3S2 0.125*((V027)-(VEX)-(VX)) + 0.5*((WVCAMG)-(WVCAFE)) \
              + 0.25*(WV12) + (DWVCAMG) + (DWVCAFE)

#define VX4X4 0.0
#define VX4X5 0.0
#define VX4X6 0.0
#define VX4X7 0.125*((V027)-(VEX)-(VX)) - 0.5*((WVCAFE)+(WVCAMG)) \
              + 0.25*(WV12) - (DWVCAMG) + (DWVCAFE) + (pV4)-(pV1)
#define VX4S1 0.0
#define VX4S2 0.125*((V027)-(VEX)-(VX)) + 0.5*((WVCAMG)-(WVCAFE)) \
              + 0.25*(WV12) + (DWVCAMG) + (DWVCAFE)

#define VX5X5 0.0
#define VX5X6 0.0
#define VX5X7 0.25*((V027)-(VEX)-(VX)) - 0.5*((WVCAFE)+(WVCAMG)-(WV12)) \
              - (DWVCAMG) + 2.0*(DWVCAFE) + (pV5)-(pV1)
#define VX5S1 0.0
#define VX5S2 0.25*((V027)-(VEX)-(VX)) + 0.5*((WVCAMG)-(WVCAFE)+(WV12)) \
              + (DWVCAMG) + 2.0*(DWVCAFE)

#define VX6X6 0.0
#define VX6X7 0.25*(WV12) - 0.25*((WVCAFE)+(WVCAMG)) \
              - 0.125*((VEX)+(VX)-(V027)) - 0.5*(DWVCAMG) + (DWVCAFE) \
              + (pVj)-0.5*((pV1)+(pV3)-(pV4)+(pV5))
#define VX6S1 0.0
#define VX6S2 0.25*(WV12) - 0.25*((WVCAFE)-(WVCAMG)) \
              - 0.125*((VEX)+(VX)-(V027)) + 0.5*(DWVCAMG) + (DWVCAFE)

#define VX7X7 0.25*((VEX)-(V027)) - 0.5*((WVCAFE)+(WVCAMG)) \
              + 0.25*((WVFEMG)-(WV12)) + 0.75*(DWVCAMG) + 0.25*(DWVCAFE) \
              + (pV7)-(pV1)+0.25*((dV027)+(dVEX)+(dVX))
#define VX7S1 0.0
#define VX7S2 0.5*((WVCAMG)-(WVCAFE)-(WV12)) + 0.25*((VEX)+(VX)-(V027)) \
              + 1.5*(DWVCAMG) - 1.5*(DWVCAFE) + 0.25*((dV027)+(dVEX)+(dVX))

#define VS1S1 0.0
#define VS1S2 0.0

#define VS2S2 0.25*((VX)-(WVFEMG)-(WV12)) - 2.25*(DWVCAMG) - 1.75*(DWVCAFE)

#define VX2X2X7 - 0.5*(DWVCAMG) + 0.5*(DWVCAFE)
#define VX2X2S2 0.5*(DWVCAMG) - 0.5*(DWVCAFE)
#define VX2X3X7 (DWVCAMG) - (DWVCAFE)
#define VX2X3S2 - (DWVCAMG) - (DWVCAFE)
#define VX2X4X7 (DWVCAMG) - (DWVCAFE)
#define VX2X4S2 - (DWVCAMG) - (DWVCAFE)
#define VX2X5X7 (DWVCAMG) - (DWVCAFE)
#define VX2X5S2 - (DWVCAMG) - (DWVCAFE)
#define VX2X6X7 0.5*(DWVCAMG) - 0.5*(DWVCAFE)
#define VX2X6S2 - 0.5*(DWVCAMG) - 0.5*(DWVCAFE)
#define VX2X7X7 2.75*(DWVCAMG) - 2.75*(DWVCAFE) + 0.5*((dV027)-(dVEX))
#define VX2X7S2 - 2.5*(DWVCAMG) - 1.5*(DWVCAFE) - 0.5*(dVX)
#define VX2S2S2 - 0.25*(DWVCAMG) + 0.25*(DWVCAFE)
#define VX3X3X7 0.5*(DWVCAMG)
#define VX3X3S2 - 0.5*(DWVCAMG)
#define VX3X4X7 (DWVCAMG) + 0.25*(DWVCAFE)
#define VX3X4S2 - (DWVCAMG) + 0.25*(DWVCAFE)
#define VX3X5X7 (DWVCAMG) - 0.25*(DWVCAFE)
#define VX3X5S2 - (DWVCAMG) - 0.25*(DWVCAFE)
#define VX3X6X7 0.5*(DWVCAMG)
#define VX3X6S2 - 0.5*(DWVCAMG)
#define VX3X7X7 0.25*(DWVCAMG) + 0.625*(DWVCAFE) + 0.125*((dV027)-(dVEX)-(dVX))
#define VX3X7S2 - 1.5*(DWVCAMG) + 1.5*(DWVCAFE) + 0.125*((dV027)-(dVEX)-(dVX))
#define VX3S2S2 1.25*(DWVCAMG) + 0.875*(DWVCAFE)
#define VX4X4X7 0.5*(DWVCAMG)
#define VX4X4S2 - 0.5*(DWVCAMG)
#define VX4X5X7 (DWVCAMG) - 0.25*(DWVCAFE)
#define VX4X5S2 - (DWVCAMG) - 0.25*(DWVCAFE)
#define VX4X6X7 0.5*(DWVCAMG) - 0.25*(DWVCAFE)
#define VX4X6S2 - 0.5*(DWVCAMG) - 0.25*(DWVCAFE)
#define VX4X7X7 0.25*(DWVCAMG) + 0.625*(DWVCAFE) + 0.125*((dV027)-(dVEX)-(dVX))
#define VX4X7S2 - 1.5*(DWVCAMG) + 1.5*(DWVCAFE) + 0.125*((dV027)-(dVEX)-(dVX))
#define VX4S2S2 1.25*(DWVCAMG) + 0.875*(DWVCAFE)
#define VX5X5X7 0.5*(DWVCAMG) - 0.5*(DWVCAFE)
#define VX5X5S2 - 0.5*(DWVCAMG) - 0.5*(DWVCAFE)
#define VX5X6X7 0.5*(DWVCAMG) - 0.5*(DWVCAFE)
#define VX5X6S2 - 0.5*(DWVCAMG) - 0.5*(DWVCAFE)
#define VX5X7X7 0.25*(DWVCAMG) - 0.25*(DWVCAFE) + 0.25*((dV027)-(dVEX)-(dVX))
#define VX5X7S2 - 1.5*(DWVCAMG) + 0.5*(DWVCAFE) + 0.25*((dV027)-(dVEX)-(dVX))
#define VX5S2S2 1.25*(DWVCAMG) + 0.75*(DWVCAFE)
#define VX6X6X7 0.125*(DWVCAMG) - 0.1875*(DWVCAFE)
#define VX6X6S2 - 0.125*(DWVCAMG) - 0.1875*(DWVCAFE)
#define VX6X7X7 0.125*(DWVCAMG) - 0.125*(DWVCAFE) + 0.125*((dV027)-(dVEX)-(dVX))
#define VX6X7S2 - 0.75*(DWVCAMG) + 0.25*(DWVCAFE) + 0.125*((dV027)-(dVEX)-(dVX))
#define VX6S2S2 0.625*(DWVCAMG) + 0.375*(DWVCAFE)
#define VX7X7X7 - 1.5*(DWVCAMG) + 1.5*(DWVCAFE) + 0.25*((dVEX)-(dV027))
#define VX7X7S2 - (DWVCAMG) + 3.0*(DWVCAFE) + 0.25*((dVEX)+(dVX)-(dV027))
#define VX7S2S2 2.5*(DWVCAMG) + 1.5*(DWVCAFE) + 0.25*(dVX)

/*
 *=============================================================================
 * Local function to compute ordering state and associated derivatives of
 * pure component endmembers
 */

#define DI_S          (S0)
#define DI_H          (H0)
#define DI_G          (H0)-t*(S0)+(p-1.0)*(V0)

#define DDI_GDT       -(S0)
#define DDI_GDP       (V0)

#define D2DI_GDT2     0.0
#define D2DI_GDTP     0.0
#define D2DI_GDP2     0.0

#define D3DI_GDT3     0.0
#define D3DI_GDT2DP   0.0
#define D3DI_GDTDP2   0.0
#define D3DI_GDP3     0.0

#define EN_S          (S0)+(SX7)-(SS2)+(SX7X7)-(SX7S2)+(SS2S2)+(SX7X7X7) \
                      -(SX7X7S2)
#define EN_H          (H0)+(HX7)-(HS2)+(HX7X7)-(HX7S2)+(HS2S2)+(HX7X7X7) \
                      -(HX7X7S2)+(HX7S2S2)  
#define EN_G          (H0)+(HX7)-(HS2)+(HX7X7)-(HX7S2)+(HS2S2)+(HX7X7X7) \
                      -(HX7X7S2)+(HX7S2S2) -t*((S0)+(SX7)-(SS2)+(SX7X7)- \
                      (SX7S2)+(SS2S2)+(SX7X7X7)-(SX7X7S2)) +(p-1.0)*((V0)+ \
                      (VX7)-(VS2)+(VX7X7)-(VX7S2)+(VS2S2)+(VX7X7X7)-(VX7X7S2)+ \
                      (VX7S2S2))

#define DEN_GDT       -(EN_S)
#define DEN_GDP       (V0)+(VX7)-(VS2)+(VX7X7)-(VX7S2)+(VS2S2)+(VX7X7X7) \
                      -(VX7X7S2)+(VX7S2S2)  

#define D2EN_GDT2     0.0
#define D2EN_GDTP     0.0
#define D2EN_GDP2     0.0

#define D3EN_GDT3     0.0
#define D3EN_GDT2DP   0.0
#define D3EN_GDTDP2   0.0
#define D3EN_GDP3     0.0

#define HD_S          (S0)+(SX2)+(SX2X2)
#define HD_H          (H0)+(HX2)+(HX2X2)
#define HD_G          (H0)+(HX2)+(HX2X2) -t*((S0)+(SX2)+(SX2X2)) \
                      + (p-1.0)*((V0)+(VX2)+(VX2X2))

#define DHD_GDT       -(HD_S)
#define DHD_GDP       (V0)+(VX2)+(VX2X2)

#define D2HD_GDT2     0.0
#define D2HD_GDTP     0.0
#define D2HD_GDP2     0.0

#define D3HD_GDT3     0.0
#define D3HD_GDT2DP   0.0
#define D3HD_GDTDP2   0.0
#define D3HD_GDP3     0.0

#define CA_S          (S0)+(SX3)+(SX3X3)
#define CA_H          (H0)+(HX3)+(HX3X3)
#define CA_G          (H0)+(HX3)+(HX3X3) -t*((S0)+(SX3)+(SX3X3)) \
                      + (p-1.0)*((V0)+(VX3)+(VX3X3))

#define DCA_GDT       -(CA_S)
#define DCA_GDP       (V0)+(VX3)+(VX3X3)

#define D2CA_GDT2     0.0
#define D2CA_GDTP     0.0
#define D2CA_GDP2     0.0

#define D3CA_GDT3     0.0
#define D3CA_GDT2DP   0.0
#define D3CA_GDTDP2   0.0
#define D3CA_GDP3     0.0

#define CF_S          (S0)+(SX4)+(SX4X4)
#define CF_H          (H0)+(HX4)+(HX4X4)
#define CF_G          (H0)+(HX4)+(HX4X4) - t*((S0)+(SX4)+(SX4X4)) \
                      + (p-1.0)*((V0)+(VX4)+(VX4X4))

#define DCF_GDT       -(CF_S)
#define DCF_GDP       (V0)+(VX4)+(VX4X4)

#define D2CF_GDT2     0.0
#define D2CF_GDTP     0.0
#define D2CF_GDP2     0.0

#define D3CF_GDT3     0.0
#define D3CF_GDT2DP   0.0
#define D3CF_GDTDP2   0.0
#define D3CF_GDP3     0.0

#define ES_S          -R*((1.0-s)*log(1.0-s) + (1.0+s)*log(1.0+s) \
                        - 2.0*log(2.0)) + \
                      (S0)+(SX5)+(SS1)*s+(SX5X5)+(SX5S1)*s+(SS1S1)*s*s
#define ES_H          (H0)+(HX5)+(HS1)*s+(HX5X5)+(HX5S1)*s+(HS1S1)*s*s
#define ES_G          R*t*((1.0-s)*log(1.0-s) + (1.0+s)*log(1.0+s) \
                        - 2.0*log(2.0)) + \
                      (H0)+(HX5)+(HS1)*s+(HX5X5)+(HX5S1)*s+(HS1S1)*s*s -t*( \
                      (S0)+(SX5)+(SS1)*s+(SX5X5)+(SX5S1)*s+(SS1S1)*s*s) + \
                      (p-1.0)*((V0)+(VX5)+(VS1)*s+(VX5X5)+(VX5S1)*s+(VS1S1)*s*s)

#define DES_GDS1      R*t*(log(1.0+s)-log(1.0-s)) + (HS1) + (HX5S1) \
                      + (HS1S1)*s*2.0 - t*((SS1)+(SX5S1)+(SS1S1)*s*2.0) \
                      + (p-1.0)*((VS1)+(VX5S1)+(VS1S1)*s*2.0)
#define DES_GDT       -(ES_S)
#define DES_GDP       (V0)+(VX5)+(VS1)*s+(VX5X5)+(VX5S1)*s+(VS1S1)*s*s

#define D2ES_GDS1S1   R*t*(1.0/(1.0+s) + 1.0/(1.0-s)) + (HS1S1)*2.0 \
                      - t*(SS1S1)*2.0 + (p-1.0)*(VS1S1)*2.0
#define D2ES_GDS1DT   R*(log(1.0+s)-log(1.0-s)) - ((SS1)+(SX5S1)+(SS1S1)*s*2.0)
#define D2ES_GDS1DP   (VS1)+(VX5S1)+(VS1S1)*s*2.0
#define D2ES_GDT2     0.0
#define D2ES_GDTP     0.0
#define D2ES_GDP2     0.0

#define D3ES_GDS1S1S1 R*t*(1.0/SQUARE(1.0-s) - 1.0/SQUARE(1.0+s))
#define D3ES_GDS1S1DT R*(1.0/(1.0+s) + 1.0/(1.0-s)) - (SS1S1)*2.0
#define D3ES_GDS1S1DP (VS1S1)*2.0
#define D3ES_GDS1DT2  0.0
#define D3ES_GDS1DTDP 0.0
#define D3ES_GDS1DP2  0.0
#define D3ES_GDT3     0.0
#define D3ES_GDT2DP   0.0
#define D3ES_GDTDP2   0.0
#define D3ES_GDP3     0.0

#define JD_S          (S0) + 0.5*(SX3) - 0.5*(SX4) + 0.5*(SX5) + (SX6) \
                      - (SS1) + 0.25*(SX3X3) - 0.25*(SX3X4) + 0.25*(SX3X5) \
                      + 0.5*(SX3X6) - 0.5*(SX3S1) + 0.25*(SX4X4) \
                      - 0.25*(SX4X5) - 0.5*(SX4X6) + 0.5*(SX4S1) \
                      + 0.25*(SX5X5) + 0.5*(SX5X6) - 0.5*(SX5S1) \
                      + (SX6X6) - (SX6S1) + (SS1S1)
#define JD_H          (H0) + 0.5*(HX3) - 0.5*(HX4) + 0.5*(HX5) + (HX6) \
                      - (HS1) + 0.25*(HX3X3) - 0.25*(HX3X4) + 0.25*(HX3X5) \
                      + 0.5*(HX3X6) - 0.5*(HX3S1) + 0.25*(HX4X4) \
                      - 0.25*(HX4X5) - 0.5*(HX4X6) + 0.5*(HX4S1) \
                      + 0.25*(HX5X5) + 0.5*(HX5X6) - 0.5*(HX5S1) \
                      + (HX6X6) - (HX6S1) + (HS1S1)
#define JD_G          (H0)-t*(S0)+(p-1.0)*(V0) \
                + 0.5*((HX3)-t*(SX3)+(p-1.0)*(VX3)) \
                - 0.5*((HX4)-t*(SX4)+(p-1.0)*(VX4)) \
                + 0.5*((HX5)-t*(SX5)+(p-1.0)*(VX5)) \
                    + ((HX6)-t*(SX6)+(p-1.0)*(VX6)) \
                    - ((HS1)-t*(SS1)+(p-1.0)*(VS1)) \
               + 0.25*((HX3X3)-t*(SX3X3)+(p-1.0)*(VX3X3)) \
               - 0.25*((HX3X4)-t*(SX3X4)+(p-1.0)*(VX3X4)) \
               + 0.25*((HX3X5)-t*(SX3X5)+(p-1.0)*(VX3X5)) \
                + 0.5*((HX3X6)-t*(SX3X6)+(p-1.0)*(VX3X6)) \
                - 0.5*((HX3S1)-t*(SX3S1)+(p-1.0)*(VX3S1)) \
               + 0.25*((HX4X4)-t*(SX4X4)+(p-1.0)*(VX4X4)) \
               - 0.25*((HX4X5)-t*(SX4X5)+(p-1.0)*(VX4X5)) \
                - 0.5*((HX4X6)-t*(SX4X6)+(p-1.0)*(VX4X6)) \
                + 0.5*((HX4S1)-t*(SX4S1)+(p-1.0)*(VX4S1)) \
               + 0.25*((HX5X5)-t*(SX5X5)+(p-1.0)*(VX5X5)) \
                + 0.5*((HX5X6)-t*(SX5X6)+(p-1.0)*(VX5X6)) \
                - 0.5*((HX5S1)-t*(SX5S1)+(p-1.0)*(VX5S1)) \
                    + ((HX6X6)-t*(SX6X6)+(p-1.0)*(VX6X6)) \
                    - ((HX6S1)-t*(SX6S1)+(p-1.0)*(VX6S1)) \
                    + ((HS1S1)-t*(SS1S1)+(p-1.0)*(VS1S1)) 

#define DJD_GDT       -(JD_S)
#define DJD_GDP       (V0) + 0.5*(VX3) - 0.5*(VX4) + 0.5*(VX5) + (VX6) \
                      - (VS1) + 0.25*(VX3X3) - 0.25*(VX3X4) + 0.25*(VX3X5) \
                      + 0.5*(VX3X6) - 0.5*(VX3S1) + 0.25*(VX4X4) \
                      - 0.25*(VX4X5) - 0.5*(VX4X6) + 0.5*(VX4S1) \
                      + 0.25*(VX5X5) + 0.5*(VX5X6) - 0.5*(VX5S1) \
                      + (VX6X6) - (VX6S1) + (VS1S1)

#define D2JD_GDT2     0.0
#define D2JD_GDTP     0.0
#define D2JD_GDP2     0.0

#define D3JD_GDT3     0.0
#define D3JD_GDT2DP   0.0
#define D3JD_GDTDP2   0.0
#define D3JD_GDP3     0.0
#undef D3JD_GDP3

/*
 * Global (to this file): activity definitions and component transforms
 *    The function conCpx defines the conversion from m[i], to r[j]
 */
                   /* Order: X2, X3. X4, X5, X6, X7 */
#define FR2(i)     (i == 2) ? 1.0 - r[0] : - r[0]
#define FR3(i)     (i == 3) ? 1.0 - r[1] : ((i == 6) ?   0.5 - r[1] : - r[1])
#define FR4(i)     (i == 4) ? 1.0 - r[2] : ((i == 6) ? - 0.5 - r[2] : - r[2])
#define FR5(i)     (i == 5) ? 1.0 - r[3] : ((i == 6) ?   0.5 - r[3] : - r[3])
#define FR6(i)     (i == 6) ? 1.0 - r[4] : - r[4]
#define FR7(i)     (i == 1) ? 1.0 - r[5] : - r[5]

                   /* Order: S1, S2 */
#define FS1(i)     (i == 5) ?   1.0 - s[0] : ((i == 6) ? - 1.0 - s[0] : - s[0])
#define FS2(i)     (i == 1) ? - 1.0 - s[1] : - s[1]

#define DFR2DR2(i) - 1.0
#define DFR3DR3(i) - 1.0
#define DFR4DR4(i) - 1.0
#define DFR5DR5(i) - 1.0
#define DFR6DR6(i) - 1.0
#define DFR7DR7(i) - 1.0

#define DFS1DS1(i) - 1.0
#define DFS2DS2(i) - 1.0

#define ENDMEMBERS  (  (1.0-r[0]-r[1]-r[2]-r[3]-r[4]/2.0-r[5])*ends[0] \
                     + r[5]*ends[1] + r[0]*ends[2] + (r[1]-r[4]/2.0)*ends[3] \
                     + (r[2]+r[4]/2.0)*ends[4] + (r[3]-r[4]/2.0)*ends[5] \
                     + r[4]*ends[6] )

#define DENDDR0  (ends[2] - ends[0])
#define DENDDR1  (ends[3] - ends[0])
#define DENDDR2  (ends[4] - ends[0])
#define DENDDR3  (ends[5] - ends[0])
#define DENDDR4  (ends[6] + 0.5*(ends[4] - ends[0] - ends[3] - ends[5]))
#define DENDDR5  (ends[1] - ends[0])

/*
 * Global (to this file): derivative definitions
 */

#define SIC -R*(xmg2m1*log(xmg2m1) \
              + xfe2m1*log(xfe2m1) \
              + xal3m1*log(xal3m1) \
              + xfe3m1*log(xfe3m1) \
              + xti4m1*log(xti4m1) \
              + (xmg2m1+xfe2m1-xti4m1)*log(xmg2m1+xfe2m1-xti4m1) \
              - (1.0-xti4m1)*log(1.0-xti4m1) \
              - (xmg2m1+xfe2m1)*log(xmg2m1+xfe2m1) \
              - 2.0*(1.0-xsi4tet)*log(1.0-xsi4tet) \
              + 2.0*xal3tet*log(xal3tet) \
              + 2.0*xfe3tet*log(xfe3tet) \
              + xca2m2*log(xca2m2) \
              + xna1m2*log(xna1m2) \
              + xmg2m2*log(xmg2m2) \
              + xfe2m2*log(xfe2m2) \
              + (1.0-xmg2m1-xfe2m1-xna1m2)*log(1.0-xmg2m1-xfe2m1-xna1m2) \
              - (1.0-xmg2m1-xfe2m1)*log(1.0-xmg2m1-xfe2m1) \
              - (1.0-xna1m2)*log(1.0-xna1m2) )
#define S   (SIC) + (S0) + \
            (SX2)*r[0] + (SX3)*r[1] + (SX4)*r[2] + (SX5)*r[3] + \
            (SX6)*r[4] + (SX7)*r[5] + (SS1)*s[0] + (SS2)*s[1] + \
            (SX2X2)*r[0]*r[0] + (SX2X3)*r[0]*r[1] + (SX2X4)*r[0]*r[2] + \
            (SX2X5)*r[0]*r[3] + (SX2X6)*r[0]*r[4] + (SX2X7)*r[0]*r[5] + \
            (SX2S1)*r[0]*s[0] + (SX2S2)*r[0]*s[1] + (SX3X3)*r[1]*r[1] + \
            (SX3X4)*r[1]*r[2] + (SX3X5)*r[1]*r[3] + (SX3X6)*r[1]*r[4] + \
            (SX3X7)*r[1]*r[5] + (SX3S1)*r[1]*s[0] + (SX3S2)*r[1]*s[1] + \
            (SX4X4)*r[2]*r[2] + (SX4X5)*r[2]*r[3] + (SX4X6)*r[2]*r[4] + \
            (SX4X7)*r[2]*r[5] + (SX4S1)*r[2]*s[0] + (SX4S2)*r[2]*s[1] + \
            (SX5X5)*r[3]*r[3] + (SX5X6)*r[3]*r[4] + (SX5X7)*r[3]*r[5] + \
            (SX5S1)*r[3]*s[0] + (SX5S2)*r[3]*s[1] + (SX6X6)*r[4]*r[4] + \
            (SX6X7)*r[4]*r[5] + (SX6S1)*r[4]*s[0] + (SX6S2)*r[4]*s[1] + \
            (SX7X7)*r[5]*r[5] + (SX7S1)*r[5]*s[0] + (SX7S2)*r[5]*s[1] + \
            (SS1S1)*s[0]*s[0] + (SS1S2)*s[0]*s[1] + (SS2S2)*s[1]*s[1] + \
            (SX2X7X7)*r[0]*r[5]*r[5] + (SX3X7X7)*r[1]*r[5]*r[5] + \
            (SX3X7S2)*r[1]*r[5]*s[1] + (SX4X7X7)*r[2]*r[5]*r[5] + \
            (SX4X7S2)*r[2]*r[5]*s[1] + (SX5X7X7)*r[3]*r[5]*r[5] + \
            (SX5X7S2)*r[3]*r[5]*s[1] + (SX6X7X7)*r[4]*r[5]*r[5] + \
            (SX6X7S2)*r[4]*r[5]*s[1] + (SX7X7X7)*r[5]*r[5]*r[5] + \
            (SX7X7S2)*r[5]*r[5]*s[1]
#define H   (H0) + \
            (HX2)*r[0] + (HX3)*r[1] + (HX4)*r[2] + (HX5)*r[3] + \
            (HX6)*r[4] + (HX7)*r[5] + (HS1)*s[0] + (HS2)*s[1] + \
            (HX2X2)*r[0]*r[0] + (HX2X3)*r[0]*r[1] + (HX2X4)*r[0]*r[2] + \
            (HX2X5)*r[0]*r[3] + (HX2X6)*r[0]*r[4] + (HX2X7)*r[0]*r[5] + \
            (HX2S1)*r[0]*s[0] + (HX2S2)*r[0]*s[1] + (HX3X3)*r[1]*r[1] + \
            (HX3X4)*r[1]*r[2] + (HX3X5)*r[1]*r[3] + (HX3X6)*r[1]*r[4] + \
            (HX3X7)*r[1]*r[5] + (HX3S1)*r[1]*s[0] + (HX3S2)*r[1]*s[1] + \
            (HX4X4)*r[2]*r[2] + (HX4X5)*r[2]*r[3] + (HX4X6)*r[2]*r[4] + \
            (HX4X7)*r[2]*r[5] + (HX4S1)*r[2]*s[0] + (HX4S2)*r[2]*s[1] + \
            (HX5X5)*r[3]*r[3] + (HX5X6)*r[3]*r[4] + (HX5X7)*r[3]*r[5] + \
            (HX5S1)*r[3]*s[0] + (HX5S2)*r[3]*s[1] + (HX6X6)*r[4]*r[4] + \
            (HX6X7)*r[4]*r[5] + (HX6S1)*r[4]*s[0] + (HX6S2)*r[4]*s[1] + \
            (HX7X7)*r[5]*r[5] + (HX7S1)*r[5]*s[0] + (HX7S2)*r[5]*s[1] + \
            (HS1S1)*s[0]*s[0] + (HS1S2)*s[0]*s[1] + (HS2S2)*s[1]*s[1] + \
            (HX2X2X7)*r[0]*r[0]*r[5] + (HX2X2S2)*r[0]*r[0]*s[1] + \
            (HX2X3X7)*r[0]*r[1]*r[5] + (HX2X3S2)*r[0]*r[1]*s[1] + \
            (HX2X4X7)*r[0]*r[2]*r[5] + (HX2X4S2)*r[0]*r[2]*s[1] + \
            (HX2X5X7)*r[0]*r[3]*r[5] + (HX2X5S2)*r[0]*r[3]*s[1] + \
            (HX2X6X7)*r[0]*r[4]*r[5] + (HX2X6S2)*r[0]*r[4]*s[1] + \
            (HX2X7X7)*r[0]*r[5]*r[5] + (HX2X7S2)*r[0]*r[5]*s[1] + \
            (HX2S2S2)*r[0]*s[1]*s[1] + (HX3X3X7)*r[1]*r[1]*r[5] + \
            (HX3X3S2)*r[1]*r[1]*s[1] + (HX3X4X7)*r[1]*r[2]*r[5] + \
            (HX3X4S2)*r[1]*r[2]*s[1] + (HX3X5X7)*r[1]*r[3]*r[5] + \
            (HX3X5S2)*r[1]*r[3]*s[1] + (HX3X6X7)*r[1]*r[4]*r[5] + \
            (HX3X6S2)*r[1]*r[4]*s[1] + (HX3X7X7)*r[1]*r[5]*r[5] + \
            (HX3X7S2)*r[1]*r[5]*s[1] + (HX3S2S2)*r[1]*s[1]*s[1] + \
            (HX4X4X7)*r[2]*r[2]*r[5] + (HX4X4S2)*r[2]*r[2]*s[1] + \
            (HX4X5X7)*r[2]*r[3]*r[5] + (HX4X5S2)*r[2]*r[3]*s[1] + \
            (HX4X6X7)*r[2]*r[4]*r[5] + (HX4X6S2)*r[2]*r[4]*s[1] + \
            (HX4X7X7)*r[2]*r[5]*r[5] + (HX4X7S2)*r[2]*r[5]*s[1] + \
            (HX4S2S2)*r[2]*s[1]*s[1] + (HX5X5X7)*r[3]*r[3]*r[5] + \
            (HX5X5S2)*r[3]*r[3]*s[1] + (HX5X6X7)*r[3]*r[4]*r[5] + \
            (HX5X6S2)*r[3]*r[4]*s[1] + (HX5X7X7)*r[3]*r[5]*r[5] + \
            (HX5X7S2)*r[3]*r[5]*s[1] + (HX5S2S2)*r[3]*s[1]*s[1] + \
            (HX6X6X7)*r[4]*r[4]*r[5] + (HX6X6S2)*r[4]*r[4]*s[1] + \
            (HX6X7X7)*r[4]*r[5]*r[5] + (HX6X7S2)*r[4]*r[5]*s[1] + \
            (HX6S2S2)*r[4]*s[1]*s[1] + (HX7X7X7)*r[5]*r[5]*r[5] + \
            (HX7X7S2)*r[5]*r[5]*s[1] + (HX7S2S2)*r[5]*s[1]*s[1]  
#define V   (V0) + \
            (VX2)*r[0] + (VX3)*r[1] + (VX4)*r[2] + (VX5)*r[3] + \
            (VX6)*r[4] + (VX7)*r[5] + (VS1)*s[0] + (VS2)*s[1] + \
            (VX2X2)*r[0]*r[0] + (VX2X3)*r[0]*r[1] + (VX2X4)*r[0]*r[2] + \
            (VX2X5)*r[0]*r[3] + (VX2X6)*r[0]*r[4] + (VX2X7)*r[0]*r[5] + \
            (VX2S1)*r[0]*s[0] + (VX2S2)*r[0]*s[1] + (VX3X3)*r[1]*r[1] + \
            (VX3X4)*r[1]*r[2] + (VX3X5)*r[1]*r[3] + (VX3X6)*r[1]*r[4] + \
            (VX3X7)*r[1]*r[5] + (VX3S1)*r[1]*s[0] + (VX3S2)*r[1]*s[1] + \
            (VX4X4)*r[2]*r[2] + (VX4X5)*r[2]*r[3] + (VX4X6)*r[2]*r[4] + \
            (VX4X7)*r[2]*r[5] + (VX4S1)*r[2]*s[0] + (VX4S2)*r[2]*s[1] + \
            (VX5X5)*r[3]*r[3] + (VX5X6)*r[3]*r[4] + (VX5X7)*r[3]*r[5] + \
            (VX5S1)*r[3]*s[0] + (VX5S2)*r[3]*s[1] + (VX6X6)*r[4]*r[4] + \
            (VX6X7)*r[4]*r[5] + (VX6S1)*r[4]*s[0] + (VX6S2)*r[4]*s[1] + \
            (VX7X7)*r[5]*r[5] + (VX7S1)*r[5]*s[0] + (VX7S2)*r[5]*s[1] + \
            (VS1S1)*s[0]*s[0] + (VS1S2)*s[0]*s[1] + (VS2S2)*s[1]*s[1] + \
            (VX2X2X7)*r[0]*r[0]*r[5] + (VX2X2S2)*r[0]*r[0]*s[1] + \
            (VX2X3X7)*r[0]*r[1]*r[5] + (VX2X3S2)*r[0]*r[1]*s[1] + \
            (VX2X4X7)*r[0]*r[2]*r[5] + (VX2X4S2)*r[0]*r[2]*s[1] + \
            (VX2X5X7)*r[0]*r[3]*r[5] + (VX2X5S2)*r[0]*r[3]*s[1] + \
            (VX2X6X7)*r[0]*r[4]*r[5] + (VX2X6S2)*r[0]*r[4]*s[1] + \
            (VX2X7X7)*r[0]*r[5]*r[5] + (VX2X7S2)*r[0]*r[5]*s[1] + \
            (VX2S2S2)*r[0]*s[1]*s[1] + (VX3X3X7)*r[1]*r[1]*r[5] + \
            (VX3X3S2)*r[1]*r[1]*s[1] + (VX3X4X7)*r[1]*r[2]*r[5] + \
            (VX3X4S2)*r[1]*r[2]*s[1] + (VX3X5X7)*r[1]*r[3]*r[5] + \
            (VX3X5S2)*r[1]*r[3]*s[1] + (VX3X6X7)*r[1]*r[4]*r[5] + \
            (VX3X6S2)*r[1]*r[4]*s[1] + (VX3X7X7)*r[1]*r[5]*r[5] + \
            (VX3X7S2)*r[1]*r[5]*s[1] + (VX3S2S2)*r[1]*s[1]*s[1] + \
            (VX4X4X7)*r[2]*r[2]*r[5] + (VX4X4S2)*r[2]*r[2]*s[1] + \
            (VX4X5X7)*r[2]*r[3]*r[5] + (VX4X5S2)*r[2]*r[3]*s[1] + \
            (VX4X6X7)*r[2]*r[4]*r[5] + (VX4X6S2)*r[2]*r[4]*s[1] + \
            (VX4X7X7)*r[2]*r[5]*r[5] + (VX4X7S2)*r[2]*r[5]*s[1] + \
            (VX4S2S2)*r[2]*s[1]*s[1] + (VX5X5X7)*r[3]*r[3]*r[5] + \
            (VX5X5S2)*r[3]*r[3]*s[1] + (VX5X6X7)*r[3]*r[4]*r[5] + \
            (VX5X6S2)*r[3]*r[4]*s[1] + (VX5X7X7)*r[3]*r[5]*r[5] + \
            (VX5X7S2)*r[3]*r[5]*s[1] + (VX5S2S2)*r[3]*s[1]*s[1] + \
            (VX6X6X7)*r[4]*r[4]*r[5] + (VX6X6S2)*r[4]*r[4]*s[1] + \
            (VX6X7X7)*r[4]*r[5]*r[5] + (VX6X7S2)*r[4]*r[5]*s[1] + \
            (VX6S2S2)*r[4]*s[1]*s[1] + (VX7X7X7)*r[5]*r[5]*r[5] + \
            (VX7X7S2)*r[5]*r[5]*s[1] + (VX7S2S2)*r[5]*s[1]*s[1]  
#define G   -t*(SIC) + (H0)-t*(S0)+(p-1.0)*(V0) + \
            ((HX2)-t*(SX2)+(p-1.0)*(VX2))*r[0] + \
            ((HX3)-t*(SX3)+(p-1.0)*(VX3))*r[1] + \
            ((HX4)-t*(SX4)+(p-1.0)*(VX4))*r[2] + \
            ((HX5)-t*(SX5)+(p-1.0)*(VX5))*r[3] + \
            ((HX6)-t*(SX6)+(p-1.0)*(VX6))*r[4] + \
            ((HX7)-t*(SX7)+(p-1.0)*(VX7))*r[5] + \
            ((HS1)-t*(SS1)+(p-1.0)*(VS1))*s[0] + \
            ((HS2)-t*(SS2)+(p-1.0)*(VS2))*s[1] + \
            ((HX2X2)-t*(SX2X2)+(p-1.0)*(VX2X2))*r[0]*r[0] + \
            ((HX2X3)-t*(SX2X3)+(p-1.0)*(VX2X3))*r[0]*r[1] + \
            ((HX2X4)-t*(SX2X4)+(p-1.0)*(VX2X4))*r[0]*r[2] + \
            ((HX2X5)-t*(SX2X5)+(p-1.0)*(VX2X5))*r[0]*r[3] + \
            ((HX2X6)-t*(SX2X6)+(p-1.0)*(VX2X6))*r[0]*r[4] + \
            ((HX2X7)-t*(SX2X7)+(p-1.0)*(VX2X7))*r[0]*r[5] + \
            ((HX2S1)-t*(SX2S1)+(p-1.0)*(VX2S1))*r[0]*s[0] + \
            ((HX2S2)-t*(SX2S2)+(p-1.0)*(VX2S2))*r[0]*s[1] + \
            ((HX3X3)-t*(SX3X3)+(p-1.0)*(VX3X3))*r[1]*r[1] + \
            ((HX3X4)-t*(SX3X4)+(p-1.0)*(VX3X4))*r[1]*r[2] + \
            ((HX3X5)-t*(SX3X5)+(p-1.0)*(VX3X5))*r[1]*r[3] + \
            ((HX3X6)-t*(SX3X6)+(p-1.0)*(VX3X6))*r[1]*r[4] + \
            ((HX3X7)-t*(SX3X7)+(p-1.0)*(VX3X7))*r[1]*r[5] + \
            ((HX3S1)-t*(SX3S1)+(p-1.0)*(VX3S1))*r[1]*s[0] + \
            ((HX3S2)-t*(SX3S2)+(p-1.0)*(VX3S2))*r[1]*s[1] + \
            ((HX4X4)-t*(SX4X4)+(p-1.0)*(VX4X4))*r[2]*r[2] + \
            ((HX4X5)-t*(SX4X5)+(p-1.0)*(VX4X5))*r[2]*r[3] + \
            ((HX4X6)-t*(SX4X6)+(p-1.0)*(VX4X6))*r[2]*r[4] + \
            ((HX4X7)-t*(SX4X7)+(p-1.0)*(VX4X7))*r[2]*r[5] + \
            ((HX4S1)-t*(SX4S1)+(p-1.0)*(VX4S1))*r[2]*s[0] + \
            ((HX4S2)-t*(SX4S2)+(p-1.0)*(VX4S2))*r[2]*s[1] + \
            ((HX5X5)-t*(SX5X5)+(p-1.0)*(VX5X5))*r[3]*r[3] + \
            ((HX5X6)-t*(SX5X6)+(p-1.0)*(VX5X6))*r[3]*r[4] + \
            ((HX5X7)-t*(SX5X7)+(p-1.0)*(VX5X7))*r[3]*r[5] + \
            ((HX5S1)-t*(SX5S1)+(p-1.0)*(VX5S1))*r[3]*s[0] + \
            ((HX5S2)-t*(SX5S2)+(p-1.0)*(VX5S2))*r[3]*s[1] + \
            ((HX6X6)-t*(SX6X6)+(p-1.0)*(VX6X6))*r[4]*r[4] + \
            ((HX6X7)-t*(SX6X7)+(p-1.0)*(VX6X7))*r[4]*r[5] + \
            ((HX6S1)-t*(SX6S1)+(p-1.0)*(VX6S1))*r[4]*s[0] + \
            ((HX6S2)-t*(SX6S2)+(p-1.0)*(VX6S2))*r[4]*s[1] + \
            ((HX7X7)-t*(SX7X7)+(p-1.0)*(VX7X7))*r[5]*r[5] + \
            ((HX7S1)-t*(SX7S1)+(p-1.0)*(VX7S1))*r[5]*s[0] + \
            ((HX7S2)-t*(SX7S2)+(p-1.0)*(VX7S2))*r[5]*s[1] + \
            ((HS1S1)-t*(SS1S1)+(p-1.0)*(VS1S1))*s[0]*s[0] + \
            ((HS1S2)-t*(SS1S2)+(p-1.0)*(VS1S2))*s[0]*s[1] + \
            ((HS2S2)-t*(SS2S2)+(p-1.0)*(VS2S2))*s[1]*s[1] + \
            ((HX2X2X7)+(p-1.0)*(VX2X2X7))*r[0]*r[0]*r[5] + \
            ((HX2X2S2)+(p-1.0)*(VX2X2S2))*r[0]*r[0]*s[1] + \
            ((HX2X3X7)+(p-1.0)*(VX2X3X7))*r[0]*r[1]*r[5] + \
            ((HX2X3S2)+(p-1.0)*(VX2X3S2))*r[0]*r[1]*s[1] + \
            ((HX2X4X7)+(p-1.0)*(VX2X4X7))*r[0]*r[2]*r[5] + \
            ((HX2X4S2)+(p-1.0)*(VX2X4S2))*r[0]*r[2]*s[1] + \
            ((HX2X5X7)+(p-1.0)*(VX2X5X7))*r[0]*r[3]*r[5] + \
            ((HX2X5S2)+(p-1.0)*(VX2X5S2))*r[0]*r[3]*s[1] + \
            ((HX2X6X7)+(p-1.0)*(VX2X6X7))*r[0]*r[4]*r[5] + \
            ((HX2X6S2)+(p-1.0)*(VX2X6S2))*r[0]*r[4]*s[1] + \
            ((HX2X7X7)-t*(SX2X7X7)+(p-1.0)*(VX2X7X7))*r[0]*r[5]*r[5] + \
            ((HX2X7S2)+(p-1.0)*(VX2X7S2))*r[0]*r[5]*s[1] + \
            ((HX2S2S2)+(p-1.0)*(VX2S2S2))*r[0]*s[1]*s[1] + \
            ((HX3X3X7)+(p-1.0)*(VX3X3X7))*r[1]*r[1]*r[5] + \
            ((HX3X3S2)+(p-1.0)*(VX3X3S2))*r[1]*r[1]*s[1] + \
            ((HX3X4X7)+(p-1.0)*(VX3X4X7))*r[1]*r[2]*r[5] + \
            ((HX3X4S2)+(p-1.0)*(VX3X4S2))*r[1]*r[2]*s[1] + \
            ((HX3X5X7)+(p-1.0)*(VX3X5X7))*r[1]*r[3]*r[5] + \
            ((HX3X5S2)+(p-1.0)*(VX3X5S2))*r[1]*r[3]*s[1] + \
            ((HX3X6X7)+(p-1.0)*(VX3X6X7))*r[1]*r[4]*r[5] + \
            ((HX3X6S2)+(p-1.0)*(VX3X6S2))*r[1]*r[4]*s[1] + \
            ((HX3X7X7)-t*(SX3X7X7)+(p-1.0)*(VX3X7X7))*r[1]*r[5]*r[5] + \
            ((HX3X7S2)-t*(SX3X7S2)+(p-1.0)*(VX3X7S2))*r[1]*r[5]*s[1] + \
            ((HX3S2S2)+(p-1.0)*(VX3S2S2))*r[1]*s[1]*s[1] + \
            ((HX4X4X7)+(p-1.0)*(VX4X4X7))*r[2]*r[2]*r[5] + \
            ((HX4X4S2)+(p-1.0)*(VX4X4S2))*r[2]*r[2]*s[1] + \
            ((HX4X5X7)+(p-1.0)*(VX4X5X7))*r[2]*r[3]*r[5] + \
            ((HX4X5S2)+(p-1.0)*(VX4X5S2))*r[2]*r[3]*s[1] + \
            ((HX4X6X7)+(p-1.0)*(VX4X6X7))*r[2]*r[4]*r[5] + \
            ((HX4X6S2)+(p-1.0)*(VX4X6S2))*r[2]*r[4]*s[1] + \
            ((HX4X7X7)-t*(SX4X7X7)+(p-1.0)*(VX4X7X7))*r[2]*r[5]*r[5] + \
            ((HX4X7S2)-t*(SX4X7S2)+(p-1.0)*(VX4X7S2))*r[2]*r[5]*s[1] + \
            ((HX4S2S2)+(p-1.0)*(VX4S2S2))*r[2]*s[1]*s[1] + \
            ((HX5X5X7)+(p-1.0)*(VX5X5X7))*r[3]*r[3]*r[5] + \
            ((HX5X5S2)+(p-1.0)*(VX5X5S2))*r[3]*r[3]*s[1] + \
            ((HX5X6X7)+(p-1.0)*(VX5X6X7))*r[3]*r[4]*r[5] + \
            ((HX5X6S2)+(p-1.0)*(VX5X6S2))*r[3]*r[4]*s[1] + \
            ((HX5X7X7)-t*(SX5X7X7)+(p-1.0)*(VX5X7X7))*r[3]*r[5]*r[5] + \
            ((HX5X7S2)-t*(SX5X7S2)+(p-1.0)*(VX5X7S2))*r[3]*r[5]*s[1] + \
            ((HX5S2S2)+(p-1.0)*(VX5S2S2))*r[3]*s[1]*s[1] + \
            ((HX6X6X7)+(p-1.0)*(VX6X6X7))*r[4]*r[4]*r[5] + \
            ((HX6X6S2)+(p-1.0)*(VX6X6S2))*r[4]*r[4]*s[1] + \
            ((HX6X7X7)-t*(SX6X7X7)+(p-1.0)*(VX6X7X7))*r[4]*r[5]*r[5] + \
            ((HX6X7S2)-t*(SX6X7S2)+(p-1.0)*(VX6X7S2))*r[4]*r[5]*s[1] + \
            ((HX6S2S2)+(p-1.0)*(VX6S2S2))*r[4]*s[1]*s[1] + \
            ((HX7X7X7)-t*(SX7X7X7)+(p-1.0)*(VX7X7X7))*r[5]*r[5]*r[5] + \
            ((HX7X7S2)-t*(SX7X7S2)+(p-1.0)*(VX7X7S2))*r[5]*r[5]*s[1] + \
            ((HX7S2S2)+(p-1.0)*(VX7S2S2))*r[5]*s[1]*s[1]

/*----------------------------------------------------------------------------*/

#define DGDR0 R*t*(log(xfe2m1) - log(xmg2m1)) + \
              ((HX2)-t*(SX2)+(p-1.0)*(VX2)) + \
              ((HX2X2)-t*(SX2X2)+(p-1.0)*(VX2X2))*r[0]*2.0 + \
              ((HX2X3)-t*(SX2X3)+(p-1.0)*(VX2X3))*r[1] + \
              ((HX2X4)-t*(SX2X4)+(p-1.0)*(VX2X4))*r[2] + \
              ((HX2X5)-t*(SX2X5)+(p-1.0)*(VX2X5))*r[3] + \
              ((HX2X6)-t*(SX2X6)+(p-1.0)*(VX2X6))*r[4] + \
              ((HX2X7)-t*(SX2X7)+(p-1.0)*(VX2X7))*r[5] + \
              ((HX2S1)-t*(SX2S1)+(p-1.0)*(VX2S1))*s[0] + \
              ((HX2S2)-t*(SX2S2)+(p-1.0)*(VX2S2))*s[1] + \
              ((HX2X2X7)+(p-1.0)*(VX2X2X7))*r[0]*r[5]*2.0 + \
              ((HX2X2S2)+(p-1.0)*(VX2X2S2))*r[0]*s[1]*2.0 + \
              ((HX2X3X7)+(p-1.0)*(VX2X3X7))*r[1]*r[5] + \
              ((HX2X3S2)+(p-1.0)*(VX2X3S2))*r[1]*s[1] + \
              ((HX2X4X7)+(p-1.0)*(VX2X4X7))*r[2]*r[5] + \
              ((HX2X4S2)+(p-1.0)*(VX2X4S2))*r[2]*s[1] + \
              ((HX2X5X7)+(p-1.0)*(VX2X5X7))*r[3]*r[5] + \
              ((HX2X5S2)+(p-1.0)*(VX2X5S2))*r[3]*s[1] + \
              ((HX2X6X7)+(p-1.0)*(VX2X6X7))*r[4]*r[5] + \
              ((HX2X6S2)+(p-1.0)*(VX2X6S2))*r[4]*s[1] + \
              ((HX2X7X7)-t*(SX2X7X7)+(p-1.0)*(VX2X7X7))*r[5]*r[5] + \
              ((HX2X7S2)+(p-1.0)*(VX2X7S2))*r[5]*s[1] + \
              ((HX2S2S2)+(p-1.0)*(VX2S2S2))*s[1]*s[1]
#define DGDR1 R*t*(0.5*log(xti4m1) - 0.5*log(xmg2m1) + 0.5*log(1.0-xti4m1) \
                - log(xmg2m1+xfe2m1-xti4m1) + 0.5*log(xmg2m1+xfe2m1) \
                + 0.5*log(1.0-xmg2m1-xfe2m1-xna1m2) - log(1.0-xsi4tet) \
                - 0.5*log(1.0-xmg2m1-xfe2m1) + log(xal3tet)) + \
              ((HX3)-t*(SX3)+(p-1.0)*(VX3)) + \
              ((HX2X3)-t*(SX2X3)+(p-1.0)*(VX2X3))*r[0] + \
              ((HX3X3)-t*(SX3X3)+(p-1.0)*(VX3X3))*r[1]*2.0 + \
              ((HX3X4)-t*(SX3X4)+(p-1.0)*(VX3X4))*r[2] + \
              ((HX3X5)-t*(SX3X5)+(p-1.0)*(VX3X5))*r[3] + \
              ((HX3X6)-t*(SX3X6)+(p-1.0)*(VX3X6))*r[4] + \
              ((HX3X7)-t*(SX3X7)+(p-1.0)*(VX3X7))*r[5] + \
              ((HX3S1)-t*(SX3S1)+(p-1.0)*(VX3S1))*s[0] + \
              ((HX3S2)-t*(SX3S2)+(p-1.0)*(VX3S2))*s[1] + \
              ((HX2X3X7)+(p-1.0)*(VX2X3X7))*r[0]*r[5] + \
              ((HX2X3S2)+(p-1.0)*(VX2X3S2))*r[0]*s[1] + \
              ((HX3X3X7)+(p-1.0)*(VX3X3X7))*r[1]*r[5]*2.0 + \
              ((HX3X3S2)+(p-1.0)*(VX3X3S2))*r[1]*s[1]*2.0 + \
              ((HX3X4X7)+(p-1.0)*(VX3X4X7))*r[2]*r[5] + \
              ((HX3X4S2)+(p-1.0)*(VX3X4S2))*r[2]*s[1] + \
              ((HX3X5X7)+(p-1.0)*(VX3X5X7))*r[3]*r[5] + \
              ((HX3X5S2)+(p-1.0)*(VX3X5S2))*r[3]*s[1] + \
              ((HX3X6X7)+(p-1.0)*(VX3X6X7))*r[4]*r[5] + \
              ((HX3X6S2)+(p-1.0)*(VX3X6S2))*r[4]*s[1] + \
              ((HX3X7X7)-t*(SX3X7X7)+(p-1.0)*(VX3X7X7))*r[5]*r[5] + \
              ((HX3X7S2)-t*(SX3X7S2)+(p-1.0)*(VX3X7S2))*r[5]*s[1] + \
              ((HX3S2S2)+(p-1.0)*(VX3S2S2))*s[1]*s[1]
#define DGDR2 R*t*(0.5*log(xti4m1) - 0.5*log(xmg2m1) + 0.5*log(1.0-xti4m1) \
                - log(xmg2m1+xfe2m1-xti4m1) + 0.5*log(xmg2m1+xfe2m1) \
                + 0.5*log(1.0-xmg2m1-xfe2m1-xna1m2) - log(1.0-xsi4tet) \
                - 0.5*log(1.0-xmg2m1-xfe2m1) + log(xfe3tet)) + \
              ((HX4)-t*(SX4)+(p-1.0)*(VX4)) + \
              ((HX2X4)-t*(SX2X4)+(p-1.0)*(VX2X4))*r[0] + \
              ((HX3X4)-t*(SX3X4)+(p-1.0)*(VX3X4))*r[1] + \
              ((HX4X4)-t*(SX4X4)+(p-1.0)*(VX4X4))*r[2]*2.0 + \
              ((HX4X5)-t*(SX4X5)+(p-1.0)*(VX4X5))*r[3] + \
              ((HX4X6)-t*(SX4X6)+(p-1.0)*(VX4X6))*r[4] + \
              ((HX4X7)-t*(SX4X7)+(p-1.0)*(VX4X7))*r[5] + \
              ((HX4S1)-t*(SX4S1)+(p-1.0)*(VX4S1))*s[0] + \
              ((HX4S2)-t*(SX4S2)+(p-1.0)*(VX4S2))*s[1] + \
              ((HX2X4X7)+(p-1.0)*(VX2X4X7))*r[0]*r[5] + \
              ((HX2X4S2)+(p-1.0)*(VX2X4S2))*r[0]*s[1] + \
              ((HX3X4X7)+(p-1.0)*(VX3X4X7))*r[1]*r[5] + \
              ((HX3X4S2)+(p-1.0)*(VX3X4S2))*r[1]*s[1] + \
              ((HX4X4X7)+(p-1.0)*(VX4X4X7))*r[2]*r[5]*2.0 + \
              ((HX4X4S2)+(p-1.0)*(VX4X4S2))*r[2]*s[1]*2.0 + \
              ((HX4X5X7)+(p-1.0)*(VX4X5X7))*r[3]*r[5] + \
              ((HX4X5S2)+(p-1.0)*(VX4X5S2))*r[3]*s[1] + \
              ((HX4X6X7)+(p-1.0)*(VX4X6X7))*r[4]*r[5] + \
              ((HX4X6S2)+(p-1.0)*(VX4X6S2))*r[4]*s[1] + \
              ((HX4X7X7)-t*(SX4X7X7)+(p-1.0)*(VX4X7X7))*r[5]*r[5] + \
              ((HX4X7S2)-t*(SX4X7S2)+(p-1.0)*(VX4X7S2))*r[5]*s[1] + \
              ((HX4S2S2)+(p-1.0)*(VX4S2S2))*s[1]*s[1]
#define DGDR3 R*t*(0.5*log(xal3m1) + 0.5*log(xfe3m1) - log(xmg2m1) \
                - log(xmg2m1+xfe2m1-xti4m1) + log(xmg2m1+xfe2m1) \
                - log(1.0-xsi4tet) + 0.5*log(xal3tet) + 0.5*log(xfe3tet) \
                + log(1.0-xmg2m1-xfe2m1-xna1m2) - log(1.0-xmg2m1-xfe2m1)) + \
              ((HX5)-t*(SX5)+(p-1.0)*(VX5)) + \
              ((HX2X5)-t*(SX2X5)+(p-1.0)*(VX2X5))*r[0] + \
              ((HX3X5)-t*(SX3X5)+(p-1.0)*(VX3X5))*r[1] + \
              ((HX4X5)-t*(SX4X5)+(p-1.0)*(VX4X5))*r[2] + \
              ((HX5X5)-t*(SX5X5)+(p-1.0)*(VX5X5))*r[3]*2.0 + \
              ((HX5X6)-t*(SX5X6)+(p-1.0)*(VX5X6))*r[4] + \
              ((HX5X7)-t*(SX5X7)+(p-1.0)*(VX5X7))*r[5] + \
              ((HX5S1)-t*(SX5S1)+(p-1.0)*(VX5S1))*s[0] + \
              ((HX5S2)-t*(SX5S2)+(p-1.0)*(VX5S2))*s[1] + \
              ((HX2X5X7)+(p-1.0)*(VX2X5X7))*r[0]*r[5] + \
              ((HX2X5S2)+(p-1.0)*(VX2X5S2))*r[0]*s[1] + \
              ((HX3X5X7)+(p-1.0)*(VX3X5X7))*r[1]*r[5] + \
              ((HX3X5S2)+(p-1.0)*(VX3X5S2))*r[1]*s[1] + \
              ((HX4X5X7)+(p-1.0)*(VX4X5X7))*r[2]*r[5] + \
              ((HX4X5S2)+(p-1.0)*(VX4X5S2))*r[2]*s[1] + \
              ((HX5X5X7)+(p-1.0)*(VX5X5X7))*r[3]*r[5]*2.0 + \
              ((HX5X5S2)+(p-1.0)*(VX5X5S2))*r[3]*s[1]*2.0 + \
              ((HX5X6X7)+(p-1.0)*(VX5X6X7))*r[4]*r[5] + \
              ((HX5X6S2)+(p-1.0)*(VX5X6S2))*r[4]*s[1] + \
              ((HX5X7X7)-t*(SX5X7X7)+(p-1.0)*(VX5X7X7))*r[5]*r[5] + \
              ((HX5X7S2)-t*(SX5X7S2)+(p-1.0)*(VX5X7S2))*r[5]*s[1] + \
              ((HX5S2S2)+(p-1.0)*(VX5S2S2))*s[1]*s[1]
#define DGDR4 R*t*(0.25*log(xal3m1) + 0.25*log(xfe3m1) - 0.5*log(xmg2m1) \
                - 0.5*log(xmg2m1+xfe2m1-xti4m1) + 0.5*log(xmg2m1+xfe2m1) \
                + 0.5*log(1.0-xsi4tet) - 0.25*log(xal3tet) \
                - 0.25*log(xfe3tet) - log(xca2m2) + log(xna1m2) \
                - 0.5*log(1.0-xmg2m1-xfe2m1-xna1m2) \
                - 0.5*log(1.0-xmg2m1-xfe2m1) + log(1.0-xna1m2)) + \
              ((HX6)-t*(SX6)+(p-1.0)*(VX6)) + \
              ((HX2X6)-t*(SX2X6)+(p-1.0)*(VX2X6))*r[0] + \
              ((HX3X6)-t*(SX3X6)+(p-1.0)*(VX3X6))*r[1] + \
              ((HX4X6)-t*(SX4X6)+(p-1.0)*(VX4X6))*r[2] + \
              ((HX5X6)-t*(SX5X6)+(p-1.0)*(VX5X6))*r[3] + \
              ((HX6X6)-t*(SX6X6)+(p-1.0)*(VX6X6))*r[4]*2.0 + \
              ((HX6X7)-t*(SX6X7)+(p-1.0)*(VX6X7))*r[5] + \
              ((HX6S1)-t*(SX6S1)+(p-1.0)*(VX6S1))*s[0] + \
              ((HX6S2)-t*(SX6S2)+(p-1.0)*(VX6S2))*s[1] + \
              ((HX2X6X7)+(p-1.0)*(VX2X6X7))*r[0]*r[5] + \
              ((HX2X6S2)+(p-1.0)*(VX2X6S2))*r[0]*s[1] + \
              ((HX3X6X7)+(p-1.0)*(VX3X6X7))*r[1]*r[5] + \
              ((HX3X6S2)+(p-1.0)*(VX3X6S2))*r[1]*s[1] + \
              ((HX4X6X7)+(p-1.0)*(VX4X6X7))*r[2]*r[5] + \
              ((HX4X6S2)+(p-1.0)*(VX4X6S2))*r[2]*s[1] + \
              ((HX5X6X7)+(p-1.0)*(VX5X6X7))*r[3]*r[5] + \
              ((HX5X6S2)+(p-1.0)*(VX5X6S2))*r[3]*s[1] + \
              ((HX6X6X7)+(p-1.0)*(VX6X6X7))*r[4]*r[5]*2.0 + \
              ((HX6X6S2)+(p-1.0)*(VX6X6S2))*r[4]*s[1]*2.0 + \
              ((HX6X7X7)-t*(SX6X7X7)+(p-1.0)*(VX6X7X7))*r[5]*r[5] + \
              ((HX6X7S2)-t*(SX6X7S2)+(p-1.0)*(VX6X7S2))*r[5]*s[1] + \
              ((HX6S2S2)+(p-1.0)*(VX6S2S2))*s[1]*s[1]
#define DGDR5 0.5*R*t*(log(xmg2m1) - 2.0*log(xca2m2) \
                + log(xmg2m2) + log(xfe2m2/xfe2m1)) + \
              ((HX7)-t*(SX7)+(p-1.0)*(VX7)) + \
              ((HX2X7)-t*(SX2X7)+(p-1.0)*(VX2X7))*r[0] + \
              ((HX3X7)-t*(SX3X7)+(p-1.0)*(VX3X7))*r[1] + \
              ((HX4X7)-t*(SX4X7)+(p-1.0)*(VX4X7))*r[2] + \
              ((HX5X7)-t*(SX5X7)+(p-1.0)*(VX5X7))*r[3] + \
              ((HX6X7)-t*(SX6X7)+(p-1.0)*(VX6X7))*r[4] + \
              ((HX7X7)-t*(SX7X7)+(p-1.0)*(VX7X7))*r[5]*2.0 + \
              ((HX7S1)-t*(SX7S1)+(p-1.0)*(VX7S1))*s[0] + \
              ((HX7S2)-t*(SX7S2)+(p-1.0)*(VX7S2))*s[1] + \
              ((HX2X2X7)+(p-1.0)*(VX2X2X7))*r[0]*r[0] + \
              ((HX2X3X7)+(p-1.0)*(VX2X3X7))*r[0]*r[1] + \
              ((HX2X4X7)+(p-1.0)*(VX2X4X7))*r[0]*r[2] + \
              ((HX2X5X7)+(p-1.0)*(VX2X5X7))*r[0]*r[3] + \
              ((HX2X6X7)+(p-1.0)*(VX2X6X7))*r[0]*r[4] + \
              ((HX2X7X7)-t*(SX2X7X7)+(p-1.0)*(VX2X7X7))*r[0]*r[5]*2.0 + \
              ((HX2X7S2)+(p-1.0)*(VX2X7S2))*r[0]*s[1] + \
              ((HX3X3X7)+(p-1.0)*(VX3X3X7))*r[1]*r[1] + \
              ((HX3X4X7)+(p-1.0)*(VX3X4X7))*r[1]*r[2] + \
              ((HX3X5X7)+(p-1.0)*(VX3X5X7))*r[1]*r[3] + \
              ((HX3X6X7)+(p-1.0)*(VX3X6X7))*r[1]*r[4] + \
              ((HX3X7X7)-t*(SX3X7X7)+(p-1.0)*(VX3X7X7))*r[1]*r[5]*2.0 + \
              ((HX3X7S2)-t*(SX3X7S2)+(p-1.0)*(VX3X7S2))*r[1]*s[1] + \
              ((HX4X4X7)+(p-1.0)*(VX4X4X7))*r[2]*r[2] + \
              ((HX4X5X7)+(p-1.0)*(VX4X5X7))*r[2]*r[3] + \
              ((HX4X6X7)+(p-1.0)*(VX4X6X7))*r[2]*r[4] + \
              ((HX4X7X7)-t*(SX4X7X7)+(p-1.0)*(VX4X7X7))*r[2]*r[5]*2.0 + \
              ((HX4X7S2)-t*(SX4X7S2)+(p-1.0)*(VX4X7S2))*r[2]*s[1] + \
              ((HX5X5X7)+(p-1.0)*(VX5X5X7))*r[3]*r[3] + \
              ((HX5X6X7)+(p-1.0)*(VX5X6X7))*r[3]*r[4] + \
              ((HX5X7X7)-t*(SX5X7X7)+(p-1.0)*(VX5X7X7))*r[3]*r[5]*2.0 + \
              ((HX5X7S2)-t*(SX5X7S2)+(p-1.0)*(VX5X7S2))*r[3]*s[1] + \
              ((HX6X6X7)+(p-1.0)*(VX6X6X7))*r[4]*r[4] + \
              ((HX6X7X7)-t*(SX6X7X7)+(p-1.0)*(VX6X7X7))*r[4]*r[5]*2.0 + \
              ((HX6X7S2)-t*(SX6X7S2)+(p-1.0)*(VX6X7S2))*r[4]*s[1] + \
              ((HX7X7X7)-t*(SX7X7X7)+(p-1.0)*(VX7X7X7))*r[5]*r[5]*3.0 + \
              ((HX7X7S2)-t*(SX7X7S2)+(p-1.0)*(VX7X7S2))*r[5]*s[1]*2.0 + \
              ((HX7S2S2)+(p-1.0)*(VX7S2S2))*s[1]*s[1]
#define DGDS0 0.5*R*t*(log(xfe3m1)-log(xal3m1)+log(xal3tet)-log(xfe3tet)) + \
              ((HS1)-t*(SS1)+(p-1.0)*(VS1)) + \
              ((HX2S1)-t*(SX2S1)+(p-1.0)*(VX2S1))*r[0] + \
              ((HX3S1)-t*(SX3S1)+(p-1.0)*(VX3S1))*r[1] + \
              ((HX4S1)-t*(SX4S1)+(p-1.0)*(VX4S1))*r[2] + \
              ((HX5S1)-t*(SX5S1)+(p-1.0)*(VX5S1))*r[3] + \
              ((HX6S1)-t*(SX6S1)+(p-1.0)*(VX6S1))*r[4] + \
              ((HX7S1)-t*(SX7S1)+(p-1.0)*(VX7S1))*r[5] + \
              ((HS1S1)-t*(SS1S1)+(p-1.0)*(VS1S1))*s[0]*2.0 + \
              ((HS1S2)-t*(SS1S2)+(p-1.0)*(VS1S2))*s[1]
#define DGDS1 0.5*R*t*(log(xmg2m1)-log(xfe2m1)+log(xfe2m2)-log(xmg2m2)) + \
              ((HS2)-t*(SS2)+(p-1.0)*(VS2)) + \
              ((HX2S2)-t*(SX2S2)+(p-1.0)*(VX2S2))*r[0] + \
              ((HX3S2)-t*(SX3S2)+(p-1.0)*(VX3S2))*r[1] + \
              ((HX4S2)-t*(SX4S2)+(p-1.0)*(VX4S2))*r[2] + \
              ((HX5S2)-t*(SX5S2)+(p-1.0)*(VX5S2))*r[3] + \
              ((HX6S2)-t*(SX6S2)+(p-1.0)*(VX6S2))*r[4] + \
              ((HX7S2)-t*(SX7S2)+(p-1.0)*(VX7S2))*r[5] + \
              ((HS1S2)-t*(SS1S2)+(p-1.0)*(VS1S2))*s[0] + \
              ((HS2S2)-t*(SS2S2)+(p-1.0)*(VS2S2))*s[1]*2.0 + \
              ((HX2X2S2)+(p-1.0)*(VX2X2S2))*r[0]*r[0] + \
              ((HX2X3S2)+(p-1.0)*(VX2X3S2))*r[0]*r[1] + \
              ((HX2X4S2)+(p-1.0)*(VX2X4S2))*r[0]*r[2] + \
              ((HX2X5S2)+(p-1.0)*(VX2X5S2))*r[0]*r[3] + \
              ((HX2X6S2)+(p-1.0)*(VX2X6S2))*r[0]*r[4] + \
              ((HX2X7S2)+(p-1.0)*(VX2X7S2))*r[0]*r[5] + \
              ((HX2S2S2)+(p-1.0)*(VX2S2S2))*r[0]*s[1]*2.0 + \
              ((HX3X3S2)+(p-1.0)*(VX3X3S2))*r[1]*r[1] + \
              ((HX3X4S2)+(p-1.0)*(VX3X4S2))*r[1]*r[2] + \
              ((HX3X5S2)+(p-1.0)*(VX3X5S2))*r[1]*r[3] + \
              ((HX3X6S2)+(p-1.0)*(VX3X6S2))*r[1]*r[4] + \
              ((HX3X7S2)-t*(SX3X7S2)+(p-1.0)*(VX3X7S2))*r[1]*r[5] + \
              ((HX3S2S2)+(p-1.0)*(VX3S2S2))*r[1]*s[1]*2.0 + \
              ((HX4X4S2)+(p-1.0)*(VX4X4S2))*r[2]*r[2] + \
              ((HX4X5S2)+(p-1.0)*(VX4X5S2))*r[2]*r[3] + \
              ((HX4X6S2)+(p-1.0)*(VX4X6S2))*r[2]*r[4] + \
              ((HX4X7S2)-t*(SX4X7S2)+(p-1.0)*(VX4X7S2))*r[2]*r[5] + \
              ((HX4S2S2)+(p-1.0)*(VX4S2S2))*r[2]*s[1]*2.0 + \
              ((HX5X5S2)+(p-1.0)*(VX5X5S2))*r[3]*r[3] + \
              ((HX5X6S2)+(p-1.0)*(VX5X6S2))*r[3]*r[4] + \
              ((HX5X7S2)-t*(SX5X7S2)+(p-1.0)*(VX5X7S2))*r[3]*r[5] + \
              ((HX5S2S2)+(p-1.0)*(VX5S2S2))*r[3]*s[1]*2.0 + \
              ((HX6X6S2)+(p-1.0)*(VX6X6S2))*r[4]*r[4] + \
              ((HX6X7S2)-t*(SX6X7S2)+(p-1.0)*(VX6X7S2))*r[4]*r[5] + \
              ((HX6S2S2)+(p-1.0)*(VX6S2S2))*r[4]*s[1]*2.0 + \
              ((HX7X7S2)-t*(SX7X7S2)+(p-1.0)*(VX7X7S2))*r[5]*r[5] + \
              ((HX7S2S2)+(p-1.0)*(VX7S2S2))*r[5]*s[1]*2.0
#define DGDT  -(S)
#define DGDP  (V)

/*----------------------------------------------------------------------------*/
