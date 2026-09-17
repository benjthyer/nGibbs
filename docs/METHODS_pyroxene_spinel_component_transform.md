# Non-negative, closure-preserving reparameterization of pyroxene and spinel components

## Motivation

The intensive-composition network predicts each phase's composition with a softmax activation over $v+1$ raw outputs, where $v$ is the number of compositionally independent degrees of freedom. Softmax guarantees, by construction, that its outputs are strictly positive and sum to one, so the network can never emit a thermodynamically illegal composition regardless of the raw (unconstrained) output it produces. That guarantee is only useful, however, if the *training targets* can themselves be expressed, exactly and invertibly, in a basis with the same non-negative, unit-sum structure. The MELTS pyroxene and spinel solid-solution models do not have this property natively: each expresses composition as mole fractions of a fixed set of components (seven for pyroxene, five for spinel) that are individually unbounded below zero — only certain linear combinations of them, corresponding to real crystallographic site occupancies (Fe2+, Na, Al, Fe3+, Ti, Ca, and Mg content per formula unit, in the case of pyroxene), are constrained to physically sensible ranges. We therefore needed an exact, invertible change of basis for each solution that maps every physically legal composition to a vector that is non-negative and sums to one. Pyroxene admitted an analytic solution; spinel did not and was instead solved numerically.

## Pyroxene (analytic)

Writing the mole fractions of the seven MELTS pyroxene components (diopside, clinoenstatite, hedenbergite, aluminobuffonite, buffonite, essenite, jadeite) as $x_0,\dots,x_6$, the site-occupancy bounds reduce to inequalities on six linear combinations:

$$
r_0=x_2,\quad r_1=x_3+\tfrac12 x_6,\quad r_2=x_4-\tfrac12 x_6,\quad r_3=x_5+\tfrac12 x_6,\quad r_4=x_6,\quad r_5=x_1 .
$$

Each $r_i$ is bounded for any physically legal pyroxene, but not on $[0,1]$.

**Physical meaning of the six $r_i$.** Pyroxene has the general formula M2 M1 T2O6 (one large octahedral site M2, one small octahedral site M1, two tetrahedral T sites). The seven MELTS components occupy these sites as follows (Sack & Ghiorso, 1994):

| component | M2 | M1 | T, T |
|---|---|---|---|
| diopside ($x_0$) | Ca | Mg | Si, Si |
| clinoenstatite ($x_1$) | Mg | Mg | Si, Si |
| hedenbergite ($x_2$) | Ca | Fe2+ | Si, Si |
| alumino-buffonite ($x_3$) | Ca | ½Mg + ½Ti | Al, Si |
| buffonite ($x_4$) | Ca | ½Mg + ½Ti | Fe3+, Si |
| essenite ($x_5$) | Ca | Fe3+ | Al, Si |
| jadeite ($x_6$) | Na | Al | Si, Si |

This table (standard for the Sack & Ghiorso pyroxene model) reproduces every atoms-per-formula-unit expression in Paul's constraint list exactly: Ca $=x_0+x_2+x_3+x_4+x_5$, Na $=x_6$, Mg $=x_0+2x_1+\tfrac12(x_3+x_4)$, Al $=x_3+x_5+x_6$, Fe3+ $=x_4+x_5$, Ti $=\tfrac12(x_3+x_4)$ — which is strong corroboration that these are the intended end-member formulas, independent of anything stated in the email thread.

Three of the six $r_i$ read off this table directly, with no splitting needed: $r_0=x_2$ is the Fe2+ content (hedenbergite is the only Fe2+-bearing component here), $r_4=x_6$ is the Na content (jadeite is the only Na-bearing component, and Na occupies a single M2 site, hence the stated bound $0\le r_4\le1$), and $r_5=x_1$ is the Mg-on-M2 content, i.e. the clinoenstatite component — the Mg-Ca exchange vector distinct from the M1 Mg already carried by diopside.

The other three, $r_1=x_3+\tfrac12x_6$, $r_2=x_4-\tfrac12x_6$, $r_3=x_5+\tfrac12x_6$, are not single-component pass-throughs; they are the unique solution of a $3\times3$ linear system requiring their pairwise sums to equal the three remaining named quantities: $r_1+r_3=x_3+x_5+x_6=\mathrm{Al}$, $r_2+r_3=x_4+x_5=\mathrm{Fe}^{3+}$, and $r_1+r_2=x_3+x_4=2\,\mathrm{Ti}$. Solving that system for $r_1,r_2,r_3$ reproduces Paul's formulas exactly, which is presumably how he arrived at them, though the emails do not show this step. Diopside ($x_0$) is not assigned its own $r_i$; it is the reference component recovered from closure ($x_0=1-\sum_{i\ge1}x_i$), which is why it has the most complex entry in the $g_2x$ back-transform.

One item we could not resolve: Paul's stated bound on $r_0$ is $0\le r_0\le2$, but Fe2+ occupies only the single M1 site of hedenbergite in the table above, which on its own would cap Fe2+ at 1 rather than 2. Neither the email thread nor the code explains the factor of two, and we did not want to guess at a specific crystallographic justification for it.

**The general pattern behind the six $r_i$.** Seven components minus one closure constraint leaves six independent directions, and they split into three groups. Three elements are carried by exactly one component each and so read off directly, with no combination required: Fe2+ (hedenbergite, $r_0$), Na (jadeite, $r_4$), and Mg on the M2 site specifically (clinoenstatite, $r_5$). Three more elements are each split across *two* components, so no single fraction can stand in for them: Fe3+ (buffonite + essenite), Al (alumino-buffonite + essenite + jadeite), and Ti (alumino-buffonite + buffonite). These require solving the $3\times3$ system above for $r_1,r_2,r_3$. The two quantities left over, Ca and Si, are never tracked directly at all; they are pinned down by requiring each crystallographic site to be exactly full, a second, more local layer of closure nested inside the overall mass-balance constraint $\sum_i x_i=1$: on M2, $\mathrm{Ca}+\mathrm{Na}+\mathrm{Mg}_{M2}=1$, so $\mathrm{Ca}=1-r_4-r_5$; on the T sites, $\mathrm{Si}+\mathrm{Al}_T+\mathrm{Fe}^{3+}_T=2$, so $\mathrm{Si}=2-r_1-r_2-r_3+r_4/2$, which is exactly Paul's "Si constraint (redundant?)" line. Total Mg (M1+M2 combined) is the one element that is never fully recovered: only its M2 slice ($r_5$) is pulled out on its own, while its M1 slice stays buried inside the reference component $x_0$ (diopside) and inside $r_1+r_2$ (alumino-buffonite/buffonite's Ti-paired Mg), and is never isolated as pure Mg.

**What non-negativity actually rests on, and what it does not.** Two distinct things are happening here, and it is worth keeping them separate. The individual $g_i$ landing in $[0,1]$ follows from real physical necessity: a legal crystal cannot have a negative atom count, and every site holds at most one atom, so every element total is both non-negative and capped (hence the two-sided bounds above, not merely "$\ge0$"). $r_0$, $r_4$, and $r_5$ inherit this directly because they were deliberately chosen to *equal* real atom counts (Fe2+, Na, Mg-on-M2) rather than raw declared component fractions — unlike the $x_i$ themselves, which are not guaranteed non-negative in this parameterization (that non-guarantee is the entire reason this exercise is necessary). $r_1,r_2,r_3$ are individually less clean: only their pairwise sums are atom counts, and their individual ranges follow from solving the $3\times3$ system together with the auxiliary "Na+Ti" inequalities Paul lists, not from a one-line atom-count argument.

The unit-sum closure of the full seven-element vector is a different matter and does *not* follow from site positivity. The vector $(g_C,g_0,\dots,g_5)$ computed directly from the physical bounds is individually non-negative but does not sum to 1 for an arbitrary composition; only $g_C\equiv1$ is guaranteed, trivially, from the overall mass-balance closure $\sum_i x_i=1$. Exact closure required the separate row-normalization step described above, and its validity for a general (non-pure-end-member) composition is not guaranteed by the site-occupancy argument — the raw $x_i$ can themselves be negative for a legal pyroxene — so this step was validated empirically, by round-tripping the full training set, rather than proven from crystal chemistry.

Rescaling and recombining the $r_i$ gives six quantities that *are* individually bounded on $[0,1]$:

$$
g_0=\tfrac12 r_0,\quad g_1=\tfrac12(r_1+r_2),\quad g_2=\tfrac12(r_1+r_3),\quad g_3=\tfrac12(r_2+r_3),\quad g_4=\tfrac12 r_4,\quad g_5=\tfrac12 r_4+r_5 .
$$

The split of the jadeite content ($r_4$) across $g_4$ and $g_5$ in the last two terms was chosen deliberately, after two earlier attempts ($g_4=r_4,\,g_5=r_5$, then $g_4=r_4,\,g_5=r_4+r_5$) turned out to under-constrain a physically meaningful quantity. Back-substituting shows that the calcium budget can be written as $\mathrm{CaO}=1-g_4-g_5$ only once the jadeite contribution is split evenly between $g_4$ and $g_5$ as above; the two earlier formulations instead carried larger coefficients on $g_4,g_5$ in this expression (2.5 and 2, then 1.5 and 2, respectively), which could formally drive CaO negative even with $g_4,g_5\in[0,1]$. With the final split, non-negativity and closure of the full component vector (below) are enough by themselves to guarantee $\mathrm{CaO}\ge0$.

Together with the trivial identity $g_C\equiv1$ (recovered from $\sum_i x_i=1$), this gives a seven-element vector $(g_C,g_0,\dots,g_5)$ that is non-negative for any legal pyroxene but does not itself sum to one. To close it, we evaluated this exact linear map at each of the seven pure end-members and divided each end-member's resulting row by its own row sum, giving the fixed $7\times7$ matrix below, in which every row is individually non-negative and sums to unity:

| component | $g_C$ | $g_0$ | $g_1$ | $g_2$ | $g_3$ | $g_4$ | $g_5$ |
|---|---|---|---|---|---|---|---|
| diopside | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| clinoenstatite | 1/2 | 0 | 0 | 0 | 0 | 0 | 1/2 |
| hedenbergite | 2/3 | 1/3 | 0 | 0 | 0 | 0 | 0 |
| aluminobuffonite | 1/2 | 0 | 1/4 | 1/4 | 0 | 0 | 0 |
| buffonite | 1/2 | 0 | 1/4 | 0 | 1/4 | 0 | 0 |
| essenite | 1/2 | 0 | 0 | 1/4 | 1/4 | 0 | 0 |
| jadeite | 2/5 | 0 | 0 | 1/5 | 0 | 1/5 | 1/5 |

Applying this matrix to any pyroxene composition (expressed as mole fractions of the seven end-members) yields an output vector that inherits non-negativity and unit-sum, and the matrix inverse recovers the original mole fractions exactly. This was confirmed against the full training set of MELTS-computed pyroxene compositions, which round-tripped to machine precision. The same matrix is applied to both the orthopyroxene and clinopyroxene solid solutions.

## Spinel (learned)

The spinel solid solution (chromite, hercynite, magnetite, spinel [MgAl$_2$O$_4$], ulvöspinel) does not reduce to as clean a set of site-occupancy inequalities — the binding constraints instead couple several components together (e.g., the combined abundance of chromite, magnetite, and ulvöspinel must be at least as positive as hercynite is negative) — so no analogous analytic transform was pursued. Instead, we fit a masked, invertible linear transform numerically. The matrix was parameterized as lower-triangular, which guarantees invertibility for any non-zero diagonal, with chromite (whose mole fraction is already non-negative) mapped onto its own dedicated output component and excluded from the fit.

The remaining entries were optimized directly against the training data (on the order of $3\times10^6$ MELTS spinel compositions) by gradient descent (PyTorch, Adam optimizer), minimizing a two-term loss: a mean-squared term driving each transformed composition's row sum toward one, plus a leaky-ReLU penalty on negative transformed entries driving them toward zero. The learning rate was annealed by a factor of 10 every 2,500 of 10,000 steps; the diagonal was reset to 1 whenever it decayed toward zero, to preserve invertibility; and coefficients that fell below 0.05 in magnitude were periodically hard-thresholded to zero to keep the fitted matrix sparse. The resulting matrix and its analytic inverse were validated by histogramming the transformed compositions and their row sums, checking their min/max range, and comparing round-tripped (transform-then-inverse) compositions against the originals across the full training set, confirming near-exact recovery with the transformed components confined almost entirely to $[0,1]$.
