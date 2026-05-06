"""
zeroin.py: Brent's method for 1D root finding

Translation of zeroin.f, which implements Brent's algorithm for finding
a zero of a function in a bracketed interval.

Original source: Numerical Recipes, based on Brent, R.P. (1973)
"Algorithms for Minimization Without Derivatives", Prentice-Hall.

Key features:
- Requires f(ax) and f(bx) to have opposite signs
- Combines bisection, secant, and inverse quadratic interpolation
- Machine epsilon: 1.0e-12
- Convergence: 4*macheps*abs(x) + tol
"""


def zeroin(ax, bx, func, tol, state=None):
    """
    Find a zero of func in the interval [ax, bx] using Brent's method.

    Implements Brent's algorithm combining bisection, secant, and inverse
    quadratic interpolation for robust root-finding.

    Parameters
    ----------
    ax : float
        Left endpoint of interval
    bx : float
        Right endpoint of interval
    func : callable
        Function to evaluate; func(x) returns scalar
    tol : float
        Desired interval uncertainty (>= 0)
    state : object, optional
        State object passed to func (HeFESToState); unused if None

    Returns
    -------
    xroot : float
        Root of func in [ax, bx]

    Notes
    -----
    - Assumes f(ax) and f(bx) have opposite signs
    - If signs are same, prints warning and returns ax
    - Converges to abs_error <= 4*eps*abs(xroot) + tol where eps ~ 1e-12
    - ITMAX is not explicitly limited (but typically converges in ~10-20 iterations)
    """
    EPS = 1.0e-12
    ITMAX = 100

    a = float(ax)
    b = float(bx)
    fa = func(a)
    fb = func(b)

    # Check that f(ax) and f(bx) have different signs
    if fa == 0.0 or fb == 0.0:
        if fa == 0.0:
            return a
        else:
            return b

    # Check sign change: if fa and fb/|fb| <= 0, we have a sign change
    if fa * (fb / abs(fb)) <= 0.0:
        # Good, proceed with root-finding
        pass
    else:
        # No sign change
        print("zeroin: f(ax) and f(bx) do not have different signs, aborting")
        return ax

    c = a
    fc = fa
    d = b - a
    e = d

    for _ in range(ITMAX):
        if abs(fc) >= abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol1 = 2.0 * EPS * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)

        if abs(xm) <= tol1 or fb == 0.0:
            # Converged
            return b

        # Decide whether to use bisection or interpolation
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            # Try interpolation
            s = fb / fa

            if a != c:
                # Inverse quadratic interpolation
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            else:
                # Linear interpolation
                p = 2.0 * xm * s
                q = 1.0 - s

            if p <= 0.0:
                q = -q
            else:
                p = -p

            s = e
            e = d

            if (2.0 * p >= 3.0 * xm * q - abs(tol1 * q)) or (p >= abs(0.5 * s * q)):
                # Interpolation not acceptable, use bisection
                d = xm
                e = d
            else:
                d = p / q
        else:
            # Use bisection
            d = xm
            e = d

        # Update a and fa
        a = b
        fa = fb

        if abs(d) <= tol1:
            if xm <= 0.0:
                b = b - tol1
            else:
                b = b + tol1
        else:
            b = b + d

        fb = func(b)

        # Check if sign change occurred (convergence)
        if fb * (fc / abs(fc)) > 0.0:
            # Lost bracket, reset
            c = a
            fc = fa
            d = b - a
            e = d

    # Return best estimate
    return b
