"""
cage.py: Root bracketing helper

Translation of cage.f from Numerical Recipes (Press et al., 1992).
Finds a bracketing interval [a, b] such that f(a)*f(b) < 0 by golden
section expansion from an initial guess.

Key parameters:
- gold = 1.618 (golden ratio)
- itmax = 100 (maximum iterations)

Error codes (ires):
- 1: Input range valid, ready to search
- 0: Bracketing failed (itmax reached)
- -1: Lower bound violation
- -2: Upper bound violation
"""


def cage(func, a_init, b_init, clow, cupp, state=None):
    """
    Find bracketing interval for a root of func between bounds [clow, cupp].

    Uses golden section expansion to find bracket [a, b] where f(a)*f(b) < 0.

    Parameters
    ----------
    func : callable
        Function to evaluate; func(x) returns scalar
    a_init : float
        Initial left bracket endpoint
    b_init : float
        Initial right bracket endpoint
    clow : float
        Absolute lower bound for bracket
    cupp : float
        Absolute upper bound for bracket
    state : object, optional
        State object passed to func (HeFESToState); unused if None

    Returns
    -------
    a : float
        Left bracket (or unchanged if error)
    b : float
        Right bracket (or unchanged if error)
    ires : int
        Status:
        - 1: Input valid, ready to expand
        - 0: Failed to bracket (itmax reached)
        - -1: Lower bound violation during expansion
        - -2: Upper bound violation during expansion

    Notes
    -----
    - Assumes a_init < b_init initially
    - If f(a)*f(b) < 0 already, returns immediately with ires=1
    - Golden ratio expansion factor: 1.618
    """
    GOLD = 1.618
    ITMAX = 100

    ires = 0

    # Validate initial range
    if a_init >= b_init:
        return a_init, b_init, ires

    if a_init < clow:
        return a_init, b_init, ires

    if b_init > cupp:
        return a_init, b_init, ires

    ires = 1
    a = float(a_init)
    b = float(b_init)
    fa = func(a)
    fb = func(b)

    # Check if already bracketed
    if fa * fb < 0.0:
        return a, b, ires

    # Expand bracket via golden section
    for i in range(ITMAX):
        if fa * fb < 0.0:
            return a, b, ires

        if abs(fa) < abs(fb):
            # Expand from a
            at = a + GOLD * (a - b)
            if at <= clow:
                ires = -1
                return a, b, ires
            at = max(at, clow)
            a = at
            fa = func(a)
        else:
            # Expand from b
            bt = b + GOLD * (b - a)
            if bt >= cupp:
                ires = -2
                return a, b, ires
            bt = min(bt, cupp)
            b = bt
            fb = func(b)

    # itmax reached without bracketing
    ires = 0
    return a, b, ires
