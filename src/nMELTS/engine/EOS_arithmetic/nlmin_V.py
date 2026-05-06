"""nlmin_V.py: bounded 1D minimizer for the solid volume spinodal search."""

from __future__ import annotations


def nlmin_V(x_init, xlow, xupp, objective, state=None, maxeval=1000):
    """Minimize a scalar objective on a closed interval."""

    if xlow > xupp:
        return x_init, float("inf"), -2

    def f(x):
        try:
            return objective(x)
        except TypeError:
            if state is None:
                raise
            return objective(x, state)

    x_init = min(max(x_init, xlow), xupp)
    if xlow == xupp:
        return xlow, f(xlow), 1

    nscan = max(5, min(33, maxeval // 4 if maxeval > 0 else 33))
    step = (xupp - xlow) / float(nscan - 1)
    samples = [xlow + i * step for i in range(nscan)]
    if x_init not in samples:
        samples.append(x_init)
    samples = sorted(set(samples))

    best_x = samples[0]
    best_f = f(best_x)
    eval_count = 1
    for x in samples[1:]:
        if eval_count >= maxeval:
            break
        fx = f(x)
        eval_count += 1
        if fx < best_f:
            best_x = x
            best_f = fx

    idx = samples.index(best_x)
    left = samples[max(0, idx - 1)]
    right = samples[min(len(samples) - 1, idx + 1)]
    if left == right:
        return best_x, best_f, 1

    gr = 0.6180339887498949
    c = right - gr * (right - left)
    d = left + gr * (right - left)
    fc = f(c)
    fd = f(d)
    eval_count += 2

    while eval_count < maxeval and abs(right - left) > 1.0e-8 * max(1.0, abs(best_x)):
        if fc < fd:
            right = d
            d = c
            fd = fc
            c = right - gr * (right - left)
            fc = f(c)
        else:
            left = c
            c = d
            fc = fd
            d = left + gr * (right - left)
            fd = f(d)
        eval_count += 1
        if fc < best_f:
            best_x, best_f = c, fc
        if fd < best_f:
            best_x, best_f = d, fd

    if best_f == float("inf"):
        return best_x, best_f, 0
    return best_x, best_f, 1
