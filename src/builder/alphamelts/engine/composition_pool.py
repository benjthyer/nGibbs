"""
Pre-mixing helper for ratio-controlled ensemble MELTS generation.

Builds a single composition pool from several compositional groups (e.g. mafic,
ultramafic, full GEOROC) so that a single call to a randommelter function can
draw from it uniformly and reproduce a prescribed group ratio, instead of
calling the melter once per group with hand-tuned iteration counts.
"""

import numpy as np


def build_ratio_pool(groups, ratios, pool_size):
    """
    Build a single composition pool where each group's rows appear in
    proportion to `ratios`. Sampling uniformly from the result (as the
    randommelter functions already do via np.random.randint each cycle)
    reproduces the ratio in expectation over many draws, so no per-iteration
    group bookkeeping is needed in the melter itself.

    Parameters
    ----------
    groups : list of np.ndarray
        Each array is GEOROC-format: column 0 is the original row index,
        remaining columns are oxides (matching a shared col_dict).
    ratios : list of int/float
        Relative weight per group, e.g. [5, 1, 3]. Need not sum to anything
        in particular; only relative magnitude matters.
    pool_size : int
        Total rows in the returned pool. Only needs enough rows to give the
        melter's per-cycle random draws reasonable diversity - sampling from
        the pool is with replacement, so this does not need to match the
        eventual target row count of the output dataset.

    Returns
    -------
    np.ndarray
        Shuffled pool of shape (pool_size, groups[0].shape[1]).
    """
    if len(groups) != len(ratios):
        raise ValueError(f"groups and ratios must have the same length, got {len(groups)} and {len(ratios)}")

    weights = np.array(ratios, dtype=float)
    weights /= weights.sum()
    counts = np.round(weights * pool_size).astype(int)
    counts[-1] = pool_size - counts[:-1].sum()  # fix rounding drift so total is exact

    rows = []
    for group, n in zip(groups, counts):
        if n <= 0:
            continue
        if group.shape[0] == 0:
            raise ValueError("Cannot draw rows from an empty compositional group.")
        rows.append(group[np.random.randint(group.shape[0], size=n)])

    pool = np.vstack(rows)
    np.random.shuffle(pool)
    return pool
