"""
Mathematical utilities.
"""

import numpy as np
# Make torch optional for WSL scripts that don't need ML features
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False


def QFM_fO2(P, K):
    """
    Calculate log10 fO2 along the QFM buffer (legacy version, uses Celsius).
    
    Args:
        P: Pressure in bars
        K: Temperature in Celsius
        
    Returns:
        float: log10(fO2)
    """
    trans1 = 573 + (0.025 * P)
    if K > trans1:
        A = -25096.3
        B = 8.735
        D = 0.11
    else:
        A = -26455.3
        B = 10.344
        D = 0.092
    K += 273.15  # Celsius to Kelvin
    logfo2 = (A/K) + B + ((D * (P-1)) / K)
    return logfo2


def QFM_fO2_torch(P, K, use_torch=False):
    """
    Calculate log10 fO2 along the QFM buffer (supports both NumPy and PyTorch).
    
    Parameters:
        P: Pressure in bars (array-like)
        K: Temperature in Kelvin (array-like)
        use_torch: If True, use PyTorch tensors; otherwise, use NumPy arrays
        
    Returns:
        log10(fO2): Logarithm base 10 of oxygen fugacity
    """
    if use_torch and not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available. Install torch to use use_torch=True")
    xp = torch if use_torch else np  # shorthand for backend

    trans1 = 573 + (0.025 * P)
    lowKind = K > trans1

    output = xp.zeros_like(K)

    if xp.any(lowKind):
        A = -25096.3
        B = 8.735
        D = 0.11
        output = output.clone() if use_torch else output  # avoid modifying shared memory
        output[lowKind] = (A / K[lowKind]) + B + ((D * (P[lowKind] - 1)) / K[lowKind])

    if xp.any(~lowKind):
        A = -26455.3
        B = 10.344
        D = 0.092
        output = output.clone() if use_torch else output
        output[~lowKind] = (A / K[~lowKind]) + B + ((D * (P[~lowKind] - 1)) / K[~lowKind])

    return output


def Fe2O3_FeO_ratio(fO2, T, P, composition, use_torch=False, device='cpu'):
    """
    Calculate ln(X_Fe2O3 / X_FeO) using Equation 7 from Kress & Carmichael (1991).

    Parameters:
        fO2: oxygen fugacity (in atm or bar, same unit used in the original calibration)
        T: temperature in Kelvin
        P: pressure in Pa
        composition: array (n x 5) of oxide compositions, NORMALIZED to sigma(X_i) = 1
            'Al2O3', 'FeO*', 'CaO', 'Na2O', 'K2O'
        use_torch: If True, use PyTorch tensors
        device: Device for PyTorch tensors ('cpu' or 'cuda')

    Returns:
        X_Fe2O3 / X_FeO
    """
    # Constants from Table 7 for natural melts
    a = 0.196
    b = 1.1492e4
    c = -6.675
    d = np.array([-2.243, -1.828, 3.201, 5.854, 6.215]).reshape(5, 1)  # set up for matrix multiplication for sum term
    e = -3.36
    f = -7.01e-7
    g = -1.54e-10
    h = 3.85e-17
    T0 = 1673.0  # Kelvin

    if use_torch:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Install torch to use use_torch=True")
        d = torch.tensor(d, dtype=torch.float32, device=device)
        dX_sum = composition @ d
        ln_ratio = (
            a * torch.log(fO2) +
            b / T +
            c +
            dX_sum.flatten() +
            e * (1 - (T0 / T) - (torch.log(T / T0))) +
            f * P / T +
            g * (T - T0) * P / T +
            h * P ** 2 / T
        )
        return torch.exp(ln_ratio)
    else:
        dX_sum = composition @ d
        ln_ratio = (
            a * np.log(fO2) +
            b / T +
            c +
            dX_sum +
            e * (1 - (T0 / T) - (np.log(T / T0))) +
            f * P / T +
            g * (T - T0) * P / T +
            h * P ** 2 / T
        )
        return np.exp(ln_ratio)


def identify_binaries(digits):
    """
    Returns numpy array of all unique binaries possible given a number of digits.
    
    Args:
        digits (int): Number of digits
        
    Returns:
        np.ndarray or str: Array of binary combinations, or error message if too large
    """
    if 2**digits > 1E7:
        return str(f"imagine there are {2**digits} of combinations supplied here. We aren't paid enough to actually generate them :P")
    digits = int(digits)
    binaries = np.zeros((2, digits))
    binaries[1, 0] = 1
    
    for b in range(1, digits):
        new_binaries = np.copy(binaries)
        new_binaries[:, b] = 1
        binaries = np.append(binaries, new_binaries, axis=0)
        
    return binaries.astype(int)


def safe_float(x):
    """
    Safely convert value to float, returning 0.0 on error.
    
    Args:
        x: Value to convert
        
    Returns:
        float: Converted value or 0.0
    """
    try:
        return float(x)
    except:
        return 0.0


safe_convert = np.vectorize(safe_float)


def blur_binary_boundaries(arr):
    """
    For blurring boundaries between phase occurrence and disappearance for training.
    Given a 2D binary array (0s and 1s), modifies boundary transitions (column-wise)
    such that:
    - A 1 that borders a 0 is set to 0.7
    - A 0 that borders a 1 is set to 0.3

    Returns a new array with blurred boundaries.
    
    Args:
        arr (np.ndarray): 2D binary array
        
    Returns:
        np.ndarray: Array with blurred boundaries
    """
    arr = arr.astype(float)  # Ensure output is float to allow fractional values
    blurred = arr.copy()

    # Shifted arrays for comparison: above and below
    above = np.roll(arr, 1, axis=0)
    below = np.roll(arr, -1, axis=0)

    # Identify boundaries (only between 1s and 0s)
    boundary_mask = (arr != above) | (arr != below)

    # Only consider cases where neighbors are valid (not from wrap-around)
    boundary_mask[0, :] &= (arr[0, :] != below[0, :])  # Top row: compare only below
    boundary_mask[-1, :] &= (arr[-1, :] != above[-1, :])  # Bottom row: compare only above

    # Set blurred values
    blurred[(boundary_mask) & (arr == 1)] = 0.7
    blurred[(boundary_mask) & (arr == 0)] = 0.3

    return blurred


def grid_sample(params, table=np.array([])):
    """
    Generates a numpy array grid sample recursively for arbitrary parameters.
    Let params be a nested list, with each sublist of [min, max, len] passed to np.linspace.
    Order of params determines column order in the output table.
    
    Args:
        params: Nested list of [min, max, len] for each parameter
        table: Accumulated table (used in recursion)
        
    Returns:
        np.ndarray: Grid sample table
    """
    params = list(params)  # Copy to avoid side-effects
    param = params.pop()
    new_col = np.linspace(*param).reshape((-1, 1))
    
    if not table.shape[0]:
        table = new_col
    else:
        table = np.append(np.repeat(new_col, table.shape[0], axis=0),
                          np.tile(table, (new_col.shape[0], 1)), axis=1)
    
    if len(params):
        return grid_sample(params, table)
    else:
        return table

def grid_sample_explicit(params, table=np.array([])):
    """
    Generates a numpy array grid sample recursively for arbitrary parameters.
    Let params be a nested list, with each sublist of [min, max, len] passed to np.linspace.
    Order of params determines column order in the output table.
    
    Args:
        params: Nested list of [min, max, len] for each parameter
        table: Accumulated table (used in recursion)
        
    Returns:
        np.ndarray: Grid sample table
    """
    params = list(params)  # Copy to avoid side-effects
    param = params.pop()
    new_col = param # Only literal values used, more custom. 
    
    if not table.shape[0]:
        table = new_col
    else:
        print(table)
        table = np.append(np.repeat(new_col, table.shape[0], axis=0),
                          np.tile(table, (new_col.shape[0], 1)), axis=1)
    
    if len(params):
        return grid_sample_explicit(params, table)
    else:
        return table

def mix_compositions(compositions: list, fractions: list) -> dict:
    """Mix composition dictionaries by normalizing each then blending by fractions.

    Each composition is normalized so its values sum to 1.0, then the normalized
    compositions are combined using the supplied fractions (which are themselves
    normalized to sum to 1.0).

    Args:
        compositions: list of dicts mapping element/oxide names to numeric amounts
        fractions   : mixing weights, same length as compositions (need not sum to 1)

    Returns:
        dict: single mixed composition (values sum to 1.0)
    """
    assert len(compositions) == len(fractions) and len(fractions) > 0, \
        "compositions and fractions must be non-empty and the same length"

    fracs = np.asarray(fractions, dtype=np.float64)
    fracs = fracs / fracs.sum()

    all_keys = sorted({k for comp in compositions for k in comp})

    mixed = {k: 0.0 for k in all_keys}

    for comp, frac in zip(compositions, fracs):
        total = sum(comp.values())
        if total == 0.0:
            raise ValueError("A composition has all-zero values and cannot be normalized.")
        for key in all_keys:
            mixed[key] += frac * comp.get(key, 0.0) / total


    return mixed



class TernaryAxes:
    """Ternary (or trapezoidal) diagram backed by a matplotlib Axes.

    Coordinate convention
    ---------------------
    All data is passed as (B, 3) arrays where columns are
    [bottom-left fraction, bottom-right fraction, top fraction].
    Rows need not sum to 1; they are normalised internally.

    The ``max_top`` parameter clips the north (top) axis:
      * 1.0  → full equilateral triangle
      * 0.5  → trapezoid showing top 50 % of north-axis range
      * 0.1  → narrow strip near the base
    """

    _H = np.sqrt(3) / 2  # height of a unit-side equilateral triangle

    def __init__(self, ax, corner_labels, max_top: float = 1.0):
        """
        Parameters
        ----------
        ax            : matplotlib Axes
        corner_labels : sequence of 3 str — [bottom-left, bottom-right, top]
        max_top       : float in (0, 1] — fraction of north axis to display
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for TernaryAxes")
        self.ax = ax
        self.corner_labels = list(corner_labels)
        self.max_top = float(np.clip(max_top, 1e-3, 1.0))
        self._data_artists = []
        self._draw_frame()

    def _to_cartesian(self, coords: np.ndarray):
        """Convert (B, 3) ternary [left, right, top] to Cartesian x, y."""
        c = np.asarray(coords, dtype=float)
        if c.ndim == 1:
            c = c[None]
        s = c.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        c = c / s
        x = c[:, 1] + 0.5 * c[:, 2]
        y = self._H * c[:, 2]
        return x, y

    def _draw_frame(self):
        ax, m, H = self.ax, self.max_top, self._H
        BL = [0.0,       0.0    ]
        BR = [1.0,       0.0    ]
        TL = [0.5 * m,   H * m  ]
        TR = [1 - 0.5*m, H * m  ]
        verts = [BL, BR, [0.5, H]] if m >= 1.0 - 1e-6 else [BL, BR, TR, TL]
        ax.add_patch(plt.Polygon(verts, closed=True, fill=False, ec='k', lw=1.5))

        # Grid lines (4 inner divisions per axis)
        for i in range(1, 5):
            f = i / 5.0
            ct = f * m                              # constant-top value
            c_top = min(1.0 - f, m)                # max top for const-a or const-b lines

            # Horizontal: constant top fraction = ct
            ax.plot([0.5*ct, 1 - 0.5*ct], [H*ct, H*ct],
                    color='grey', lw=0.35, alpha=0.45)
            # Left-to-right diagonal: constant left fraction = f
            ax.plot([1 - f,           1 - f - 0.5*c_top],
                    [0.0,             H * c_top         ],
                    color='grey', lw=0.35, alpha=0.45)
            # Right-to-left diagonal: constant right fraction = f
            ax.plot([f,               f + 0.5*c_top    ],
                    [0.0,             H * c_top         ],
                    color='grey', lw=0.35, alpha=0.45)

        off = 0.06
        ax.text(-off, -0.04, self.corner_labels[0],
                ha='right', va='top', fontsize=20, fontweight='bold')
        ax.text(1 + off, -0.04, self.corner_labels[1],
                ha='left',  va='top', fontsize=20, fontweight='bold')
        ax.text(0.5, H*m + 0.04, self.corner_labels[2],
                ha='center', va='bottom', fontsize=20, fontweight='bold')

        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(-0.22, 1.22)
        ax.set_ylim(-0.13, H * m + 0.20)

    def clear_data(self):
        """Remove all artists added by add_points, keeping the frame."""
        for a in self._data_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._data_artists.clear()

    def add_points(self, data, text_labels=None, **kwargs):
        """Plot points on the ternary.

        Parameters
        ----------
        data        : dict {str: array-like (3,)} or ndarray (B, 3)
                      Ternary coordinates [left, right, top].  If a dict,
                      keys are used as text labels plotted next to each point.
        text_labels : sequence of str, optional — per-row labels for ndarray input
        **kwargs    : forwarded to ``ax.plot`` (e.g. marker='+', color='k', ls='none')

        Returns self for chaining.
        """
        if isinstance(data, dict):
            for lbl, coords in data.items():
                x, y = self._to_cartesian(np.asarray(coords, dtype=float).reshape(1, 3))
                lines = self.ax.plot(x, y, **kwargs)
                txt = self.ax.text(float(x[0]), float(y[0]) + 0.02, lbl,
                                   fontsize=7, ha='center', va='bottom', clip_on=True)
                self._data_artists.extend(lines)
                self._data_artists.append(txt)
        else:
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 1:
                arr = arr[None]
            x, y = self._to_cartesian(arr)
            lines = self.ax.plot(x, y, **kwargs)
            self._data_artists.extend(lines)
            if text_labels is not None:
                for xi, yi, lbl in zip(x, y, text_labels):
                    txt = self.ax.text(float(xi), float(yi) + 0.02, str(lbl),
                                       fontsize=7, ha='center', va='bottom', clip_on=True)
                    self._data_artists.append(txt)
        return self


def make_ternary(corner_labels, data=None, ax=None, max_top: float = 1.0, figsize=None):
    """Create a ternary (or trapezoidal) diagram and return a TernaryAxes.

    Parameters
    ----------
    corner_labels : sequence of 3 str — [bottom-left, bottom-right, top]
    data          : dict {str: (3,)} or ndarray (B, 3), optional — initial data
    ax            : matplotlib Axes; creates a new figure if None
    max_top       : float in (0, 1] — fraction of north axis to show
    figsize       : (w, h) for new figure

    Returns
    -------
    TernaryAxes — call ``.add_points()`` to overlay more data on the same diagram
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for make_ternary")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (5, 4))
    t = TernaryAxes(ax, corner_labels, max_top=max_top)
    if data is not None:
        t.add_points(data)
    return t


def squash_to_range(x, min_=0.1, max_=0.95):
    """
    Squash values to a range.
    
    Args:
        x: Input values
        min_: Minimum of output range
        max_: Maximum of output range
        
    Returns:
        Squashed values
    """
    return x * (max_ - min_) + min_


def unsquash_from_range(x, min_=0.1, max_=0.95):
    """
    Unsquash values from a range.
    
    Args:
        x: Input values
        min_: Minimum of input range
        max_: Maximum of input range
        
    Returns:
        Unsquashed values
    """
    return (x - min_) / (max_ - min_)


def projected_nnls(A, b, max_iter=10, lr=0.1):
    """
    Projected non-negative least squares solver.
    
    Args:
        A: (batch, n_elements, n_phases): Element contribution from each phase
        b: (batch, n_elements): Negative Element deficits in liquid
        max_iter: Maximum iterations
        lr: Learning rate
        
    Returns:
        torch.Tensor: shape (batch, n_phases)
    """
    r = torch.zeros(A.shape[0], A.shape[2], device=A.device)

    for _ in range(max_iter):
        residual = (A @ r.unsqueeze(2)).squeeze(2) - b  # (batch, n_elements)
        grad = (A.transpose(1, 2) @ residual.unsqueeze(2)).squeeze(2)  # (batch, n_phases)
        r = r - lr * grad
        r = torch.clamp(r, min=0.0)  # projection
    return r


def masked_column_assign(target, row_mask, col_idx, values):
    """
    Efficient in-place assignment for a masked subset of rows and explicit column indices.

    Args:
        target (torch.Tensor): 2D tensor to modify in place.
        row_mask (BoolTensor): Mask selecting which rows to modify (shape [n_rows]).
        col_idx (1D LongTensor): Indices of columns to assign to.
        values (Tensor): Values to assign (shape [sum(row_mask), len(col_idx)]).
    """
    # Sanity checks
    assert target.dim() == 2, "target must be 2D"
    assert values.dim() == 2, "values must be 2D"
    assert values.size(1) == col_idx.numel(), "values must match number of columns"
    assert values.size(0) == row_mask.sum(), "values must match number of selected rows"

    # Get actual row indices (1D LongTensor)
    row_idx = torch.nonzero(row_mask, as_tuple=False).squeeze(1)

    # Perform in-place indexed assignment efficiently
    target.index_put_((row_idx.unsqueeze(1).expand(-1, col_idx.numel()),
                       col_idx.unsqueeze(0).expand(row_idx.numel(), -1)),
                      values)

def projected_nnls(A, b, max_iter=10, lr=0.1):
    # A: (batch, n_elements, n_phases): Element contribution from each phase
    # b: (batch, n_elements): Negative Element deficits in liquid
    r = torch.zeros(A.shape[0], A.shape[2], device=A.device)

    for _ in range(max_iter):
        residual = (A @ r.unsqueeze(2)).squeeze(2) - b  # (batch, n_elements)
        grad = (A.transpose(1, 2) @ residual.unsqueeze(2)).squeeze(2)  # (batch, n_phases)
        r = r - lr * grad
        r = torch.clamp(r, min=0.0)  # projection
    return r  # shape: (batch, n_phases)

class Normalizer:
    """Quick Normalizing object that holds minima and ranges for a dataset and converts into and out of [0,1]
    min-max normalization for interfacing with neural networks"""
    
    def __init__(self, min_tensor, range_tensor, cuda = False):
        
        if cuda:
            self.miner = min_tensor.cuda()
            self.ranger = range_tensor.cuda()
            self.dev = 'cuda'
        else:
            self.miner = min_tensor.cpu()
            self.ranger = range_tensor.cpu()
            self.dev = 'cpu'

    def denorm(self, x):
        if isinstance(x, torch.Tensor):
            miner = self.miner.to(device=x.device, dtype=x.dtype)
            ranger = self.ranger.to(device=x.device, dtype=x.dtype)
            return x * ranger + miner
        if isinstance(x, np.ndarray):
            miner = self.miner.detach().cpu().numpy().astype(x.dtype, copy=False)
            ranger = self.ranger.detach().cpu().numpy().astype(x.dtype, copy=False)
            return x * ranger + miner
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
    
    def norm(self, x):
        if isinstance(x, np.ndarray):
            miner = self.miner.detach().cpu().numpy().astype(x.dtype, copy=False)
            ranger = self.ranger.detach().cpu().numpy().astype(x.dtype, copy=False)
            out = np.zeros_like(x, dtype=x.dtype)
            mask = ranger != 0
            out[:, mask] = (x[:, mask] - miner[mask]) / ranger[mask]
            return out
        elif isinstance(x, torch.Tensor):
            miner = self.miner.to(device=x.device, dtype=x.dtype)
            ranger = self.ranger.to(device=x.device, dtype=x.dtype)
            out = torch.zeros_like(x)
            mask = ranger != 0
            out[:, mask] = (x[:, mask] - miner[mask]) / ranger[mask]
            return out
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
    
    def to_state_dict(self):
        """
        Export Normalizer state to a dictionary.
        
        Returns a dict with 'min' and 'range' as numpy arrays for serialization.
        Device information ('cuda' or 'cpu') is included for reconstruction.
        
        Returns
        -------
        dict
            Dictionary with keys: 'min', 'range', 'device'
        """
        # Convert to numpy for JSON/NPZ serialization
        if isinstance(self.miner, torch.Tensor):
            min_array = self.miner.cpu().numpy()
            range_array = self.ranger.cpu().numpy()
        else:
            min_array = np.asarray(self.miner)
            range_array = np.asarray(self.ranger)
        
        return {
            'min': min_array.astype(np.float32),
            'range': range_array.astype(np.float32),
            'device': self.dev
        }
    
    @classmethod
    def from_state_dict(cls, state_dict, cuda=False, device='cpu'):
        """
        Reconstruct Normalizer from saved state dictionary.
        
        Parameters
        ----------
        state_dict : dict
            Dictionary with 'min' and 'range' keys containing numpy arrays
        cuda : bool, optional
            Deprecated. Use 'device' parameter instead.
        device : str, optional
            Device to place tensors on ('cpu' or 'cuda'). Defaults to 'cpu'.
        
        Returns
        -------
        Normalizer
            Reconstructed Normalizer instance
        """
        # Handle legacy cuda parameter
        if cuda:
            device = 'cuda'
        
        # Convert numpy arrays to torch tensors
        if TORCH_AVAILABLE:
            min_tensor = torch.from_numpy(state_dict['min']).float()
            range_tensor = torch.from_numpy(state_dict['range']).float()
        else:
            min_tensor = state_dict['min']
            range_tensor = state_dict['range']
        
        return Normalizer(min_tensor, range_tensor, cuda=(device == 'cuda'))

# BJT Cooked indexing helper for Ji Ching's isentropic-isothermal divide problem
def IDX_2D_Lithosphere(min_idxes, n_idx):
    """Generates two 2D indexing arrays given a list of minimum indexes for each column and the total number of column IDXs,
    where isentropic is higher and isothermal is lower"""
    isentropicxIDX = np.empty((0,), dtype=int)
    isentropicyIDX = np.empty((0,), dtype=int)
    isothermalxIDX = np.empty((0,), dtype=int)
    isothermalyIDX = np.empty((0,), dtype=int)
    for i, mIDX in enumerate(min_idxes):
        new_S_IDX = np.arange(mIDX, n_idx)
        new_T_IDX = np.arange(mIDX)
        isentropicyIDX = np.append(isentropicyIDX, new_S_IDX)
        isothermalyIDX = np.append(isothermalyIDX, new_T_IDX)
        isentropicxIDX = np.append(isentropicxIDX, np.full_like(new_S_IDX, i))
        isothermalxIDX = np.append(isothermalxIDX, np.full_like(new_T_IDX, i))
    return (isentropicxIDX.astype(int), isentropicyIDX.astype(int)), (isothermalxIDX.astype(int), isothermalyIDX.astype(int))


def match_rows(
    query: np.ndarray,
    reference: np.ndarray,
    tol: float = 1e-5,
    return_unmatched: bool = False,
) -> np.ndarray:
    """For each row in *query*, return the index of its matching row in *reference*.

    Rows are matched by rounding every value to the nearest multiple of *tol*
    and comparing the resulting integer tuples, so values differing by less than
    ``tol / 2`` are treated as identical.  Designed for correlating grid searches
    assembled by different mechanisms (e.g. ``grid_sample`` vs. a hand-built
    meshgrid).

    Parameters
    ----------
    query           : (M, F) array of rows to look up
    reference       : (N, F) array to search in
    tol             : absolute matching precision
    return_unmatched: if True, unmatched rows receive index -1 and no exception
                      is raised; if False (default) a ValueError is raised on
                      the first unmatched row

    Returns
    -------
    np.ndarray, shape (M,), dtype int
        ``reference[indices[i]]`` ≈ ``query[i]`` for every i.
        Unmatched entries are -1 when *return_unmatched* is True.

    Raises
    ------
    ValueError
        If any query row has no match and *return_unmatched* is False.
    """
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if query.ndim == 1:
        query = query[None]
    if reference.ndim == 1:
        reference = reference[None]

    ref_keys = np.round(reference / tol).astype(np.int64)
    qry_keys = np.round(query / tol).astype(np.int64)

    lookup = {tuple(row): i for i, row in enumerate(ref_keys)}

    indices = np.full(len(query), -1, dtype=np.intp)
    for i, row in enumerate(qry_keys):
        idx = lookup.get(tuple(row), -1)
        if idx == -1 and not return_unmatched:
            raise ValueError(
                f"Row {i} of query has no match in reference within tol={tol}:\n"
                f"  query[{i}] = {query[i]}"
            )
        indices[i] = idx

    return indices


