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



