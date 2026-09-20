"""
Neural Network Emulator for MELTS.

Contains the NN_MELTS class which wraps the neural network model and provides
high-level interfaces for phase prediction, mass balancing, and fractional crystallization.
"""

import time
import numpy as np
import torch
import pandas as pd

# Import utility functions
from ..utils.math_utils import QFM_fO2, Fe2O3_FeO_ratio, QFM_fO2_torch, Normalizer
from ..config.constants import OXIDE_MOLAR_MASSES as oxide_molar_masses, ptt_longs, ptt_order_oxides, ptt_to_short, ptt_oxide_indexer
from .mass_balance import MassBalanceProjector

# Mass-balance modes accepted by NN_MELTS(mass_balance=...) / forwardMB(mass_balance=...).
_MASS_BALANCE_MODES = ('iterative', 'pinv', 'none')

ferric_to_ferrous_ratio =  (2*oxide_molar_masses['FeO'])/oxide_molar_masses['Fe2O3']

class NN_MELTS:
    """
    Neural Network Emulator of MELTS.
    
    Wrapper around MidLevelNetwork providing high-level interfaces for:
    - Phase saturation prediction
    - Mass balancing and composition calculations
    - Fractional crystallization simulations
    - Unit conversions (oxides <-> elements, intensive <-> extensive)
    
    All algebraic operations and unit conversions are handled here, keeping the
    neural network focused on prediction tasks.
    """
    
    def __init__(self, model, cuda=False, mass_balance='iterative', *,
                 massbalance_iters=3, massbalance_tikhonov=1.0e-6,
                 massbalance_damping=1.0, massbalance_relative=True,
                 massbalance_weight_floor=1.0e-9):
        """
        Initialize the NN_MELTS emulator.

        Parameters
        ----------
        model : MidLevelNetwork or ContinuousModel
            The neural network model with attached ml_indexer. Both architectures emit
            raw, NON mass-balanced component moles through an identical
            `forward(detailed=True)` 7-tuple -- this wrapper is the sole owner of the
            mass-balance correction and never branches on model type.
        cuda : bool, default=False
            Whether to use CUDA/GPU acceleration
        mass_balance : {'iterative', 'pinv', 'none'}, default='iterative'
            Default correction applied by `forwardMB` (overridable per call):
            - 'iterative' : `MassBalanceProjector` (support-restricted, non-negativity
              clamped, GPU-safe, runs on `self.dev`)
            - 'pinv'      : one-shot pseudo-inverse `polish_masses` (CPU-only)
            - 'none'      : no correction; raw heads output straight through
              (for intercomparing the correctors)
        massbalance_iters, massbalance_tikhonov, massbalance_damping, massbalance_relative,
        massbalance_weight_floor
            `MassBalanceProjector` construction knobs (see that class). `relative=True`
            gives the weighted (relative) least-norm objective that `polish_masses` also
            uses, so a low-abundance phase is not corrected by an absolute amount that is
            relatively enormous.
        """
        if type(model).__name__ not in ('MidLevelNetwork', 'ContinuousModel'):
            raise TypeError(f"Model must be MidLevelNetwork or ContinuousModel, got {type(model)}")
        if not hasattr(model, 'ml_indexer') or model.ml_indexer is None:
            raise ValueError("Model must have an attached ml_indexer")
        if mass_balance not in _MASS_BALANCE_MODES:
            raise ValueError(f"mass_balance must be one of {_MASS_BALANCE_MODES}, got {mass_balance!r}")

        # Store model and set evaluation mode
        self.model = model.eval()
        self.ml_indexer = model.ml_indexer

        # Set device
        self.dev = 'cuda' if cuda else 'cpu'
        if cuda:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        # Extract and convert indexer attributes to tensors on appropriate device
        self._setup_indexer_tensors()

        # Setup normalizer from ml_indexer
        self._setup_normalizer()

        # Derive compound matrices
        self.compToEl = self.compToOx @ self.oxToEl

        # Mass balance (see apply_mass_balance). The projector lives on this wrapper, not
        # the model -- it needs only componentMoles, self.compToEl and the bulk target.
        self.mass_balance = mass_balance
        self._projector = MassBalanceProjector(
            iters=massbalance_iters, tikhonov=massbalance_tikhonov,
            damping=massbalance_damping, relative=massbalance_relative,
            weight_floor=massbalance_weight_floor,
        ).to(self.dev)


    
    def _setup_indexer_tensors(self):
        """Convert ml_indexer numpy arrays to torch tensors on correct device."""
        idx = self.ml_indexer
        dev = self.dev
        
        # Element and phase mappings
        self.Elkeys = idx.Elkeys
        self.Oxides = idx.Oxides
        self.all_phases = idx.all_phases
        self.label_indices = idx.label_indices
        self.label_indices_comp = idx.label_indices_comp
        self.detail_label_indices = idx.detail_label_indices
        self.mass_phasedict = idx.mass_phasedict
        self.compositionally_variable_subset = idx.compositionally_variable_subset
        
        # Transformation matrices
        self.compToOx = torch.tensor(idx.compToOx, dtype=torch.float32, device=dev)
        self.oxToEl = torch.tensor(idx.OxToEl, dtype=torch.float32, device=dev)
        self.MM = torch.tensor(idx.MM, dtype=torch.float32, device=dev)
        self.Minv = torch.tensor(idx.Minv, dtype=torch.float32, device=dev)
        self.Mtot = torch.tensor(idx.Mtot, dtype=torch.float32, device=dev).flatten()
        self.phaseToCompMap = torch.tensor(idx.phaseToCompMap, dtype=torch.float32, device=dev)
        self.variedToAllComp = torch.tensor(idx.variedToAllComp, dtype=torch.float32, device=dev)
        self.fixed_phaseToCompMap = torch.tensor(idx.fixed_phaseToCompMap, dtype=torch.float32, device=dev)
        self.boolTransCompToOx = torch.tensor(idx.boolTransCompToOx, dtype=torch.float32, device=dev)
        self.elToOx = torch.tensor(idx.ElToOx, dtype=torch.float32, device=dev)
        
        # Feature offset (number of non-element features before element columns)
        self.feature_offset = len(idx.featureNames)
        
        # Oxide dictionary for iron speciation
        self.oxide_dict = {ox: i for i, ox in enumerate(self.Oxides)}
    
    def _setup_normalizer(self):
        """Setup normalizer from ml_indexer state."""
        idx = self.ml_indexer
        self.norm_features = idx.feature_normalizer
        """
        if hasattr(idx, 'normalizer_state') and idx.normalizer_state is not None:
            # Use saved normalizer state
            min_array = idx.normalizer_state['min']
            range_array = idx.normalizer_state['range']
            min_tensor = torch.tensor(min_array, dtype=torch.float32)
            range_tensor = torch.tensor(range_array, dtype=torch.float32)
            self.norm_features = Normalizer(min_tensor, range_tensor, cuda=(self.dev == 'cuda'))
        else:
            print(ml_indexer.)
            raise ValueError("No normalizer state found in ml_indexer. Normalization cannot be performed.")
            # Fallback: no normalization (identity)
            dummy_shape = self.feature_offset + len(self.Elkeys)
            min_tensor = torch.zeros(dummy_shape, dtype=torch.float32)
            range_tensor = torch.ones(dummy_shape, dtype=torch.float32)
            self.norm_features = Normalizer(min_tensor, range_tensor, cuda=(self.dev == 'cuda'))
            print("Warning: No normalizer state found in ml_indexer. Using identity normalization.")"""

    def reorder_input_table(self, table, headers=None, composition_space='elements',
                            strict=False, fill_value=0.0, return_type='same'):
        """
        Reorder input table columns into NN_MELTS feature order.

        Supports either:
        - Headered table-like inputs (e.g., pandas DataFrame), or
        - Array/tensor inputs with a separate ``headers`` argument.

        Parameters
        ----------
        table : array-like, torch.Tensor, or DataFrame-like
            Input rows of conditions + composition columns.
        headers : sequence of str, optional
            Column names for ``table`` when ``table`` does not carry headers.
            If provided, these are used even when ``table`` has columns.
        composition_space : str, default='elements'
            Composition columns expected in ``table``:
            - 'elements': uses ``self.Elkeys``
            - 'oxides': uses ``self.Oxides[:len(self.Elkeys)]`` (excludes ferric column)
        strict : bool, default=True
            If True, raises on missing expected columns.
            If False, fills missing expected columns with ``fill_value``.
        fill_value : float, default=0.0
            Value used to fill missing expected columns when ``strict=False``.
        return_type : str, default='same'
            Output type:
            - 'same': keep input style when possible
            - 'numpy': return numpy.ndarray
            - 'torch': return torch.Tensor on ``self.dev``
            - 'dataframe': return pandas.DataFrame (requires pandas)

        Returns
        -------
        array-like
            Reordered table with columns in:
            ``self.ml_indexer.featureNames + composition_headers``
        """
        composition_key = composition_space.lower()
        if composition_key not in ['elements', 'oxides']:
            raise ValueError("composition_space must be 'elements' or 'oxides'")

        if composition_key == 'elements':
            composition_headers = list(self.Elkeys)
        else:
            composition_headers = list(self.Oxides[:len(self.Elkeys)]) # Eventually we will support constant ferric iron and we will need to refactor to handle

        expected_headers = list(self.ml_indexer.featureNames) + composition_headers

        has_columns = hasattr(table, 'columns') and hasattr(table, 'to_numpy')
        if headers is None:
            if not has_columns:
                raise ValueError(
                    "headers must be provided when table has no column labels"
                )
            input_headers = [str(col) for col in table.columns]
        else:
            input_headers = [str(col) for col in headers]

        if len(input_headers) != len(set(input_headers)):
            raise ValueError("Input headers contain duplicates")

        if torch.is_tensor(table):
            values = table.detach().cpu().numpy()
        elif has_columns:
            values = table.to_numpy()
        else:
            values = np.asarray(table)

        if values.ndim != 2:
            raise ValueError(f"table must be 2D, got shape {values.shape}")
        if values.shape[1] != len(input_headers):
            raise ValueError(
                f"Header count ({len(input_headers)}) does not match "
                f"table columns ({values.shape[1]})"
            )

        header_to_idx = {name: idx for idx, name in enumerate(input_headers)}
        missing = [name for name in expected_headers if name not in header_to_idx]

        if missing and strict:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing required input columns: {missing_str}")

        reordered = np.full(
            (values.shape[0], len(expected_headers)),
            fill_value,
            dtype=np.float32
        )

        for out_idx, name in enumerate(expected_headers):
            if name in header_to_idx:
                reordered[:, out_idx] = values[:, header_to_idx[name]]

        if composition_space == 'oxides':
            reordered = self.convertOxToMol(torch.tensor(reordered, dtype=torch.float32, device=self.dev), convert=True)

        out_key = return_type.lower()
        if out_key not in ['same', 'numpy', 'torch', 'dataframe']:
            raise ValueError("return_type must be one of: same, numpy, torch, dataframe")

        if out_key == 'numpy':
            return reordered

        if out_key == 'torch':
            if isinstance(reordered, torch.Tensor):
                return reordered.clone().detach().to(dtype=torch.float32, device=self.dev)
            return torch.tensor(reordered, dtype=torch.float32, device=self.dev)

        if out_key == 'dataframe':
            return pd.DataFrame(reordered, columns=expected_headers)

        if has_columns:
            try:
                return table.__class__(reordered, columns=expected_headers, index=table.index)
            except Exception:
                pass

        if torch.is_tensor(table):
            return torch.tensor(reordered, dtype=table.dtype, device=table.device)

        return reordered

    def convertOxToMol(self, features, convert=True):
        """
        Convert oxide weight percent to elemental moles.
        
        Parameters
        ----------
        features : torch.Tensor
            Input features with [intensive_features..., oxides...]
        convert : bool, default=True
            Whether to perform conversion
            
        Returns
        -------
        torch.Tensor
            Features with oxides converted to elemental moles
        """

        conditions = features[:, :self.feature_offset]

        if convert:
            # Split features into non-chemical condition and composition
            oxides = features[:, self.feature_offset:]
            
            # Convert oxides to elements
            colsize = oxides.shape[1]
            unclosed = (oxides @ self.Minv[:colsize, :colsize]) @ self.oxToEl[:colsize]
            closedmoles = unclosed / unclosed.sum(dim=1, keepdim=True)
            #pd.DataFrame(closedmoles.detach().cpu().numpy(), columns=self.ml_indexer.Elkeys).to_csv('closedmoles.csv', index=False) #Temp for grabbing element comp. 
            #print("Closed moles saved to 'closedmoles.csv'")

            return torch.cat([conditions, closedmoles], dim=1)
        else:
            unclosed = features[:, self.feature_offset:]
            closedmoles = unclosed / unclosed.sum(dim=1, keepdim=True)

        return torch.cat([conditions, closedmoles], dim=1)

    def getExtensiveComps(self, intensiveLabels, molarLabels):
        """
        Convert intensive component labels to extensive system-scale component moles.
        
        Parameters
        ----------
        intensiveLabels : torch.Tensor
            (B, VC) - Intensive component abundances (molar proportion of components within a phase)
        molarLabels : torch.Tensor
            (B, P) - Moles of each phase
            
        Returns
        -------
        componentMoles : Torch.Tensor (B, C) of extensive component moles for whole system
        """
        Cmoles = molarLabels @ self.phaseToCompMap # (B, C)
        intensiveWeights = intensiveLabels @ self.variedToAllComp # (B, C)

        # Only pure-phase components (not represented in VC) should have unit weights.
        pure_comp_mask = self.variedToAllComp.sum(dim=0) == 0
        intensiveWeights[:, pure_comp_mask] = 1.0

        componentMoles = Cmoles * intensiveWeights # (B, C)
        

        return componentMoles

    def _recon_residual(self, componentMoles, norm_features):
        """L2 norm, per row, of `normalize(componentMoles @ compToEl) - bulk_target` --
        how far the (already scale-normalised) reconstructed bulk is from the requested
        composition. Reported by `apply_mass_balance` for every mode so `'none'`,
        `'iterative'` and `'pinv'` can be compared on the same footing.
        """
        cm = componentMoles.to(self.dev)
        compToEl = self.compToEl.to(self.dev)
        bulk_dir = norm_features[:, self.feature_offset:].to(self.dev)
        bl = cm @ compToEl
        recon = bl / bl.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (recon - bulk_dir).norm(dim=1).to('cpu')

    def _assemble_component_outputs(self, componentMoles, feats, requested_outputs,
                                    comp_table_out, wtDelComponentMoles=None):
        """Build the `{selector: value}` output dict from a (corrected or raw) component-
        mole tensor. Shared by `polish_masses` (the 'pinv' backend) and
        `apply_mass_balance`'s 'iterative' / 'none' branches so there is one definition
        of what each selector means.
        """
        requested_outputs = set(requested_outputs)
        compToOx = self.compToOx.to(componentMoles.device)
        MM = self.MM.to(componentMoles.device)
        phaseToCompMap = self.phaseToCompMap.to(componentMoles.device)

        out = {}
        if 'phase_tables' in requested_outputs:
            out['phase_tables'] = self.make_phase_tables(
                componentMoles, compToOx, MM, compPhaseMap=phaseToCompMap.T,
                features=feats, eps=1e-12, out=comp_table_out,
            )
        if 'component_moles' in requested_outputs:
            out['component_moles'] = componentMoles
        if 'wt_del_component_moles' in requested_outputs:
            if wtDelComponentMoles is None:
                raise KeyError("'wt_del_component_moles' is only available from the 'pinv' "
                               "mass-balance mode")
            out['wt_del_component_moles'] = wtDelComponentMoles
        if requested_outputs & {'phase_moles', 'chem_out', 'phase_present'}:
            phaseComponentMoles = componentMoles[:, None, :] * phaseToCompMap[None, :, :]  # (B, P, C)
            phaseMoles = phaseComponentMoles.sum(dim=2)  # (B, P)
            if 'phase_moles' in requested_outputs:
                out['phase_moles'] = phaseMoles
            if 'phase_present' in requested_outputs:
                out['phase_present'] = (phaseMoles > 0).to(phaseMoles.dtype)
            if 'chem_out' in requested_outputs:
                vc = np.array(self.ml_indexer.compositionally_variable_subset).astype(int)
                out['chem_out'] = (phaseComponentMoles / (phaseMoles[:, :, None] + 1e-12)).sum(dim=1)[:, vc]
        return out

    def apply_mass_balance(self, phaseMoles, reconBulk, componentMoles, phaseProportions,
                           norm_features, *, mode=None, optimize_masses=False,
                           protect_opx=False, comp_table_out='oxides', outputs=None):
        """Correct raw model component moles onto the requested bulk, by the named method.

        `mode` (default `self.mass_balance`, set at construction):
        - 'none'      : no correction; assemble outputs from the raw moles.
        - 'iterative' : `self._projector` (MassBalanceProjector) -- support-restricted,
          non-negativity clamped, weighted least-norm, runs on `self.dev` (GPU-safe).
        - 'pinv'      : `self.polish_masses` -- one-shot pseudo-inverse, CPU-only,
          honours `optimize_masses` / `protect_opx`.

        Returns the same shape as `polish_masses`: a `(compTable, massTable)` tuple when
        `outputs is None`, else a `{selector: value}` dict. A `'reconstruction_residual'`
        selector (per-row post-correction L2 residual) is available in every mode.
        """
        mode = mode or self.mass_balance
        if mode not in _MASS_BALANCE_MODES:
            raise ValueError(f"mode must be one of {_MASS_BALANCE_MODES}, got {mode!r}")

        want_resid = outputs is not None and 'reconstruction_residual' in outputs

        if mode == 'pinv':
            if outputs is None:
                return self.polish_masses(
                    phaseMoles, reconBulk, componentMoles, phaseProportions,
                    features=norm_features, optimize_masses=optimize_masses,
                    protect_opx=protect_opx, comp_table_out=comp_table_out, outputs=None)
            keys = [k for k in outputs if k != 'reconstruction_residual']
            # residual needs the corrected component moles; request them, drop later.
            need_cm = want_resid and 'component_moles' not in keys
            polish_keys = keys + (['component_moles'] if need_cm else [])
            result = self.polish_masses(
                phaseMoles, reconBulk, componentMoles, phaseProportions,
                features=norm_features, optimize_masses=optimize_masses,
                protect_opx=protect_opx, comp_table_out=comp_table_out,
                outputs=polish_keys or ['component_moles'])
            if want_resid:
                result['reconstruction_residual'] = self._recon_residual(
                    result['component_moles'], norm_features)
                if need_cm:
                    del result['component_moles']
            return result

        # 'none' / 'iterative' ---------------------------------------------------------
        # The projector runs on self.dev (GPU-safe: batched linalg.solve); the cheap
        # phase-table bookkeeping that follows is done on CPU, matching polish_masses
        # (make_phase_tables' 'oxides' path assumes CPU-side index tensors).
        if mode == 'iterative':
            cm = componentMoles.to(self.dev)
            bulk_dir = norm_features[:, self.feature_offset:].to(self.dev)
            cm, resid = self._projector(cm, self.compToEl.to(self.dev), bulk_dir)
        else:  # 'none'
            cm = componentMoles
            resid = self._recon_residual(cm, norm_features)

        cm = cm.to('cpu')
        feats = norm_features.to('cpu')
        resid = resid.to('cpu')

        if outputs is None:
            return self.make_phase_tables(
                cm, self.compToOx.to('cpu'), self.MM.to('cpu'),
                compPhaseMap=self.phaseToCompMap.to('cpu').T,
                features=feats, eps=1e-12, out=comp_table_out,
            )

        requested = set(k for k in outputs if k != 'reconstruction_residual')
        result = self._assemble_component_outputs(cm, feats, requested, comp_table_out)
        if want_resid:
            result['reconstruction_residual'] = resid
        return result

    def forwardMB(self, features, Normalize=True, WtPercent=False, comp_table_out='oxides',
                  optimize_masses=False, protect_opx=False, outputs=None, mass_balance=None):
        """
        Forward pass through the model, then mass-balance the component moles.

        Both architectures (MidLevelNetwork, ContinuousModel) return raw, NON
        mass-balanced output through an identical 7-tuple; the correction is applied here
        by `apply_mass_balance`.

        Parameters:
        -----------
        features : torch.Tensor
            Input features
        Normalize : bool, default=True
            Whether to normalize the whole feature vector (not the composition, which is
            always closed).
        WtPercent : bool, default=False
            Whether composition inputs are weight percent, otherwise mole fraction.
        comp_table_out DEPRECATED: str, default='oxides'
            Output format: 'oxides', 'comps', 'components', or None
        optimize_masses, protect_opx : bool
            Passed through to `polish_masses`; only used when the effective
            `mass_balance` mode is 'pinv'.
        mass_balance : {'iterative', 'pinv', 'none'} or None
            Overrides `self.mass_balance` for this call. None -> use the default set at
            construction.
        outputs : sequence[str] or None, default=None
            Optional selector list. Valid values:
            - 'likelihoods'             raw model presence signal (gate prob for the
              gated model, phaseMoles>0 for the continuous one), PRE mass balance
            - 'phase_present'           post-mass-balance phase presence (0/1)
            - 'chem_out'
            - 'phase_tables'
            - 'component_moles'
            - 'phase_moles'
            - 'reconstruction_residual' per-row L2 |normalize(recon bulk) - target|

        Returns:
        --------
        tuple or dict
            outputs is None -> (compTable, massTable). Otherwise a dict of requested keys.
        """
        _VALID_MB_OUTPUTS = {
            'likelihoods', 'phase_present', 'chem_out', 'phase_tables',
            'component_moles', 'phase_moles', 'reconstruction_residual',
        }
        if Normalize:
            norm_features = self.norm_features.norm(self.convertOxToMol(features, convert=WtPercent))
        else:
            norm_features = self.convertOxToMol(features, convert=WtPercent)
        with torch.no_grad():
            likelihoods, chem_out, logMoles, reconBulk, componentMoles, phaseProportions, phaseMoles = \
                self.model.forward(norm_features, detailed=True)

            mb_kwargs = dict(mode=mass_balance, optimize_masses=optimize_masses,
                             protect_opx=protect_opx, comp_table_out=comp_table_out)

            if outputs is None:
                return self.apply_mass_balance(
                    phaseMoles, reconBulk, componentMoles, phaseProportions,
                    norm_features, outputs=None, **mb_kwargs)

            requested = []
            for key in outputs:
                if key not in _VALID_MB_OUTPUTS:
                    raise KeyError(
                        f"Unknown forwardMB output selector '{key}'. "
                        f"Valid selectors are: {', '.join(sorted(_VALID_MB_OUTPUTS))}"
                    )
                if key not in requested:
                    requested.append(key)

            # 'likelihoods' is the raw pre-correction model signal -- everything else
            # comes from apply_mass_balance. Ask it for at least one thing so it runs.
            mb_outputs = [k for k in requested if k != 'likelihoods'] or ['phase_tables']
            result = self.apply_mass_balance(
                phaseMoles, reconBulk, componentMoles, phaseProportions,
                norm_features, outputs=mb_outputs, **mb_kwargs)

            if 'likelihoods' in requested:
                result['likelihoods'] = likelihoods.detach()
            return {k: v for k, v in result.items() if k in requested}


    def forwardNN(self, features, Normalize=True, WtPercent=False, outputs=None):
        """
        Forward pass with NO mass balancing -- identical to
        `forwardMB(..., mass_balance='none')`, kept as a named entry point.

        Parameters:
        -----------
        features : torch.Tensor
            Input features
        Normalize : bool, default=True
            Whether to normalize the whole feature vector
        WtPercent : bool, default=False
            Whether composition inputs are weight percent, otherwise mole fraction
        outputs : sequence[str]
            Required. Valid: 'chem_out', 'component_moles', 'phase_moles',
            'phase_tables', 'phase_present'.

        Returns:
        --------
        dict
            The requested keys only.
        """
        _VALID = {'chem_out', 'component_moles', 'phase_moles', 'phase_tables', 'phase_present'}
        if outputs is None:
            raise ValueError("Outputs must be specified for forwardNN. Valid selectors: "
                             f"{', '.join(sorted(_VALID))}")
        bad = [k for k in outputs if k not in _VALID]
        if bad:
            raise KeyError(f"Unknown forwardNN output selector(s) {bad}. "
                           f"Valid selectors: {', '.join(sorted(_VALID))}")

        if Normalize:
            norm_features = self.norm_features.norm(self.convertOxToMol(features, convert=WtPercent))
        else:
            norm_features = self.convertOxToMol(features, convert=WtPercent)
        with torch.no_grad():
            _lik, _chem, _lm, reconBulk, componentMoles, phaseProportions, phaseMoles = \
                self.model.forward(norm_features, detailed=True)
            return self.apply_mass_balance(
                phaseMoles, reconBulk, componentMoles, phaseProportions, norm_features,
                mode='none', outputs=list(outputs))

    def Iron_Speciator(self, oxides, Normedfeatures=None, P=None, T=None, fO2=None):
        """
        Speciate iron between FeO and Fe2O3 based on fO2.
        
        Parameters
        ----------
        oxides : torch.Tensor
            Tensor of size (n, O) where columns are liquid molar oxides (except for Fe2O3)
        Normedfeatures : torch.Tensor
            Normalized features (assumed normalized)
        P : float, optional
            Pressure in bars (if not provided, uses features)
        T : float, optional
            Temperature in Celsius (if not provided, uses features)
        fO2 : float, optional
            log(fO2) relative to QFM (if not provided, uses features)
            
        Returns
        -------
        torch.Tensor
            Oxides with Fe2O3 column added
        """

        assert (P is not None) or ('Pressure(System_main)' in self.ml_indexer.featureNames), "Pressure must be provided or present in features"
        assert (T is not None) or ('Temperature(System_main)' in self.ml_indexer.featureNames), "Temperature must be provided or present in features"
        assert (fO2 is not None) or ('logfO2-QFM(System_main)' in self.ml_indexer.featureNames), "logfO2 (Delta QFM) must be provided or present in features"

        features = self.norm_features.denorm(Normedfeatures)[:, :self.feature_offset]
        unNormed = oxides.clone()[:, :len(self.Elkeys)]  # Ensure we don't grab the potentially empty ferric column
        
        fO2_composition_Nos = torch.tensor(
            [self.oxide_dict[ox] for ox in ['Al2O3', 'FeO', 'CaO', 'Na2O', 'K2O']],
            device=self.dev
        )
        fO2_composition_ind = torch.zeros(len(self.Elkeys), device=self.dev).to(torch.bool)
        fO2_composition_ind[fO2_composition_Nos] = True
        
        row_sums = unNormed.sum(dim=1, keepdim=True)
        nonzero_mask = row_sums != 0
        row_sums[~nonzero_mask] = 1.0
        temp_renorm = unNormed * (1 / row_sums)
        temp_renorm[~nonzero_mask.expand_as(unNormed)] = 0.0
        
        # Extract P, T, fO2 from features using feature names if not passed as arguments
        if P is None:
            P_idx = self.ml_indexer.featureNames.index('Pressure(System_main)') 
            P = features[nonzero_mask.flatten()][:, P_idx]
        if T is None:
            T_idx = self.ml_indexer.featureNames.index('Temperature(System_main)') 
            T = features[nonzero_mask.flatten()][:, T_idx]
        if fO2 is None:
            fO2_idx = self.ml_indexer.featureNames.index('logfO2-QFM(System_main)') 
            fO2 = features[nonzero_mask.flatten()][:, fO2_idx]
   
        T_C = T # alter-ego
        fO2_delta = fO2 
        
        # Get ferric/ferrous ratio
        IronR = Fe2O3_FeO_ratio(
            fO2=10**(QFM_fO2_torch(P=P, K=T_C + 273.15, use_torch=True) + fO2_delta),
            T=T_C + 273.15,
            P=1e5 * P,
            composition=temp_renorm[nonzero_mask.flatten()][:, fO2_composition_ind],
            use_torch=True,
            device=self.dev
        )
        
        ferricPerTot = 1 / (2 + (1 / IronR))
        ferrousPerTot = 1 / ((2 * IronR) + 1)

        ferric = torch.zeros((unNormed.size()[0], 1), device=self.dev, dtype=torch.float32)
        idx = nonzero_mask.flatten().nonzero(as_tuple=True)[0]

        ferric[idx, 0] = unNormed[idx, self.oxide_dict['FeO']] * ferricPerTot
        unNormed[idx, self.oxide_dict['FeO']] *= ferrousPerTot

        unNormed_out = torch.cat([unNormed, ferric], dim=1)
        return unNormed_out

    def get_liquid_oxides(self, componentMoles, features, compToOx=None, Normalize=True):
        """
        Molar liquid-oxide composition, with iron already speciated into
        FeO/Fe2O3 -- this is the "liquid_oxides" input MELTSAPI.
        get_property_melts_vectorized_from_assemblage expects (see that
        method's own docstring in API.py).

        This factors out exactly the liquid-phase slice of what
        make_phase_tables already does for every phase in one pass (the
        same self.ml_indexer.compToOx component->oxide projection, then
        self.Iron_Speciator for models where ferric iron isn't already a
        distinct tracked species) -- deliberately mirroring
        make_phase_tables's own tensor construction line for line (same
        compPhaseMap/einsum/comp_phasedict indexing) rather than a
        "simplified" reimplementation, so this inherits make_phase_tables's
        already-exercised correctness instead of risking a fresh P-vs-VP
        indexing mistake. NOT yet cross-checked against a real trained
        checkpoint's ml_indexer in this session -- verify comp_phasedict/
        mass_phasedict resolve as expected here before relying on this for
        real work (see get_property_melts_vectorized_from_assemblage's own
        "Scope and known gaps" docstring section for the same caveat).

        Parameters
        ----------
        componentMoles : torch.Tensor, shape (B, C)
            Raw or mass-balanced component moles in self.ml_indexer.
            label_names order -- e.g. the 'component_moles' or 'chem_out'
            output of forwardMB/forwardNN for this batch.
        features : torch.Tensor, shape (B, F)
            The SAME features passed to forwardNN/forwardMB for this batch
            (P, T, fO2, ...) -- Iron_Speciator reads P/T/fO2 off of these
            when the model needs them (open/fO2-buffered models only;
            ignored entirely for closed models where Fe3+ is already a
            distinct tracked component -- see the 'Fe3' not in self.Elkeys
            branch below, identical to make_phase_tables's own).
        compToOx : torch.Tensor, shape (C, O), optional
            Defaults to self.ml_indexer.compToOx.
        Normalize : bool, default=True
            Whether `features` is normalized already (mirrors Iron_
            Speciator's own Normedfeatures convention) -- pass
            Normalize=False if `features` is raw/unnormalized.

        Returns
        -------
        torch.Tensor, shape (B, O) or (B, O+1)
            Molar oxide composition of the liquid phase. For a closed
            model (Fe3+ already a distinct ml_indexer component), this is
            self.ml_indexer.compToOx's own O-column oxide order, Fe2O3
            already correct, unchanged. For an open (fO2-buffered) model,
            this is that same O-column order with one Fe2O3 column
            APPENDED at the end (mirrors Iron_Speciator's own output
            convention: its "oxides" input excludes Fe2O3, and it appends
            a freshly computed Fe2O3 column) -- i.e. Elkeys+1 columns.
        """
        if compToOx is None:
            compToOx = torch.tensor(self.ml_indexer.compToOx, dtype=torch.float32, device=self.dev)
        compPhaseMap = torch.tensor(self.ml_indexer.phaseToCompMap.T, dtype=torch.float32, device=self.dev)

        # Same construction as make_phase_tables: (B, C, P) masked component
        # moles, projected through compToOx to (B, P, O), then pick out the
        # liquid phase's own row -- P here is comp_phasedict's index space
        # (compositionally-variable phases), matching make_phase_tables's
        # own indexing of phaseOxMolar by comp_phasedict, not mass_phasedict.
        phaseComps = componentMoles.unsqueeze(-1) * compPhaseMap  # (B, C, P)
        phaseOxMolar = torch.einsum("bcp,co->bpo", phaseComps, compToOx)  # (B, P, O)
        liqOxMolar = phaseOxMolar[:, self.ml_indexer.comp_phasedict['melts-liquid']]  # (B, O)

        if 'Fe3' not in self.Elkeys:  # open/fO2-buffered model: speciate
            normed = features if Normalize else self.norm_features.norm(features)
            return self.Iron_Speciator(oxides=liqOxMolar.to(self.dev), Normedfeatures=normed.to(self.dev))
        return liqOxMolar  # closed model: Fe3+ already its own component, Fe2O3 column already correct


    def batched_lstsq_masked(self, A, b, mask=None, rcond=1e-6):
        """
        Batched least squares with optional masking.
        
        Parameters:
        -----------
        A : torch.Tensor
            (B, E, P) - Element contribution from each phase
        b : torch.Tensor
            (B, E) or (B, E, 1) - Target values
        mask : torch.Tensor, optional
            (B, E) boolean, True=keep row, False=mask row
        rcond : float, default=1e-6
            Reciprocal condition number for least squares
            
        Returns:
        --------
        torch.Tensor
            (B, P) - Solution
        """
        if b.ndim == 2:
            b = b.unsqueeze(-1)  # (B, E, 1)

        if mask is None:
            mask = (A.abs().sum(dim=2, keepdim=True) > 0).float()

        #print(mask.size())
        A = A * mask
        b = b * mask

        sol = torch.linalg.lstsq(A, b, rcond=rcond).solution  # (B, P, 1)
        return sol.squeeze(-1)

    def masked_pinv_no_cf(self, A, b, rcond=1e-6):
        """
        Masked pseudoinverse.
        
        Parameters:
        -----------
        A : torch.Tensor
            (E, P) or (B, E, P)
        b : torch.Tensor
            (E,) or (B, E)
        rcond : float, default=1e-6
            Reciprocal condition number
            
        Returns:
        --------
        torch.Tensor
            Solution
        """
        mask = (A.abs().sum(dim=2, keepdim=True) > 0).float()  # (E,1) or (B,E,1)
        A_masked = A * mask
        b_masked = b * mask.squeeze(-1)
        #print(A_masked)
        #print(b_masked)
        #print(mask)
        pinvA = torch.linalg.pinv(A_masked, rcond=rcond)  # (P, E) or (B, P, E)
        return torch.einsum('bce,be->bc', pinvA, b_masked)

    def clamp_descent(self, Ac, b, x0c, steps=10, lr=1e-2, device="cuda"):
        """
        Projected gradient descent to push solution nonnegative.
        
        Parameters:
        -----------
        Ac : torch.Tensor
            (B, E, P) - Element contribution from each phase
        b : torch.Tensor
            (B, E) - Target values
        x0c : torch.Tensor
            (B, P) - Initial guess
        steps : int, default=10
            Number of gradient descent steps
        lr : float, default=1e-2
            Learning rate
        device : str, default="cuda"
            Device to use for computation
            
        Returns:
        --------
        torch.Tensor
            (B, P) - Nonnegative solution
        """
        inner_start = time.time()
        if self.dev != device:
            pass_data = True
            Ac = Ac.clone().to(device)
            b = b.clone().unsqueeze(-1).to(device)
            x = x0c.clone().to(device)
        else:
            pass_data = False
            b = b.clone().unsqueeze(-1)
            x = x0c.clone()

        for _ in range(steps):
            residual = b - (Ac @ x.unsqueeze(-1))  # (B, E, 1)
            grad = -(Ac.transpose(1, 2) @ residual).squeeze(-1)  # (B, P)
            x = x - (lr * grad)
            x = x.clamp_min(0.0)  # nonnegativity projection
        
        # print(f'Descent Time: {time.time() - inner_start}')

        if pass_data:
            x = x.to(self.dev)

        return x

    def clamp_descent_newcomps(self, b, compToEl, newComps0, steps=10, lr=1e-2, device="cuda"):
        """
        Projected gradient descent to fit compositions (newComps) nonnegative.

        Parameters:
        -----------
        b : torch.Tensor
            (B, E) - bulk elements
        compToEl : torch.Tensor
            (C, E) - component -> element map
        newComps0 : torch.Tensor
            (B, C) - initial guess of components
        steps : int, default=10
            Number of gradient descent steps
        lr : float, default=1e-2
            Learning rate
        device : str, default="cuda"
            Device to use

        Returns:
        --------
        torch.Tensor
            (B, C) - Nonnegative solution
        """
        start = time.time()
        if self.dev != device:
            pass_data = True
            bc = b.clone().to(device)
            x = compToEl.clone().to(device)
            Ac = newComps0.clone().to(device)
        else:
            pass_data = False
            bc = b.clone()
            x = compToEl
            Ac = newComps0.clone()

        zero_mask = torch.ones_like(Ac, device=device)
        zero_mask[Ac == 0] = 0

        for _ in range(steps):
            residual = bc - (Ac @ x)
            grad = -(residual @ x.T)
            grad *= zero_mask
            Ac = Ac - lr * grad
            Ac = Ac.clamp_min(0.0)

        # print(f"Clamp descent time: {time.time() - start:.4f}s")
        if pass_data:
            Ac = Ac.to(self.dev)

        return Ac

    def make_phase_tables(self, newComps, compToOx=None, MM=None, compPhaseMap=None, features=None, out='oxides', eps=1e-12):
        """
        Compute phase oxide wt% tables and phase mass fractions.

        Features only needed for liquid-bearing simulations (PTfO2 iron speciation).

        Parameters:
        -----------
        newComps : torch.Tensor
            (B, C) - component abundances (moles)
        compToOx : torch.Tensor
            (C, O) - component->oxide stoichiometry
        MM : torch.Tensor
            (O, O) - diagonal matrix of oxide molar masses
        compPhaseMap : torch.Tensor
            (C, P) - component->phase membership (binary)
        features : torch.Tensor
            Input features for iron speciation
        out : str, default='oxides'
            Output format: 'oxides', 'comps', 'components', or None
        eps : float, default=1e-12
            Small epsilon for numerical stability

        Returns:
        --------
        tuple or torch.Tensor
            Phase oxide wt% tables and/or phase mass fractions
        """

        # Use ml_indexer as default if transformation matrices not provided
        if compToOx is None:
            compToOx = torch.tensor(self.ml_indexer.compToOx, dtype=torch.float32, device=self.dev)
        if MM is None:
            MM = torch.tensor(self.ml_indexer.MM, dtype=torch.float32, device=self.dev)
        if compPhaseMap is None:
            compPhaseMap = torch.tensor(self.ml_indexer.phaseToCompMap.T, dtype=torch.float32, device=self.dev)

        phaseComps = newComps.unsqueeze(-1) * compPhaseMap  # (B, C, P)

        # Convert to oxides per phase (plug in iron speciator). Moles, then grams
        phaseOxMolar = torch.einsum("bcp,co->bpo", phaseComps, compToOx)
        if 'Fe3' not in self.Elkeys: # Only apply iron speciation if ferric iron is not already included in the model! HeFESTo always has Fe3+ as a component
            liqWithFerric = self.Iron_Speciator(
                oxides=phaseOxMolar[:, self.ml_indexer.comp_phasedict['melts-liquid']].to(self.dev),
                Normedfeatures=features.to(self.dev)
            )
            phaseOxMolar[:, self.ml_indexer.comp_phasedict['melts-liquid']] = liqWithFerric

        phaseOxMass = torch.einsum("bpo,oo->bpo", phaseOxMolar, MM)

        # Compute total phase masses
        phaseMass = phaseOxMass.sum(dim=-1)  # (B, P)

        # Normalize systemwide to 100%
        systemTotal = phaseMass.sum(dim=-1, keepdim=True)  # (B, 1)
        phaseMassNorm = 100.0 * phaseMass / (systemTotal + eps)

        if out == 'oxides':
            # Normalize oxide masses within each phase to 100%
            phaseSums = phaseOxMass.sum(dim=-1, keepdim=True)  # (B, P, 1)
            phaseOxWt = 100.0 * phaseOxMass / (phaseSums + eps)
            return phaseOxWt[:, self.model.comp_binaries.cpu()], phaseMassNorm # This phase is all cpu

        elif out in ['comps', 'components']:
            # Returns extensive, chemically variable components
            phasesums = phaseComps.sum(dim=1, keepdim=True)  # (B, 1, P)
            phaseIntensive = phaseComps / (phasesums + 1E-12)  # (B, C, P)
            #print(phaseIntensive.size())
            systemComps = torch.einsum('bcp,pc->bc', phaseIntensive, compPhaseMap.T)  # (B, C)
            return systemComps, phaseMassNorm

        else:
            return phaseMassNorm

    def retrieveMassesFast(self, components, features, binaries, descent=False, pinv=False, verbose=False):
        """
        Retrieve phase masses from components and binaries.
        
        Parameters
        ----------
        components : torch.Tensor
            Intensive component matrix
        features : torch.Tensor
            Input features
        binaries : torch.Tensor
            Phase saturation binaries
        descent : bool, default=False
            Whether to use gradient descent
        pinv : bool, default=False
            Whether to use pseudoinverse
        verbose : bool, default=False
            Whether to print verbose output
            
        Returns
        -------
        tuple
            (compTens, massTens) - component and mass tensors
        """
        if self.dev == 'cuda' and not descent:
            fundev = 'cpu'
            # print('converting...')
            self.compToEl = self.compToEl.to('cpu')
            self.compToOx = self.compToOx.to('cpu')
            self.oxToEl = self.oxToEl.to('cpu')
            components = components.to('cpu')
            features = features.to('cpu')
            binaries = binaries.to('cpu')
        else:
            fundev = self.dev

        start = time.time()
        nrows = components.size()[0]
        ncomps = self.ml_indexer.ncomps
        nphases = self.ml_indexer.nphases
        bulk = features[:, self.feature_offset:].clone()

        # Organize phaseToComp Matrix w/ NN output
        phaseToComp = torch.zeros((nrows, nphases, ncomps), device=fundev)  # (B, P, C)
        phaseToCompMap = torch.zeros((nphases, ncomps), device=fundev)  # (P, C)

        for phase, binary_ind in self.mass_phasedict.items():
            phaseToCompMap[binary_ind, self.label_indices[phase]] = 1
            if len(self.label_indices[phase]) > 1:
                phaseToComp[:, binary_ind, self.label_indices[phase]] = components[:, self.label_indices_comp[phase]]
            else:
                phaseToComp[:, binary_ind, self.label_indices[phase]] = binaries[:, binary_ind].unsqueeze(-1)

        #print(self.compToOx.size())
        #   print(self.compToOx.device)
        
        # Calculate phaseToEl/Ox matrix
        phaseToOx = torch.einsum('bpc,co->bpo', phaseToComp, self.compToOx)  # (B, P, O)
        phaseToEl = torch.einsum('bpo,oe->bpe', phaseToOx, self.oxToEl)  # (B, P, E)

        A = phaseToEl.transpose(1, 2)
        #print(A)
        if pinv:
            molPhase0 = self.masked_pinv_no_cf(A, bulk)
        else:
            molPhase0 = self.batched_lstsq_masked(A, bulk)

        if descent:
            molPhase = self.clamp_descent(A, b=bulk, x0c=molPhase0, steps=10, lr=1e-2)
        else:
            molPhase = molPhase0.clone()
        molPhase = molPhase * binaries
        molPhase[molPhase < 1E-6] = 0

        # Relax the constant composition constraint
        residuals = (phaseToEl.transpose(1, 2) @ molPhase.unsqueeze(-1)).squeeze(-1) - bulk

        # Nomenclature from Asimow and Ghiorso, 1998
        Msol = torch.diag_embed(torch.einsum('bp,bpc->bc', molPhase, phaseToComp))

        # Build weighted matrix
        A2 = (Msol @ self.compToEl).transpose(1, 2)  # (B, E, C)
        res = residuals  # (B, E)

        if pinv:
            wtDelComp = self.masked_pinv_no_cf(A2, res)
        else:
            wtDelComp = self.batched_lstsq_masked(A2, res)

        delComp = torch.einsum('bcc,bc->bc', Msol, wtDelComp).squeeze(-1)
        oldComps = torch.einsum('bcp,bpZ->bcZ', phaseToComp.transpose(1, 2), molPhase.unsqueeze(-1)).squeeze(-1)
        newComps = oldComps - delComp

        if descent:
            descend_time = time.time()
            newComps1 = self.clamp_descent_newcomps(bulk, self.compToEl, newComps, steps=20, lr=1e-3, device=self.dev)
            newBulk1 = newComps1 @ self.compToEl
            res1 = newBulk1 - bulk

            Msol1 = torch.diag_embed(newComps1)
            A3 = (Msol1 @ self.compToEl).transpose(1, 2)  # (B, E, C)
            if pinv:
                wtDelComp1 = self.masked_pinv_no_cf(A3, res1)
            else:
                wtDelComp1 = self.batched_lstsq_masked(A3, res1)
            DelComp1 = torch.einsum('bcc,bc->bc', Msol1, wtDelComp1).squeeze(-1)
            newComps = newComps1 - DelComp1
            # print(f'Time for second Descent and residual fitting: {time.time() - descend_time}')

        if self.dev == 'cuda' and not descent:
            self.compToEl = self.compToEl.to('cuda')
            self.compToOx = self.compToOx.to('cuda')
            self.oxToEl = self.oxToEl.to('cuda')

        # print(f'Total Linear Algebra Time: {time.time() - start} seconds')

        compTens, massTens = self.make_phase_tables(
            newComps, self.compToOx, self.MM, compPhaseMap=phaseToCompMap,
            features=features, eps=1e-12
        )
        compTens = compTens[:, self.model.comp_binaries].detach().cpu().numpy()
        massTens = massTens.detach().cpu().numpy()
        return compTens, massTens


    def polish_masses(self, phaseMoles, reconBulk, componentMoles, phaseProportions, features,
                      optimize_masses=True, output_componentMoles=False, protect_opx=False,
                      comp_table_out='oxides',outputs=None):
        """
        Polish masses to fit bulk composition.
        
        Parameters:
        -----------
        phaseMoles : torch.Tensor
            Phase moles from model
        reconBulk : torch.Tensor
            Reconstructed bulk from model
        componentMoles : torch.Tensor (B, C)
            extensive component moles from model 
        phaseProportions : torch.Tensor
            Phase proportions from model
        features : torch.Tensor
            Input features
        optimize_masses : bool, default=True
            Whether to optimize phase masses first
        output_componentMoles : bool, default=False
            Whether to output component moles
        protect_opx : bool, default=False
            Whether to protect orthopyroxene during fitting
        comp_table_out : str, default='oxides'
            Output format
            
        outputs : sequence[str] or None, default=None
            Optional selector list. Valid values are:
            - 'phase_tables'
            - 'component_moles'
            - 'wt_del_component_moles'
            - 'phase_moles'
            - 'chem_out'

        Returns:
        --------
        tuple, torch.Tensor, or dict
            If outputs is None, preserves historical return behavior.
            If outputs is provided, returns a dict with requested keys only.
        """

        phaseMoles = phaseMoles.to('cpu')
        reconBulk = reconBulk.to('cpu')
        componentMoles = componentMoles.to('cpu')
        phaseProportions = phaseProportions.to('cpu')
        feats = features.to('cpu')
        bulk = feats[:, self.feature_offset:]
        compToEl = self.compToEl.to('cpu')
        compToOx = self.compToOx.to('cpu')
        phaseToCompMap = self.phaseToCompMap.to('cpu')
        MM = self.MM.to('cpu')

        residual = bulk - reconBulk
        if optimize_masses:
            phaseComponentMoles = componentMoles[:, None, :] * phaseToCompMap[None, :, :]  # (B, P, C)
            phaseAtomMoles = torch.einsum('bpc,ce->bpe', phaseComponentMoles, compToEl)  # (B, P, E)
            #print(phaseAtomMoles[:3])
            wtDelPhaseMoles = ((torch.linalg.pinv(phaseAtomMoles.transpose(1, 2)) @ residual.unsqueeze(-1)).squeeze(-1)).clamp(-0.8, 0.8) # Do not allow phases to go negative
            DelPhaseMoles = wtDelPhaseMoles * phaseMoles
            #print('Before Phase Mass adjustments, phasemoles, then residual')
            #print(phaseMoles)
            #print(residual)
            phaseMoles = phaseMoles + DelPhaseMoles
            componentMoles = phaseProportions * (phaseMoles @ phaseToCompMap)
            reconBulk = componentMoles @ compToEl
            residual = bulk - reconBulk
            #print('After Phase Mass adjustments, phasemoles, then residual')
            #print(phaseMoles)
            #print(residual)

        # Solve underconstrained problem
        componentAtomMoles = componentMoles.unsqueeze(-1) * compToEl  # (B, C, E)
        if protect_opx:
            componentAtomMoles[:, self.label_indices['orthopyroxene']] *= 0
        #print('componentAtomMoles')
        #print(componentAtomMoles)
        #if torch.any(torch.isnan(componentAtomMoles)):
        #    # print(f"NaN values found in componentAtomMoles at indices: {torch.where(torch.isnan(componentAtomMoles))}")
        wtDelComponentMoles = (torch.linalg.pinv(componentAtomMoles.transpose(1, 2)) @ residual.unsqueeze(-1)).squeeze(-1)
        DelComponentMoles = wtDelComponentMoles * componentMoles
        componentMoles = componentMoles + DelComponentMoles

        _VALID = {'phase_tables', 'component_moles', 'wt_del_component_moles',
                  'phase_moles', 'chem_out', 'phase_present'}
        if outputs is not None:
            bad = [k for k in outputs if k not in _VALID]
            if bad:
                raise KeyError(f"Unknown polish_masses output selector(s) {bad}. "
                               f"Valid selectors are: {', '.join(sorted(_VALID))}")

        if outputs is None:
            phase_tables = self.make_phase_tables(componentMoles, compToOx, MM,
                                                  compPhaseMap=phaseToCompMap.T,
                                                  features=feats, eps=1e-12, out=comp_table_out)
            if output_componentMoles:
                return phase_tables, componentMoles, wtDelComponentMoles
            return phase_tables

        return self._assemble_component_outputs(
            componentMoles, feats, set(outputs), comp_table_out,
            wtDelComponentMoles=wtDelComponentMoles,
        )

    def find_liquidus(self, features, resolution=25):
        if 'melts-liquid' not in self.ml_indexer.label_indices_comp:
            print("Model does not include a liquid phase, returning 2000 C as default.")
            return 2000
        """Returns lowest identified superliquidus temperature between 800 and 2000 C"""
        T_idx = self.ml_indexer.featureNames.index('Temperature')
        T_test = torch.tensor(np.linspace(800, 2000, int(1200/resolution) + 1), device=self.dev)
        feat_input = np.zeros((int(1200/resolution) + 1, len(features)))
        feat_input[:] = features
        feat_input[:, T_idx] = T_test
        binaries = self.forward_binary(torch.tensor(feat_input, device=self.dev) > 0.5).float()
        liquids = binaries[:, :-1].sum(dim=1) == 0
        lowL = torch.where(liquids)[0]
        if len(lowL):
            temp = T_test[lowL[0]]
        else:
            temp = 2000
        return temp

    def find_liquidi(self, features, resolution=25, weightOxinput=False):
        """
        Vectorized version of find_liquidus.
        
        Parameters:
        -----------
        features : torch.Tensor
            (N, F) where F >= 2 and temperature is in featureNames
        resolution : int, default=25
            Temperature resolution
        weightOxinput : bool, default=False
            Whether input is in weight percent
            
        Returns:
        --------
        torch.Tensor
            (N) - Liquidus temperatures for each composition
        """

        if 'melts-liquid' not in self.ml_indexer.label_indices_comp:
            print("Model does not include a liquid phase, returning 2000 C as default.")
            return torch.full((features.shape[0],), 2000.0, device=self.dev)
        
        if weightOxinput:
            features_batch = self.convertOxToMol(features)
        else:
            features_batch = features
        N, F = features_batch.shape
        T_idx = self.ml_indexer.featureNames.index('Temperature')
        T_test = torch.linspace(800, 2000, steps=int(1200 / resolution) + 1)
        n_temps = T_test.shape[0]

        feat_input = features_batch.unsqueeze(1).repeat(1, n_temps, 1)  # [N, T, F]
        feat_input[:, :, T_idx] = T_test.unsqueeze(0).repeat(N, 1)
        feat_input = feat_input.view(N * n_temps, F)

        with torch.no_grad():
            binaries = (self.forward_binary(feat_input) > 0.5).float()

        fully_liquid = binaries[:, :-1].sum(dim=1) == 0
        fully_liquid = fully_liquid.view(N, n_temps)

        first_liquid_idx = torch.argmax(fully_liquid.to(torch.int), dim=1)
        has_liquid = fully_liquid.any(dim=1)

        liquidus_temperatures = T_test[first_liquid_idx]
        liquidus_temperatures[~has_liquid] = 2000.0

        return liquidus_temperatures

    def fractional_crystalization(self, features, T_path, fit_residual=True, WtPercent=True):
        """
        Perform fractional crystallization simulation.

        Parameters:
        -----------
        features : torch.Tensor
            (nB, n_features) where features follow ml_indexer ordering:
            - [:, :feature_offset] = intensive variables (P, T, fO2, etc.)
            - [:, feature_offset:] = elemental/oxide composition (normalized to 1)
        T_path : list or torch.Tensor
            Sequence of temperatures to iterate through
        fit_residual : bool, default=True
            Whether to fit residuals for mass balance
        WtPercent : bool, default=True
            Whether input is in weight percent
            
        Returns:
        --------
        tuple
            (component_tensor, mass_tensor) - Component and mass evolution over T_path
        """

        if 'melts-liquid' not in self.ml_indexer.label_indices_comp:
            raise ValueError("Model does not include a liquid phase, cannot perform fractional crystallization.")
        
        with torch.no_grad():
            if WtPercent:
                inp_tensor = self.convertOxToMol(features)
            else:
                inp_tensor = features

            nB = inp_tensor.size(0)
            nEl = self.oxToEl.size(1)
            nC = self.ml_indexer.ncomps
            nP = self.ml_indexer.nphases
            nSteps = len(T_path)
            T_idx = self.ml_indexer.featureNames.index('Temperature')

            component_tensor = torch.zeros((nB, nC, nSteps), dtype=torch.float32, device=self.dev)
            mass_tensor = torch.zeros((nB, nP, nSteps), dtype=torch.float32, device=self.dev)

            is_alive = torch.ones(nB, dtype=torch.bool, device=self.dev)
            prev_melt_frac = torch.ones(nB, dtype=torch.float32, device=self.dev)
            active_comps = inp_tensor[:, self.feature_offset:].clone().to(self.dev)

            for i, temp in enumerate(T_path):
                #print(temp)
                if not is_alive.any():
                    # print('All systems froze!')
                    break

                idx_alive = torch.nonzero(is_alive, as_tuple=True)[0]
                inp_batch = inp_tensor[idx_alive]
                inp_batch[:, T_idx] = temp
                inp_batch[:, self.feature_offset:] = active_comps

                if inp_batch.numel() == 0:
                    raise RuntimeError("inp_batch is empty before normalization.")

                #print(inp_batch)
                #print(idx_alive)
                _, transcomponent_hat, logMoles, reconBulk, componentMoles, phaseProportions, phaseMoles = self.model.forward(
                    self.norm_features.norm(inp_batch), detailed=True
                )
                
                if fit_residual:
                    polish_out = self.polish_masses(
                        phaseMoles,
                        reconBulk,
                        componentMoles,
                        phaseProportions,
                        features=self.norm_features.norm(inp_batch),
                        optimize_masses=False,
                        protect_opx=True,
                        comp_table_out='None',
                        outputs=['phase_tables', 'component_moles'],
                    )
                    massTens = polish_out['phase_tables']
                    componentMoles2 = polish_out['component_moles']
                    component_tensor[idx_alive, :, i] = componentMoles2
                    mass_tensor[idx_alive, :, i] = massTens * prev_melt_frac[idx_alive, None]
                else:
                    massTens = self.make_phase_tables(
                        newComps=componentMoles, compToOx=self.compToOx, MM=self.MM,
                        compPhaseMap=self.phaseToCompMap.T,
                        features=self.norm_features.norm(inp_batch), eps=1e-12, out=None
                    )
                    component_tensor[idx_alive, :, i] = componentMoles
                    mass_tensor[idx_alive, :, i] = massTens * prev_melt_frac[idx_alive, None]
                    new_liquid_el = transcomponent_hat[:, -nEl].clone()

                prev_melt_frac = (mass_tensor[:, -1, i].clone()) / 100
                is_alive = prev_melt_frac > 0.005

                new_liquid_el = component_tensor[:, -nEl:, i].clone()
                active_comps = new_liquid_el[is_alive].clone()

            return component_tensor, mass_tensor

    ## Staging algorithm with batch size parameter
    def _STAGE(func, input, batch_size = 2**16, **kwargs):
        """
        Staged execution of a function with batching.
        
        Parameters:
        -----------
        func : callable
            Function to execute in stages. Should accept an arbitrary size inputs and return list of outputs.
        input : torch.Tensor
            Input data to be processed in stages.
        batch_size : int, default=2**16
            Number of samples to process in each batch
        **kwargs :
            Additional keyword arguments to pass to func

        Returns:
        function outputs in list
        """

        if input.size(0) <= batch_size: # Trial run without batching if input is small
             return func(input, **kwargs)
        
        else:
            additional_cycles = (input.size(0) // batch_size) 
            outs = func(input[:batch_size], **kwargs) # first outputs
            for batch in (np.arange(additional_cycles)+1):
                stop = min((batch+1)*batch_size, input.size(0))
                batch_outs = func(input[batch*batch_size:stop], **kwargs)
                for i in range(len(outs)):
                    outs[i] = torch.cat((outs[i], batch_outs[i]), dim=0)

            return outs



