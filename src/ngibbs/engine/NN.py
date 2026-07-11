"""
Neural Network Architecture / object.

Contains TunableModel base class and MidLevelNetwork subclass for MELTS emulator.
TunableModel is never used without the MidLevelNetwork. MidLevelNetwork handles the enforcement of physical constraints (e.g. mass balance) constraints during inference
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import zipfile
from pathlib import Path
from datetime import datetime

# Import utility functions
from ..utils.string_utils import pull_letter, pull_number_range, apply_type_conversions

# Import constants and mappings from config (fallbacks)
from ..config.constants import TYPE_CONVERSION_MAP
"""from nMELTS.config import (
    Elkeys as DEFAULT_ELKEYS,
    label_indices as DEFAULT_LABEL_INDICES,
    label_indices_comp as DEFAULT_LABEL_INDICES_COMP,
    compToOx as DEFAULT_COMP_TO_OX,
    oxToEl as DEFAULT_OX_TO_EL,
    MM as DEFAULT_MM,
    Minv as DEFAULT_MINV,
    Mtot as DEFAULT_MTOT,
    phaseToCompMap as DEFAULT_PHASE_TO_COMP,
    variedToAllComp as DEFAULT_VARIED_TO_ALL_COMP,
    fixed_phaseToCompMap as DEFAULT_FIXED_PHASE_TO_COMP,
    boolTransCompToOx as DEFAULT_BOOL_TRANS_COMP_TO_OX,
    compositionally_variable_subset as DEFAULT_COMP_VAR_SUBSET,
)"""


class VariableGeometryFCNNRegressor(nn.Module):
    """Small fully connected regressor with configurable hidden geometry."""

    def __init__(self, input_dim, output_dim, hidden_dims=(256, 128, 64), activation_leak=0.05, dropout=0.0):
        super().__init__()

        assert int(input_dim) > 0, "input_dim must be > 0"
        assert int(output_dim) > 0, "output_dim must be > 0"
        assert float(dropout) >= 0.0 and float(dropout) < 1.0, "dropout must be in [0, 1)"
        assert hidden_dims is not None and len(hidden_dims) > 0, "hidden_dims must contain at least one layer width"

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        assert min(self.hidden_dims) > 0, "All hidden_dims values must be > 0"
        self.activation_leak = float(activation_leak)
        self.dropout = float(dropout)

        layers = []
        self._dropout_layers = []
        prev_dim = self.input_dim
        for width in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, width))
            if self.activation_leak > 0:
                layers.append(nn.LeakyReLU(self.activation_leak))
            else:
                layers.append(nn.ReLU())
            dropout_layer = nn.Dropout(self.dropout)
            layers.append(dropout_layer)
            self._dropout_layers.append(dropout_layer)
            prev_dim = width

        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.network = nn.Sequential(*layers)

        self.config = {
            "model_class": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": list(self.hidden_dims),
            "activation_leak": self.activation_leak,
            "dropout": self.dropout,
        }

    def forward(self, x):
        return self.network(x)

    def set_dropout_rate(self, dropout_rate):
        dropout_rate = float(dropout_rate)
        assert 0.0 <= dropout_rate < 1.0, "dropout_rate must be in [0, 1)"
        self.dropout = dropout_rate
        self.config["dropout"] = dropout_rate
        for layer in self._dropout_layers:
            layer.p = dropout_rate


def _get_input_min_range(payload):
    if "input_min" in payload and "input_range" in payload:
        return payload["input_min"], payload["input_range"]
    if "input_min_range" in payload:
        return payload["input_min_range"]
    raise KeyError("Temperature checkpoint payload is missing input normalization data")


def _get_target_min_range(payload):
    if "target_min" in payload and "target_range" in payload:
        return payload["target_min"], payload["target_range"]
    if "target_min_range" in payload:
        return payload["target_min_range"]
    raise KeyError("Temperature checkpoint payload is missing target normalization data")

def _load_temperature_model( # Loading function for get_T FCNN for isentropic models
    checkpoint_path: Path,
    device: torch.device,
): # -> Tuple[VariableGeometryFCNNRegressor, Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = payload.get("model_config")
    if not model_config:
        raise KeyError(f"Missing model_config in temperature checkpoint: {checkpoint_path}")

    model = VariableGeometryFCNNRegressor(
        input_dim=int(model_config["input_dim"]),
        output_dim=int(model_config["output_dim"]),
        hidden_dims=list(model_config["hidden_dims"]),
        activation_leak=float(model_config.get("activation_leak", 0.05)),
        dropout=float(model_config.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    x_min, x_range = _get_input_min_range(payload)
    y_min, y_range = _get_target_min_range(payload)
    return model, payload, x_min, x_range, y_min, y_range

class TunableModel(nn.Module):
    def __init__(self,
                 encoderLayerUp=1, # Number of layers to expand encodings, feature detection
                 encoderLayerDown=0, # Number of layers to downscale the encodings, feature compression
                 #satLayerDown = 0,
                 middleLayerUp=2,
                 middleLayerDown=1,
                 low_regularization='none',
                 high_regularization='none',
                 activation_leak=0.05,
                 ml_indexer=None):
        """
        regularization: 'none' | 'batchnorm' | 'dropout<frac>' (e.g. 'dropout0.2')
        activation_factory: a callable returning an nn.Module activation (default: LeakyReLU)
        Note: code uses ml_indexer-derived mappings.
        """
        super().__init__()

        low_regname = pull_letter(low_regularization).lower()
        high_regname = pull_letter(high_regularization).lower()

        assert low_regname in ['none', 'batchnormdropout', 'batchnorm', 'dropout', 'layernorm', 'layernormdropout'], \
            "low_regularization arg must be one of: ['none', 'batchnormdropout', 'batchnorm' 'dropout', 'layernorm', 'layernormdropout']"
        assert high_regname in ['none', 'batchnormdropout', 'batchnorm', 'dropout', 'layernorm', 'layernormdropout'], \
            "high_regularization arg must be one of: ['none', 'batchnormdropout', 'batchnorm' 'dropout', 'layernorm', 'layernormdropout']"
        assert encoderLayerUp >= encoderLayerDown, "encoderLayerUp must be >= encoderLayerDown"
        assert middleLayerUp >= middleLayerDown, "middleLayerUp must be >= middleLayerDown"

        activation_factory = lambda: nn.LeakyReLU(activation_leak) if activation_leak > 0 else lambda: nn.ReLU()

        self._set_indexer(ml_indexer)

        self.n_phases = len(list(self.label_indices.keys()))
        input_dim = len(ml_indexer.featureNames) + len(self.Elkeys)

        self.activation_factory = activation_factory
        self.input_dim = input_dim
        #self.molar_epsilon = ml_indexer.molar_epsilon

        # Regularization factory -> returns list of modules to append to layer list

        #Build Regularization Sequences for encoder and middleBrain

        def low_reg_modules(neurons):

            lowRegSequence = []
            if 'batchnorm' in low_regname:
                lowRegSequence.append(nn.BatchNorm1d(int(neurons)))
            if 'layernorm' in low_regname:
                lowRegSequence.append(nn.LayerNorm(int(neurons)))

            lowRegSequence.append(activation_factory())

            if 'dropout' in low_regname:
                fraction, _ = pull_number_range(low_regularization)
                assert fraction is not None and 0.0 <= fraction < 1.0, "dropout fraction must be between 0 and 1"
                lowRegSequence.append(nn.Dropout(fraction))

            return lowRegSequence

        def high_reg_modules(neurons):
            
            highRegSequence = []
            if 'batchnorm' in high_regname:
                highRegSequence.append(nn.BatchNorm1d(int(neurons)))
            if 'layernorm' in high_regname:
                highRegSequence.append(nn.LayerNorm(int(neurons)))

            highRegSequence.append(activation_factory())

            if 'dropout' in high_regname:
                fraction, _ = pull_number_range(high_regularization)
                assert fraction is not None and 0.0 <= fraction < 1.0, "dropout fraction must be between 0 and 1"
                highRegSequence.append(nn.Dropout(fraction))

            return highRegSequence

        # --- Build encoder ---
        encoder_layers = []
        prev_neuron = int(64)

        # first layer
        encoder_layers.append(nn.Linear(input_dim, prev_neuron))
        encoder_layers += low_reg_modules(prev_neuron)

        # expand up
        for _ in range(encoderLayerUp):
            next_neuron = prev_neuron * 2
            encoder_layers.append(nn.Linear(prev_neuron, next_neuron))
            encoder_layers += low_reg_modules(next_neuron)
            prev_neuron = next_neuron

        # downscale
        for _ in range(encoderLayerDown):
            next_neuron = max(1, prev_neuron // 2)
            encoder_layers.append(nn.Linear(prev_neuron, next_neuron))
            encoder_layers += low_reg_modules(next_neuron)
            prev_neuron = next_neuron

        self.encoder = nn.Sequential(*encoder_layers)  # shared encoder
        self._encoder_output_dim = prev_neuron


        ########### UPDTED NOV 12 TO TRY 1-OUTPUT PHASE-SPECIFIC HEADS AGAIN AFTER POOR PERFORMANCE
        # --- saturation heads (one binary head per phase) ---
        self.sat_head = nn.ModuleList()
        for _ in range(self.n_phases):
            head_layers = [nn.Linear(self._encoder_output_dim, 16)] #Intermediate  feature compression step
            #head_layers = [nn.Linear(self._encoder_output_dim, 1)]
            head_layers += low_reg_modules(16)
            head_layers.append(nn.Linear(16, 1))
            self.sat_head.append(nn.Sequential(*head_layers))

        #self.sat_head = nn.Sequential(nn.Linear(self._encoder_output_dim, self.n_phases)) 



        # placeholders for chemical heads; we must build them below after we compute number of chem-heads
        # we'll collect mappings as lists so subclass can use them
        self.comp_mappingsL = []    # list mapping component index -> chem-head index
        self.comp_binariesL = []    # which phases have >1 component (indices)
        self.chem_list_templates = []  # temporary placeholders for PhaseHead constructors

        # --- Middle brain (build but wait for comp counts) ---
        # We'll build middle brain after we know how many chem heads there are.
        self.middleBrain = None
        self._middleLayerUp = middleLayerUp
        self._middleLayerDown = middleLayerDown
        self._reg_modules = high_reg_modules

        # Mole head placeholder (built after we know middle output dim)
        self.mole_head = None

        # --- Define PhaseHead as inner class ---
        class PhaseHead(nn.Module):
            def __init__(self, n_components, input_dim):
                super().__init__()
                self.fc = nn.Linear(input_dim, n_components)

            def forward(self, x, inf_mask=None, train_inf_mask=None):
                raw = self.fc(x)
                if train_inf_mask is not None:
                    raw = raw.clone()
                    raw[train_inf_mask] = -1e9
                    proportions = F.softmax(raw, dim=-1)
                elif inf_mask is not None:
                    raw = raw.clone()
                    raw[inf_mask] = -float('inf')
                    proportions = F.softmax(raw, dim=-1)
                    proportions[torch.isnan(proportions)] = 0.0
                else:
                    proportions = F.softmax(raw, dim=-1)
                return proportions

        self._PhaseHead = PhaseHead

    def finish_build(self):
        """
        After creating TunableModel instance, call finish_build() to finalize:
         - create chem heads from label_indices
         - build middleBrain and mole_head
        This separation allows use of indexer mappings that may be set after __init__.
        """
        # build comp mappings and chem heads
        j = 0
        self.chem_heads = nn.ModuleList()
        for i, (label, inds) in enumerate(self.label_indices.items()):
            n_components = len(inds)
            if n_components > 1:
                self.comp_binariesL.append(i)
                self.comp_mappingsL += np.repeat(j, n_components).tolist()
                # create PhaseHead template: input_dim will be middle output (unknown yet) so use encoder output for now
                # We'll set proper input_dim when constructing middle/mole heads below; for now append tuple info
                self.chem_list_templates.append((n_components, i))  # store n_components and which phase
                j += 1

        self._n_chem_heads = len(self.chem_list_templates)

        # --- Build middle brain now that we can decide input dim ---
        # Input to middle brain is encoder output dim + n_phases
        prev_neuron = self._encoder_output_dim
        middle_in = prev_neuron + self.n_phases + self.input_dim

        middle_layers = []
        if self._middleLayerUp >= 1:
            next_neuron = int(prev_neuron * 2)
            middle_layers.append(nn.Linear(middle_in, next_neuron))
            middle_layers += self._reg_modules(next_neuron)
            prev_neuron = next_neuron

            for _ in range(self._middleLayerUp - 1):
                next_neuron = prev_neuron * 2
                middle_layers.append(nn.Linear(prev_neuron, next_neuron))
                middle_layers += self._reg_modules(next_neuron)
                prev_neuron = next_neuron

            for _ in range(self._middleLayerDown):
                next_neuron = max(1, prev_neuron // 2)
                middle_layers.append(nn.Linear(prev_neuron, next_neuron))
                middle_layers += self._reg_modules(next_neuron)
                prev_neuron = next_neuron

            self.middleBrain = nn.Sequential(*middle_layers)
            middle_out = prev_neuron
        else:
            # no middle brain; core output is encoder + binaries
            self.middleBrain = None
            middle_out = prev_neuron + self.n_phases

        # --- Mole head ---
        mole_layers = [nn.Linear(middle_out, 64)]
        mole_layers += self._reg_modules(64)
        mole_layers.append(nn.Linear(64, self.n_phases))
        if not self.ml_indexer.molar_epsilon:
            # print("YOU HAVE INSTANTIATED A LINEAR MOLE HEAD WITH SOFTPLUS ACTIVATION.")
            mole_layers.append(nn.Softplus()) # Not a good fit with logspace transform
        #else:
        #    # print("YOU HAVE INSTANTIATED A LOG-SPACE MOLE HEAD (eps: {}) WITH NO ACTIVATION.".format(self.ml_indexer.molar_epsilon))
        self.mole_head = nn.Sequential(*mole_layers)

        # Now build actual chem_heads with correct input_dim = middle_out
        for (n_components, phase_index) in self.chem_list_templates:
            self.chem_heads.append(self._PhaseHead(n_components=n_components, input_dim=middle_out))

        # record some handy tensors/arrays as plain Python lists so subclass can register buffers
        self.comp_mappingsL = self.comp_mappingsL
        self.comp_binariesL = self.comp_binariesL
        self._n_chem_head_count = len(self.chem_heads)

    def _set_indexer(self, ml_indexer):
        self.ml_indexer = ml_indexer

        """if ml_indexer is None:
            self.Elkeys = DEFAULT_ELKEYS
            self.label_indices = DEFAULT_LABEL_INDICES
            self.label_indices_comp = DEFAULT_LABEL_INDICES_COMP
            self.compToOx_raw = DEFAULT_COMP_TO_OX
            self.oxToEl_raw = DEFAULT_OX_TO_EL
            self.MM_raw = DEFAULT_MM
            self.Minv_raw = DEFAULT_MINV
            self.Mtot_raw = DEFAULT_MTOT
            self.phaseToCompMap_raw = DEFAULT_PHASE_TO_COMP
            self.variedToAllComp_raw = DEFAULT_VARIED_TO_ALL_COMP
            self.fixed_phaseToCompMap_raw = DEFAULT_FIXED_PHASE_TO_COMP
            self.boolTransCompToOx_raw = DEFAULT_BOOL_TRANS_COMP_TO_OX
            self.compositionally_variable_subset_raw = DEFAULT_COMP_VAR_SUBSET
            return"""

        self.Elkeys = ml_indexer.Elkeys
        self.label_indices = ml_indexer.label_indices
        self.label_indices_comp = ml_indexer.label_indices_comp
        self.detail_label_indices = ml_indexer.detail_label_indices
        self.compToOx_raw = ml_indexer.compToOx
        self.oxToEl_raw = getattr(ml_indexer, "OxToEl", None) 
        self.MM_raw = ml_indexer.MM
        self.Minv_raw = ml_indexer.Minv
        self.Mtot_raw = ml_indexer.Mtot
        self.phaseToCompMap_raw = ml_indexer.phaseToCompMap
        self.variedToAllComp_raw = ml_indexer.variedToAllComp
        self.fixed_phaseToCompMap_raw = ml_indexer.fixed_phaseToCompMap
        self.boolTransCompToOx_raw = ml_indexer.boolTransCompToOx
        self.compositionally_variable_subset_raw = ml_indexer.compositionally_variable_subset


class MidLevelNetwork(TunableModel):
    def __init__(self, encoderLayerUp=0, encoderLayerDown=0,
                 middleLayerUp=0, middleLayerDown=0,
                 low_regularization='none', high_regularization='none', 
                 activation_leak=0.05,
                 lowWD = 0, # Weight Decays for use when training lower and upper model respectively 
                 highWD = 0,
                 noise = 0,
                 description='',
                 ml_indexer=None,
                 device = torch.device('cpu')):
        # call TunableModel constructor
        super().__init__(
            encoderLayerUp,
            encoderLayerDown,
            middleLayerUp,
            middleLayerDown,
            low_regularization,
            high_regularization,
            activation_leak,
            ml_indexer=ml_indexer,
        )

        # Save Model Configuration:
        self.config = dict(
            encoderLayerUp=encoderLayerUp, 
            encoderLayerDown=encoderLayerDown,
            middleLayerUp=middleLayerUp, 
            middleLayerDown=middleLayerDown,
            low_regularization=low_regularization, 
            high_regularization=high_regularization, 
            activation_leak=activation_leak,
            lowWD=lowWD,
            highWD=highWD,
            noise=noise,
            description=description)

        self.encoderLayerUp=encoderLayerUp
        self.encoderLayerDown=encoderLayerDown
        self.middleLayerUp=middleLayerUp
        self.middleLayerDown=middleLayerDown
        self.low_regularization=low_regularization 
        self.high_regularization=high_regularization 
        self.activation_leak=activation_leak
        self.lowWD=lowWD
        self.highWD=highWD
        self.noise = noise 
        self.description=description

        # finish the build steps that rely on global label_indices etc.
        self.finish_build()

        # Build comp_mappings matrix (rows = chem-heads, cols = total components used by variable phases)
        # comp_mappingsL maps each component-column to a chem-head index; we turn into a dense one-hot mapping
        num_cols = len(self.comp_mappingsL)
        num_rows = self._n_chem_head_count
        comp_mappings = torch.zeros((num_rows, num_cols), dtype=torch.float32)
        for col_idx, row_idx in enumerate(self.comp_mappingsL):
            comp_mappings[int(row_idx), int(col_idx)] = 1.0

        # store buffers; these names match your existing code expectations
        self.register_buffer('comp_mappings', comp_mappings)
        self.register_buffer('comp_binaries', torch.tensor(self.comp_binariesL, dtype=torch.long))

        # pure_binaries_bool vector (length = n_phases) where True indicates pure binary (no chem head)
        pure_bool = torch.ones(self.n_phases, dtype=torch.bool)
        if len(self.comp_binariesL) > 0:
            pure_bool[self.comp_binariesL] = False

        self.compToOx = torch.tensor(self.compToOx_raw, dtype=torch.float, device=device)
        self.oxToEl = torch.tensor(self.oxToEl_raw, dtype=torch.float, device=device)
        self.elToOx = torch.linalg.inv(self.oxToEl[:len(self.Elkeys)]) # For FeOt only
        self.Minv = torch.tensor(self.Minv_raw, dtype=torch.float, device=device)
        self.MM = torch.tensor(self.MM_raw, dtype=torch.float, device=device)
        self.Mtot = torch.tensor(self.Mtot_raw, dtype=torch.float, device=device).flatten()

        self.register_buffer('boolTransCompToEl', torch.tensor(self.boolTransCompToOx_raw @ self.oxToEl_raw, dtype=torch.int))
        self.register_buffer('compositionally_variable_subset', torch.tensor(self.compositionally_variable_subset_raw, dtype=int))
        self.register_buffer('phaseToCompMap', torch.tensor(self.phaseToCompMap_raw, dtype=torch.float))
        self.register_buffer('variedToAllComp', torch.tensor(self.variedToAllComp_raw, dtype=torch.float))
        self.register_buffer('fixed_phaseToCompMap', torch.tensor(self.fixed_phaseToCompMap_raw, dtype=torch.float))
        self.register_buffer('compToEl', torch.tensor(self.compToOx_raw @ self.oxToEl_raw, dtype=torch.float))
        self.register_buffer('molar_epsilon', torch.tensor(self.ml_indexer.molar_epsilon, dtype=torch.float32))
        # print("Registered molar_epsilon buffer with value: {}".format(self.molar_epsilon.item()))

    def save(self, DictFilePath, config_yaml=None, training_yaml=None, processing_yaml=None, stats=None, log_text=None):
        """
        Save model as zip package with complete metadata for deployment and checkpointing.
        
        Creates a zip file (.pt format for compatibility) containing:
        - state_dict.pt: PyTorch state_dict() binary
        - config.json: Model configuration (architecture hyperparameters)
        - ml_indexer/: Directory with saved ml_indexer state (metadata.json, structure.json, arrays.npz)
        - model.yaml: YAML used to train this model
        - training.yaml: Original training data bundle YAML (if provided)
        - stats.txt: Post-filtering statistics from training data (if provided)
        - log.txt: Training/tuning log (if provided)
        
        Parameters
        ----------
        DictFilePath : str or Path
            Full path to save zip file. Should end in .pt for compatibility.
        config_yaml : str, optional
            YAML string of model configuration used during training
        training_yaml : str, optional
            YAML string of training data bundle configuration
        stats : str, optional
            Post-filtering statistics from training data
        log_text : str, optional
            Complete log from training and optional tuning operations
        """
        DictFilePath = Path(DictFilePath)
        if DictFilePath.suffix == "":
            DictFilePath = DictFilePath.with_suffix(".pt")
        DictFilePath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory for zip contents
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # === 1. Save state_dict as .pt ===
            state_dict_path = temp_path / 'state_dict.pt'
            torch.save(self.state_dict(), state_dict_path)
            
            # === 2. Save config as JSON ===
            config_path = temp_path / 'config.json'
            # Apply type conversions for JSON safety: convert matching keys to target types,
            # and convert any unmatched values to strings to avoid JSON serialization issues
            safe_config = apply_type_conversions(self.config, TYPE_CONVERSION_MAP, default_dtype=str)
            with open(config_path, 'w') as f:
                json.dump(safe_config, f, indent=2)
            
            # === 3. Save ml_indexer state ===
            indexer_dir = temp_path / 'ml_indexer'
            self.ml_indexer.save(str(indexer_dir))
            
            # === 4. Save metadata about when this was saved ===
            metadata = {
                'saved_at': datetime.now().isoformat(),
                'torch_version': torch.__version__,
                'numpy_version': np.__version__,
            }
            metadata_path = temp_path / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # === 5. Save optional YAML and logs ===
            if config_yaml is not None:
                with open(temp_path / 'model.yaml', 'w') as f:
                    f.write(config_yaml)
            
            if processing_yaml is not None:
                with open(temp_path / 'data_processing.yaml', 'w') as f:
                    f.write(processing_yaml)
            
            if training_yaml is not None:
                with open(temp_path / 'training.yaml', 'w') as f:
                    f.write(training_yaml)

            if stats is not None:
                with open(temp_path / 'stats.txt', 'w') as f:
                    f.write(stats)
            
            if log_text is not None:
                with open(temp_path / 'log.txt', 'w') as f:
                    f.write(log_text)
            
            # === 6. Create zip archive ===
            # Use zipfile instead of gzip for better compatibility and faster access
            with zipfile.ZipFile(DictFilePath, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in temp_path.glob('**/*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(temp_path)
                        zf.write(file_path, arcname=arcname)


    def forward_binaries(self, x):
        """Outputs satuation logits only, to be passed through sigmoid. Useful for training with BCEwithlogits loss"""
        # Encode features
        latent = self.encoder(x)

        # Apply each head to the shared latent vector
        outputs = []
        for head in self.sat_head:
            out = head(latent)  # shape: (batch_size, 1)
            outputs.append(out)

        return torch.cat(outputs, dim=1)


    def forward_phase_moles(self, latentx, binary_mask, intensiveComponents, details_out = False):
        """Predicts molar abundance of phases and reconstructs bulk composition."""
        logMoles = (self.mole_head(latentx) * binary_mask) # Apply binary mask to zero out non-present phases        eps = logMoles.new_tensor(self.ml_indexer.molar_epsilon)
        eps = self.molar_epsilon #* torch.ones_like(logMoles, dtype=logMoles.dtype, device=logMoles.device) # Use registered buffer for molar epsilon
        if eps:
            #print("Applying molar epsilon in forward_phase_moles: {}".format(eps.item()))
            phaseMoles = (torch.exp(logMoles * torch.log(torch.tensor(10, device=logMoles.device, dtype=logMoles.dtype))) - eps) * binary_mask #  invert log transform; add small epsilon to assert lower bound for the outputs
        else:
            phaseMoles = logMoles
            #print("No molar epsilon applied in forward_phase_moles; using raw mole head outputs as phase moles.")
        compMultipliers = phaseMoles @ self.phaseToCompMap #(B,C)
        intensivePhaseProportions = intensiveComponents @ self.variedToAllComp #BV, VC -> BC #NEED TO GET BINARIES AND PROPORTIONS TOGETHER IN COMPONENT FORM, RECREATE PHASETOCOMP (B,P,C).vASK IF INDEXING TO BUILD IS THE MOST EFFICIENT WAY
        phaseProportions = intensivePhaseProportions + self.fixed_phaseToCompMap # How to project? BC + 1C -> BC. Get ones where all pure phase components are
        componentMoles = phaseProportions * compMultipliers
        #print(f"componentMoles NaNs in forward_phase_moles base function: {torch.isnan(componentMoles).sum()}")
        reconBulkUnNormed = componentMoles @ self.compToEl #(B,E)
        totals = reconBulkUnNormed.sum(dim=1)#torch.ones(reconBulkUnNormed.size()[0], device = 'cuda')   #(B) TEMP NO NORMALIZATION, LET LINEAR ALGEBRA TAKE CARE OF IT
        #print(f"Normalization Totals is nan: {torch.isnan(totals).sum()}, is zero: {(totals == 0).sum()}")
        reconBulk = reconBulkUnNormed / totals.unsqueeze(-1).clamp(min=1e-6) 

        if details_out:
            return logMoles, reconBulk, componentMoles / totals.unsqueeze(-1), phaseProportions, phaseMoles # Apply identical normalization to components for equality
        else:
            return logMoles, reconBulk

    """def polish_negative_px(self, phaseProportions):
        # Check and correct for below zero CaO
        PosS = torch.sum(phaseProportions[:,label_indices_comp['orthopyroxene'][:5]], dim = -1) # Sumterm
        NegS = (phaseProportions[:,label_indices_comp['orthopyroxene'][5]] * 2) + phaseProportions[:,label_indices_comp['orthopyroxene'][6]]
        illegal = NegS >  PosS 
        if illegal.sum(): # Calculate scaling factors to set CaO to 0
            denom = ( (2 * phaseProportions[illegal, label_indices_comp['orthopyroxene'][6]]) + (3 * phaseProportions[illegal, label_indices_comp['orthopyroxene'][5]]) )
            b = 1 / denom
            a = NegS[illegal] / ( denom * PosS[illegal] )
            #Scale Illegal Values
            phaseProportions[illegal, label_indices_comp['orthopyroxene'][:5]] = a * phaseProportions[illegal, label_indices_comp['orthopyroxene'][:5]]
            phaseProportions[illegal, label_indices_comp['orthopyroxene'][-2:]] = b * phaseProportions[illegal, label_indices_comp['orthopyroxene'][-2:]]
        return phaseProportions"""
    
    def polish_negative_px(self, intensiveComponents):
        opx = self.detail_label_indices['orthopyroxene']
        opx_jadeite = opx['jadeite']
        opx_essenite = opx['essenite']
        opx_pos_idxs = [
            opx['diopside'],
            opx['clinoenstatite'],
            opx['hedenbergite'],
            opx['alumino-buffonite'],
            opx['buffonite'],
        ]
        # Send g5 Fliers back to reason, rescale remaining components.
        fliers = intensiveComponents[:, opx_jadeite] > 0.5
        intensiveComponents[fliers, opx_jadeite] = 0.5
        r_fly, c_fly = torch.meshgrid(
            torch.nonzero(fliers, as_tuple=False).squeeze(-1).to(torch.int),
            torch.tensor(opx_pos_idxs + [opx_essenite], dtype=torch.int, device=intensiveComponents.device),
            indexing="ij",
        )
        intensiveComponents[r_fly, c_fly] =intensiveComponents[r_fly, c_fly] * (0.5 / (intensiveComponents[r_fly, c_fly].sum(dim=-1, keepdim=True)))

        # Check and correct for below zero CaO
        PosS = torch.sum(
            intensiveComponents[:, opx_pos_idxs], dim=-1
        )  # Sumterm
        NegS = (
            intensiveComponents[:, opx_essenite] * 2
        ) + intensiveComponents[:, opx_jadeite]

        illegal = NegS > PosS

        if illegal.any():
            #print(PosS[illegal])
            #print(NegS[illegal])
            #print(torch.where(illegal))
            denom = (
                2 * intensiveComponents[illegal, opx_jadeite]
                + 3 * intensiveComponents[illegal, opx_essenite]
            )
            b = 1.0 / denom
            a = NegS[illegal] / (denom * PosS[illegal])

            # Rows (batch indices) where constraint is violated
            row_idx = torch.nonzero(illegal, as_tuple=False).squeeze(-1).to(torch.int)
            #print(row_idx)
            # Column indices (orthopyroxene subset)
            cols_pos = torch.tensor(
                opx_pos_idxs, device=intensiveComponents.device, dtype=torch.int
            )
            cols_neg = torch.tensor(
                [opx_essenite, opx_jadeite], device=intensiveComponents.device, dtype=torch.int
            )

            # Build broadcastable index grids
            rr_pos, cc_pos = torch.meshgrid(row_idx, cols_pos, indexing="ij")
            rr_neg, cc_neg = torch.meshgrid(row_idx, cols_neg, indexing="ij")
            #print(intensiveComponents[rr_pos, cc_pos])
            #print(intensiveComponents[rr_neg, cc_neg])

            # Scale updates
            intensiveComponents[rr_pos, cc_pos] = a[:, None] * intensiveComponents[rr_pos, cc_pos]
            intensiveComponents[rr_neg, cc_neg] = b[:, None] * intensiveComponents[rr_neg, cc_neg]

        return intensiveComponents
    
    def polish_negative_spFe(self, intensiveComponents):
        sp = self.detail_label_indices['spinel']
        sp_chromite = sp['chromite'] if 'chromite' in sp else None
        sp_hercynite = sp['hercynite']
        sp_magnetite = sp['magnetite']
        sp_spinel = sp['spinel']
        sp_ulvospinel = sp['ulvospinel']
        # Check and correct for below zero FeO
        pos_idxs = [sp_hercynite, sp_magnetite, sp_ulvospinel]
        if 'chromite' in sp:
            pos_idxs = [sp_chromite] + pos_idxs
        PosS = torch.sum(intensiveComponents[:, pos_idxs], dim=-1) + (
            intensiveComponents[:, sp_ulvospinel] * 2.25
        )
        NegS = intensiveComponents[:, sp_spinel] * 19

        illegal = NegS > PosS
        if illegal.any():
            # First form indexers
            # Rows (batch indices) where constraint is violated
            row_idx = torch.nonzero(illegal, as_tuple=False).squeeze(-1).to(torch.int)
            #print(row_idx)
            # Column indices (spinel subset)
            cols_pos = torch.tensor(pos_idxs, device=intensiveComponents.device, dtype=torch.int)
            cols_neg = torch.tensor(sp_spinel, device=intensiveComponents.device, dtype=torch.int)

            # Build broadcastable index grids
            rr_pos, cc_pos = torch.meshgrid(row_idx, cols_pos, indexing="ij")
            rr_neg, cc_neg = torch.meshgrid(row_idx, cols_neg, indexing="ij")
            A = torch.sum(intensiveComponents[rr_pos, cc_pos], dim=-1)
            #print(PosS[illegal])
            #print(NegS[illegal.unsqueeze(-1)])
            #print(torch.where(illegal))
            denom = (19*A) + PosS[illegal] 
            a = 19.0 / denom
            b = PosS[illegal] / (denom * intensiveComponents[illegal, sp_spinel])

            # Scale updates
            intensiveComponents[rr_pos, cc_pos] = a[:, None] * intensiveComponents[rr_pos, cc_pos]
            intensiveComponents[rr_neg, cc_neg] = b[:, None] * intensiveComponents[rr_neg, cc_neg]

        return intensiveComponents
    
    def polish_negative_spAl(self, intensiveComponents):
        sp = self.detail_label_indices['spinel']
        sp_hercynite = sp['hercynite']
        sp_magnetite = sp['magnetite']
        sp_spinel = sp['spinel']
        sp_ulvospinel = sp['ulvospinel']
        # Check and correct for negative Al in spinel
        NegS = (intensiveComponents[:, sp_magnetite] * (2/3)) + (
            intensiveComponents[:, sp_ulvospinel] * (1/4)
        )
        PosS = intensiveComponents[:, sp_hercynite] + intensiveComponents[:, sp_spinel]
        illegal = NegS > PosS
        if illegal.sum():  # Calculate scaling factors to zero out negative hercynite without affecting Chromite
            A = intensiveComponents[illegal, sp_magnetite] + intensiveComponents[illegal, sp_ulvospinel]
            RemS = A + PosS[illegal] # equivalent to 1-chromite
            a = RemS / (A + NegS[illegal])
            b = (RemS * NegS[illegal]) / ( PosS[illegal] * (A + NegS[illegal]) )
            intensiveComponents[illegal, sp_magnetite], intensiveComponents[illegal, sp_ulvospinel] = (
                a * intensiveComponents[illegal, sp_magnetite],
                a * intensiveComponents[illegal, sp_ulvospinel],
            )
            intensiveComponents[illegal, sp_hercynite], intensiveComponents[illegal, sp_spinel] = (
                b * intensiveComponents[illegal, sp_hercynite],
                b * intensiveComponents[illegal, sp_spinel],
            )
        return intensiveComponents
    
    def polish_negative_sp(self, intensiveComponents, trial = 0):
        # Indices for spinel components
        sp = self.detail_label_indices['spinel']
        i2 = sp['hercynite']
        i3 = sp['magnetite']
        i4 = sp['spinel']
        i5 = sp['ulvospinel']

        # Extract components
        if 'chromitte' in sp:
            i1 = sp['chromite']
            c1 = intensiveComponents[:, i1]
        else:
            i1 = None
            c1 = torch.zeros_like(intensiveComponents[:, i2])
        c2 = intensiveComponents[:, i2]
        c3 = intensiveComponents[:, i3]
        c4 = intensiveComponents[:, i4]
        c5 = intensiveComponents[:, i5]

        # Constraint 1
        pos1 = c1 + c2 + c3 + 2.25 * c5
        neg1 = 19 * c4
        illegal1 = neg1 > pos1

        # Constraint 2
        pos2 = c2 + c4
        neg2 = (2/3) * c3 + 0.25 * c5
        illegal2 = neg2 > pos2

        # Rows that violate any constraint
        illegal = illegal1 | illegal2
        if illegal.any():
            row_idx = torch.nonzero(illegal, as_tuple=False).squeeze(-1).to(torch.int)

            # Define groups
            if i1 is not None:
                cols_a = torch.tensor([i1, i2], device=intensiveComponents.device, dtype=torch.int)
            else:
                cols_a = torch.tensor([i2], device=intensiveComponents.device, dtype=torch.int)
            cols_b = torch.tensor([i3, i5], device=intensiveComponents.device, dtype=torch.int)
            cols_c = torch.tensor([i4], device=intensiveComponents.device, dtype=torch.int)

            rr_a, cc_a = torch.meshgrid(row_idx, cols_a, indexing="ij")
            rr_b, cc_b = torch.meshgrid(row_idx, cols_b, indexing="ij")
            rr_c, cc_c = torch.meshgrid(row_idx, cols_c, indexing="ij")

            # Extract per-row values
            if i1 is not None:
                A = intensiveComponents[row_idx][:, [i1, i2]].sum(dim=-1)  # sum(c1,c2)
            else:
                A = intensiveComponents[row_idx][:, i2]
            B = intensiveComponents[row_idx][:, [i3, i5]].sum(dim=-1)  # sum(c3,c5)
            C = intensiveComponents[row_idx][:, i4]                    # c4

            # Terms for constraints
            if i1 is not None:
                L1_c1c2 = intensiveComponents[row_idx][:, [i1, i2]].sum(dim=-1)
            else:
                L1_c1c2 = intensiveComponents[row_idx][:, i2]
            L1_c3c5 = intensiveComponents[row_idx][:, i3] + 2.25 * intensiveComponents[row_idx][:, i5]

            L2_c2 = intensiveComponents[row_idx][:, i2]
            L2_c3c5 = (2/3) * intensiveComponents[row_idx][:, i3] + 0.25 * intensiveComponents[row_idx][:, i5]

            # Build coefficient matrices (batch, 3, 3)
            # Order of unknowns: [a, b, c]
            M = torch.zeros((row_idx.shape[0], 3, 3), device=intensiveComponents.device, dtype=intensiveComponents.dtype)
            rhs = torch.zeros((row_idx.shape[0], 3), device=intensiveComponents.device, dtype=intensiveComponents.dtype)

            # Equation 1: normalization
            M[:, 0, 0] = A
            M[:, 0, 1] = B
            M[:, 0, 2] = C
            rhs[:, 0] = A + B + C

            # Equation 2: 19 cC = a*(c1+c2) + b*(c3+2.25c5)
            M[:, 1, 0] = L1_c1c2
            M[:, 1, 1] = L1_c3c5
            M[:, 1, 2] = -19 * C
            rhs[:, 1] = 0.0

            # Equation 3: a*c2 + c*C = (2/3)b*c3 + (1/4)b*c5
            M[:, 2, 0] = L2_c2
            M[:, 2, 1] = -L2_c3c5
            M[:, 2, 2] = C
            rhs[:, 2] = 0.0

            # Solve batch of linear systems
            try:
                sol = torch.linalg.solve(M, rhs)  # shape (rows, 3). A wannabe pure MgAl2O3 makes a singular matrix. Handle this edge case with a simpler fix then recursion.
            except:
                lIDX = [i2, i3, i4, i5]
                if i1 is not None:
                    lIDX = [i1] + lIDX
                rr_e, cc_e = torch.meshgrid(
                    row_idx,
                    torch.tensor(
                        lIDX,
                        dtype=torch.int,
                        device=intensiveComponents.device,
                    ),
                    indexing="ij",
                )
                #print('SPINEL COMPOSITIONS:')
                #print(intensiveComponents[rr_e, cc_e])
                if trial == 2:
                    raise ValueError('Negative Spinel Solve failed! Singular Matrix?')
                intensiveComponents = self.polish_negative_spFe(intensiveComponents)
                return  self.polish_negative_sp(intensiveComponents, trial = trial+1) # Retry after FeO solve
            
            a = sol[:, 0]
            b = sol[:, 1]
            c = sol[:, 2]

            # Apply scaling
            intensiveComponents[rr_a, cc_a] = a[:, None] * intensiveComponents[rr_a, cc_a]
            intensiveComponents[rr_b, cc_b] = b[:, None] * intensiveComponents[rr_b, cc_b]
            intensiveComponents[rr_c, cc_c] = c[:, None] * intensiveComponents[rr_c, cc_c]

        return intensiveComponents

    

    ### INCORPERATE MIDDLE LAYER
    def forward(self, x, binaries=None, detailed = False, NN_only = False):
        """ 
        Forward pass for both training and inference.

        Args:
            x (Tensor): Input system representation [batch_size, input_dim]
            binaries (Tensor or None): If provided, used as ground-truth saturation labels.
                                       If None, saturation predictions are used.

        Returns:
            Tuple of (saturation logits or likelihoods, masked chemistry predictions)
        """
        # Encode features
        latent = self.encoder(x)

        # Build mask to exclude components that require elements not in inputs
        inf_mask = ((x[:,len(self.ml_indexer.featureNames):] == 0).to(torch.float32) @ self.boolTransCompToEl[self.compositionally_variable_subset].T.to(torch.float32)) != 0 #be,ec->bc 


        # Phase saturation logits (not yet sigmoid)
        sat_outputs = [head(latent) for head in self.sat_head]
        logits = torch.cat(sat_outputs, dim=1)
        #logits = self.sat_head(latent)

        """likelihoods = torch.sigmoid(logits) # Testing superliquidus no-compute 10/10/25
        binary_pred = (likelihoods > 0.5).float()

        if binaries is None:
            # Inference mode — use predicted binaries
            binary_inp = binary_pred

            
        else:
            # Training mode — use provided ground truth binaries
            binary_inp = binaries

        # Construct masking matrix for chemistry predictions
        zero_mask = binary_inp[:, self.comp_binaries] @ self.comp_mappings  # [batch, n_components]

        if binaries is None: #Inference
            chem_outputs = [head(latent, inf_mask = inf_mask[:,(self.comp_mappings[i]).to(torch.bool)]) for i, head in enumerate(self.chem_heads)]
            chem_out = torch.cat(chem_outputs, dim=1) 
            if not NN_only:
                begin_refit = time.time()
                chem_out = self.polish_negative_px(chem_out)
                begin_spinel = time.time()
                chem_out = self.polish_negative_sp(chem_out)
                # print(f"To solve 0CaO Px: {round((begin_spinel-begin_refit)*1E6)} microsec; To solve 0FeO/Al2O3 Sp: {round((time.time()-begin_spinel)*1E6)} microsec ")

            
        else: #Training
            chem_outputs = [head(latent, train_inf_mask = inf_mask[:,(self.comp_mappings[i]).to(torch.bool)]) for i, head in enumerate(self.chem_heads)]
            chem_out = torch.cat(chem_outputs, dim=1) 
        
        phaseMass, reconBulk, componentMoles, phaseProportions = self.forward_phase_moles(latent, binary_mask=binary_pred.detach(), intensiveComponents=chem_out, details_out=True)

        if binaries is None:
            if detailed:
                return likelihoods, chem_out*zero_mask, phaseMass, reconBulk, componentMoles, phaseProportions
            else:
                return likelihoods, chem_out*zero_mask, phaseMass, reconBulk # Inference
        else:
            return logits, chem_out*zero_mask, zero_mask, phaseMass, reconBulk # Training, return zero mask for loss masking of intensive chemistries"""

        likelihoods = torch.sigmoid(logits)
        binary_pred = (likelihoods > 0.5).float()


        # For MELTS only Identify superliquidus rows: only one head > 0.5
        if 'melts-liquid' in self.label_indices_comp:
            superliquidus = (binary_pred.sum(dim=1) == 1) #& (binary_pred[:, -1] == 1) # ASSUMES MELTS IS THE ONLY SINGLE PHASE. I've never seen MELTS report anything monomineralic, though it is possible.
        else: # HeFESTo / subsolidus only, no superliquidus condition
            superliquidus = torch.zeros(binary_pred.size(0), dtype=torch.bool) # Zeros of len batch size
        non_super = ~superliquidus
        force_count = 0

        if binaries is None: # For inference, force saturation of phases to explain bulk composition to stabilize linear algebra. 
            force_phases = True
            while force_phases: # Iterate until no more phases to force saturation for. Should be 1-3 iterations at most.
                present_oxides = (binary_pred @ self.phaseToCompMap) @ self.compToEl 
                unexplained_oxides = ((x[:, len(self.ml_indexer.featureNames):]==0).to(torch.float32) + present_oxides) == 0
                if unexplained_oxides.any():
                    force_count += 1
                    #assert force_count <= 3, "Phase forcing did not satisfy mass balance even after 3 iterations"
                    if force_count > 3:
                        # print("Phase forcing did not satisfy mass balance even after 3 iterations")
                        break
                    unexplained_rows = torch.sum(unexplained_oxides, dim=1) > 0
                    unexplained_columns = torch.sum(unexplained_oxides, dim=0) 
                    #print(f"Unexplained columns: {unexplained_columns}")
                    #for col_idx in torch.where(unexplained_columns)[0]:
                        #oxide_name = self.ml_indexer.Oxides[col_idx]
                        #print(f"Unexplained oxide: {oxide_name} in {unexplained_rows.sum()} samples of {unexplained_rows.size(0)}.")
                    #print(f"Unexplained oxides: {unexplained_oxides.sum()} across {unexplained_rows.sum()} samples. Forcing saturation of one additional phase for these samples.")
                    phases_to_force = (unexplained_oxides.to(torch.float32) @ self.compToEl.T) @ self.phaseToCompMap.T > 0 # B x P
                    force_idx = torch.argmax((likelihoods*phases_to_force)[unexplained_rows], dim=1)
                    row_idx = torch.nonzero(unexplained_rows, as_tuple=False).to(torch.long)
                    binary_pred[row_idx, force_idx] = 1.0
                else: 
                    force_phases = False

            binary_inp = binary_pred

        else:
            binary_inp = binaries




        features = x  # alias for clarity

        if self.middleBrain is not None: # If there is a middle encoder, use it. Otherwise prepare first encodings and phase saturation
            CoreOutput = self.middleBrain(torch.cat([latent, logits, features], dim=1))
        else: 
            CoreOutput = torch.cat([latent, logits, features], dim=1)

        zero_mask = binary_inp[:, self.comp_binaries] @ self.comp_mappings  # [batch, n_components]

        if binaries is None:  # Inference
            chem_outputs = [
                head(CoreOutput, inf_mask=inf_mask[:, (self.comp_mappings[i]).to(torch.bool)])
                for i, head in enumerate(self.chem_heads)
            ]
            chem_out = torch.cat(chem_outputs, dim=1)

            # For liquid models only: Treat trivial case where liquid is the bulk composition when it is the only phase
            if 'melts-liquid' in self.label_indices_comp:
                chem_out[superliquidus] = 0.0 # Zero out superliquidus rows

                # Overwrite liquid component columns with feature composition
                liq_idx = torch.tensor(self.label_indices_comp['melts-liquid'], device=chem_out.device)
                #chem_out[superliquidus][:, liq_idx] = features[superliquidus, 3:]
                row_idx = torch.nonzero(superliquidus, as_tuple=False).squeeze(-1).to(torch.long)
                rr_liq, cc_liq = torch.meshgrid(row_idx, liq_idx.to(torch.long), indexing='ij')
                chem_out[rr_liq, cc_liq] = features[superliquidus, len(self.ml_indexer.featureNames):] # Assumes same element ordering in liquid and features

                # print(f"NANs in chem_out after superliquidus assignment: {torch.isnan(chem_out).sum()}")

                if not NN_only and non_super.any() and 'melts-liquid' in self.label_indices: # Only do this for MELTS: Not HeFESTo, which lacks negative components. 
                    chem_out[non_super] = self.polish_negative_px(chem_out[non_super])
                    chem_out[non_super] = self.polish_negative_sp(chem_out[non_super])

        else:  # Training
            chem_outputs = [
                head(CoreOutput, train_inf_mask=inf_mask[:, (self.comp_mappings[i]).to(torch.bool)])
                for i, head in enumerate(self.chem_heads)
            ]
            chem_out = torch.cat(chem_outputs, dim=1)

            if 'melts-liquid' in self.label_indices_comp:
                # Overwrite liquid component columns with feature composition
                liq_idx = torch.tensor(self.label_indices_comp['melts-liquid'], device=chem_out.device)
                row_idx = torch.nonzero(superliquidus, as_tuple=False).squeeze(-1).to(torch.long)
                rr_liq, cc_liq = torch.meshgrid(row_idx, liq_idx.to(torch.long), indexing='ij')
                chem_out[rr_liq, cc_liq] = features[superliquidus, len(self.ml_indexer.featureNames):] # Assumes same element ordering in liquid and features


        # Compute phase properties
        logMoles, reconBulk, componentMoles, phaseProportions, phaseMoles = self.forward_phase_moles(
            CoreOutput, binary_mask=binary_pred.detach(), intensiveComponents=chem_out, details_out=True
        )

        if 'melts-liquid' in self.label_indices_comp:  # Assign direct values for superliquidus rows
            liq_idx_phase = torch.tensor(self.label_indices['melts-liquid'], device=chem_out.device)

            reconBulk[superliquidus] = features[superliquidus, len(self.ml_indexer.featureNames):]
            componentMoles[superliquidus][:, liq_idx_phase] = features[superliquidus, len(self.ml_indexer.featureNames):]
            phaseProportions[superliquidus][:, liq_idx_phase] = features[superliquidus, len(self.ml_indexer.featureNames):]
            reconBulk[superliquidus] = features[superliquidus, len(self.ml_indexer.featureNames):]
            phaseMoles[superliquidus, self.ml_indexer.mass_phasedict['melts-liquid']] = 1.0

        # Set logMoles such that phaseMoles = 1.0 when inverted: log10(1 + eps)
        if self.molar_epsilon:
            #print("Applying molar epsilon for superliquidus logMoles assignment: {}".format(self.molar_epsilon.item()))
            log_ten = torch.log(torch.tensor(10.0, device=logMoles.device, dtype=logMoles.dtype))
            target_logMoles = torch.log(1.0 + self.molar_epsilon) / log_ten

            if 'melts-liquid' in self.label_indices_comp:
                logMoles[superliquidus, self.ml_indexer.mass_phasedict['melts-liquid']] = target_logMoles
        else:
            logMoles = phaseMoles



        if binaries is None:
            if detailed:
                return likelihoods, chem_out*zero_mask, logMoles, reconBulk, componentMoles, phaseProportions, phaseMoles # Inference with residual fitting
            else:
                return likelihoods, chem_out*zero_mask, logMoles, reconBulk # Inference
        else:
            return logits, chem_out*zero_mask, zero_mask, logMoles, reconBulk # Training, return zero mask for loss masking of intensive chemistries"""


def load_model_from_zip(zip_path, substitutions=None, low_only=False, epsilon = None, load_prefixes=None):
    """
    Load MidLevelNetwork from zip package created by MidLevelNetwork.save().
    
    Reconstructs model architecture, loads weights, and restores ml_indexer state
    without requiring access to original data or CSV projection matrices.
    
    Parameters
    ----------
    zip_path : str or Path
        Path to model zip file (.pt extension)
    substitutions : dict, optional
        Architecture parameter overrides. Applied after loading config.
        Useful for building upper model on lower model weights.
    low_only : bool, default=False
        If True, only load encoder and saturation heads (lower model components).
        Useful for warm-starting upper model training.
    
    Returns
    -------
    MidLevelNetwork
        Reconstructed model with loaded weights, config, and ml_indexer state.
    
    Raises
    ------
    FileNotFoundError
        If zip file or required contents are missing.
    """
    from ..config.ml_indexer import load_ml_indexer_from_state
    
    zip_path = Path(zip_path)
    
    # Extract to temporary directory and load
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Extract zip contents
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_path)
        
        # === Load config ===
        with open(temp_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        # === Apply substitutions ===
        if substitutions is not None:
            for parameter, setting in substitutions.items():
                config[parameter] = setting
        
        # === Load ml_indexer ===
        ml_indexer = load_ml_indexer_from_state(temp_path / 'ml_indexer')
        if epsilon is not None:
            ml_indexer.molar_epsilon = epsilon
        config['ml_indexer'] = ml_indexer
        
        # print(f"ml_indexer.molar_epsilon = {ml_indexer.molar_epsilon}")
        # === Create model with loaded config ===
        model = MidLevelNetwork(**config)
        
        # === Load state_dict ===
        state_dict_path = temp_path / 'state_dict.pt'
        saved_state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=False)
        
        if low_only or load_prefixes is not None:
            effective_prefixes = load_prefixes if load_prefixes is not None else ["encoder.", "sat_head."]
            _load_matching_state_dict(model, saved_state_dict, load_prefixes=effective_prefixes)
        else:
            # Load full model
            model.load_state_dict(saved_state_dict, strict=False)

        model.molar_epsilon.fill_(float(model.ml_indexer.molar_epsilon))
    
    return model


def _load_matching_state_dict(model, saved_state_dict, load_prefixes=None):
    model_dict = model.state_dict()
    filtered_dict = {}
    for key, value in saved_state_dict.items():
        if load_prefixes is not None and not any(key.startswith(prefix) for prefix in load_prefixes):
            continue
        if key not in model_dict:
            continue
        if model_dict[key].shape != value.shape:
            continue
        filtered_dict[key] = value

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)


def rebuild_MELTS_model(DictFilePath, substitutions=None, low_only=False, ml_indexer=None, epsilon = None, load_prefixes=None):
    """
    Load MELTS NN model from checkpoint file.
    
    Handles both legacy single-file format (.pt with dict) and new zip format.
    Reconstructs architecture, loads weights, and applies configuration substitutions.
    
    Parameters
    ----------
    DictFilePath : str or Path
        Path to model file (.pt). May be:
        - Legacy single-file format: dict with 'config', 'state_dict', 'ml_indexer' keys
        - New zip format: .pt file containing state_dict.pt, config.json, ml_indexer/, etc.
    substitutions : dict, optional
        Configuration overrides. Applied to architecture parameters.
        Useful for building upper model on trained lower model.
    low_only : bool, default=False
        If True, only load encoder and saturation heads (lower model components).
    ml_indexer : MlIndexer, optional. Necesary if loading legacy format without ml_indexer state. Ignored if ml_indexer state is present in checkpoint.
    Returns
    -------
    MidLevelNetwork
        Reconstructed model with loaded weights and restored state.
    """
    DictFilePath = Path(DictFilePath)
    if DictFilePath.suffix == "":
        #pt_candidate = DictFilePath.with_suffix(".pt")
        zip_candidate = DictFilePath.with_suffix(".zip")
        if zip_candidate.exists():
            DictFilePath = zip_candidate
        #else: 
        #    raise FileNotFoundError(f"No file found at {DictFilePath} with .zip extension.")
    
    # Check extension first: explicit .zip files should use zip loading
    if DictFilePath.suffix == '.zip':
        # print(f"Attempting to load model from .zip file: {DictFilePath}")
        return load_model_from_zip(DictFilePath, substitutions=substitutions, low_only=low_only, epsilon=epsilon, load_prefixes=load_prefixes)
    
    # print(f"Attempting to load model from .pt file: {DictFilePath}")

    # Try legacy format first, fall back to zip if it's actually a zip file
    try:
        ckpt = torch.load(DictFilePath, map_location='cpu', weights_only=False)
        configuration = ckpt['config']
        
        if substitutions is not None:
            for parameter, setting in substitutions.items():
                # print(f"Overriding config parameter '{parameter}' with value: {setting}")
                configuration[parameter] = setting
        
        # Handle both old ml_indexer objects and restore if available
        if 'ml_indexer' in ckpt:
            configuration['ml_indexer'] = ckpt['ml_indexer']
        elif ml_indexer is not None:
            configuration['ml_indexer'] = ml_indexer
        
        if epsilon is not None:
            configuration['ml_indexer'].molar_epsilon = epsilon

        model = MidLevelNetwork(**configuration)
        
        if low_only or load_prefixes is not None:  # Only load selected model components
            effective_prefixes = load_prefixes if load_prefixes is not None else ["encoder.", "sat_head."]
            _load_matching_state_dict(model, ckpt['state_dict'], load_prefixes=effective_prefixes)
        else:
            model.load_state_dict(ckpt['state_dict'], strict=False)

        model.molar_epsilon.fill_(float(model.ml_indexer.molar_epsilon))
        
        return model
    
    except Exception as e:
        # If legacy format failed and file is actually a zip, try zip format
        if zipfile.is_zipfile(DictFilePath):
            # print(f"Legacy format failed, attempting to load as zip format: {e}")
            return load_model_from_zip(DictFilePath, substitutions=substitutions, low_only=low_only, epsilon=epsilon)
        else:
            raise e
