"""
Neural Network Architecture / object.

Contains TunableModel base class and MidLevelNetwork subclass for MELTS emulator.
TunableModel is never used without the MidLevelNetwork. MidLevelNetwork handles the enforcement of physical constraints (e.g. mass balance) constraints during inference
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import utility functions
from ..utils.string_utils import pull_letter, pull_number

# Import constants and mappings from config
from ..config import (
    Elkeys,
    label_indices,
    label_indices_comp,
    compToOx,
    oxToEl,
    MM,
    Minv,
    Mtot,
    phaseToCompMap,
    variedToAllComp,
    fixed_phaseToCompMap,
    boolTransCompToOx,
    compositionally_variable_subset,
)


class TunableModel(nn.Module):
    def __init__(self,
                 encoderLayerUp=1, # Number of layers to expand encodings, feature detection
                 encoderLayerDown=0, # Number of layers to downscale the encodings, feature compression
                 #satLayerDown = 0,
                 middleLayerUp=2,
                 middleLayerDown=1,
                 low_regularization='none',
                 high_regularization='none',
                 activation_leak=0.05):
        """
        regularization: 'none' | 'batchnorm' | 'dropout<frac>' (e.g. 'dropout0.2')
        activation_factory: a callable returning an nn.Module activation (default: LeakyReLU)
        Note: code expects globals: label_indices, Elkeys (and later, mapping buffers in subclass).
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

        self.n_phases = len(list(label_indices.keys()))
        input_dim = 3 + len(Elkeys)
        self.activation_factory = activation_factory
        self.input_dim = input_dim

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
                fraction = pull_number(low_regularization)
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
                fraction = pull_number(high_regularization)
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
        This separation allows use of globals that may be set after __init__ in code.
        """
        # build comp mappings and chem heads
        j = 0
        self.chem_heads = nn.ModuleList()
        for i, (label, inds) in enumerate(label_indices.items()):
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
        mole_layers.append(nn.Softplus())
        self.mole_head = nn.Sequential(*mole_layers)

        # Now build actual chem_heads with correct input_dim = middle_out
        for (n_components, phase_index) in self.chem_list_templates:
            self.chem_heads.append(self._PhaseHead(n_components=n_components, input_dim=middle_out))

        # record some handy tensors/arrays as plain Python lists so subclass can register buffers
        self.comp_mappingsL = self.comp_mappingsL
        self.comp_binariesL = self.comp_binariesL
        self._n_chem_head_count = len(self.chem_heads)


class MidLevelNetwork(TunableModel):
    def __init__(self, encoderLayerUp=0, encoderLayerDown=0,
                 middleLayerUp=0, middleLayerDown=0,
                 low_regularization='none', high_regularization='none', 
                 activation_leak=0.05,
                 lowWD = 0, # Weight Decays for use when training lower and upper model respectively 
                 highWD = 0,
                 noise = 0,
                 description=''):# Use Description Arg to keep track of model's target (e.g. 'MELTS 1.0, Fxtal, NoCr')
        # call TunableModel constructor
        super().__init__(encoderLayerUp, encoderLayerDown, middleLayerUp, middleLayerDown, low_regularization, high_regularization, activation_leak)

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
            noise = 0,
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



        self.compToOx = torch.tensor(compToOx, dtype = torch.float, device = 'cuda')
        self.oxToEl = torch.tensor(oxToEl, dtype = torch.float, device = 'cuda')
        self.elToOx = torch.linalg.inv(self.oxToEl[:len(Elkeys)]) #For FeOt only
        self.Minv = torch.tensor(Minv, dtype = torch.float, device = 'cuda')
        self.MM = torch.tensor(MM, dtype = torch.float, device = 'cuda')
        self.Mtot = torch.tensor(Mtot, dtype = torch.float, device = 'cuda').flatten()

        self.register_buffer('boolTransCompToOx', torch.tensor(boolTransCompToOx)) 
        self.register_buffer('compositionally_variable_subset', torch.tensor(compositionally_variable_subset,dtype = int))
        self.register_buffer('comp_mappings', comp_mappings) 

        
        self.register_buffer('comp_binaries', torch.tensor(comp_binariesL, dtype = torch.int)) # Use == 0 for pure binaries

        self.register_buffer('phaseToCompMap', torch.tensor(phaseToCompMap, dtype = torch.float))
        self.register_buffer('variedToAllComp', torch.tensor(variedToAllComp, dtype = torch.float))
        self.register_buffer('fixed_phaseToCompMap', torch.tensor(fixed_phaseToCompMap, dtype = torch.float))
        self.register_buffer('compToEl', torch.tensor(compToOx @ oxToEl, dtype = torch.float))

    def save(self, DictFilePath):
        torch.save({'state_dict': self.state_dict(), 'config': self.config}, DictFilePath)

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
        """Predicts molar abundance of phases and reconstructs bulk composition. Let intensive components be indexed by label_indices_comp"""

        phaseMoles = self.mole_head(latentx) * binary_mask
        compMultipliers = phaseMoles @ self.phaseToCompMap #(B,C)
        intensivePhaseProportions = intensiveComponents @ self.variedToAllComp #BV, VC -> BC #NEED TO GET BINARIES AND PROPORTIONS TOGETHER IN COMPONENT FORM, RECREATE PHASETOCOMP (B,P,C).vASK IF INDEXING TO BUILD IS THE MOST EFFICIENT WAY
        phaseProportions = intensivePhaseProportions + self.fixed_phaseToCompMap # How to project? BC + 1C -> BC. Get ones where all pure phase components are
        componentMoles = phaseProportions * compMultipliers
        #print(f"componentMoles NaNs in forward_phase_moles base function: {torch.isnan(componentMoles).sum()}")
        reconBulkUnNormed = componentMoles @ self.compToEl #(B,E)
        totals = reconBulkUnNormed.sum(dim=1)#torch.ones(reconBulkUnNormed.size()[0], device = 'cuda')   #(B) TEMP NO NORMALIZATION, LET LINEAR ALGEBRA TAKE CARE OF IT
        #print(f"Normalization Totals is nan: {torch.isnan(totals).sum()}, is zero: {(totals == 0).sum()}")
        reconBulk = reconBulkUnNormed / totals.unsqueeze(-1) # How to project? BE / B1 -> BE

        if details_out:
            return phaseMoles, reconBulk, componentMoles / totals.unsqueeze(-1), phaseProportions # Apply identical normalization to components for equality
        else:
            return phaseMoles, reconBulk

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
        # Send g5 Fliers back to reason, rescale remaining components.
        fliers = intensiveComponents[:, label_indices_comp['orthopyroxene'][6]] > 0.5
        intensiveComponents[fliers, label_indices_comp['orthopyroxene'][6]] = 0.5
        r_fly, c_fly = torch.meshgrid(torch.nonzero(fliers, as_tuple=False).squeeze(-1).to(torch.int), torch.tensor(label_indices_comp['orthopyroxene'][:6],dtype = torch.int, device = intensiveComponents.device), indexing="ij")
        intensiveComponents[r_fly, c_fly] =intensiveComponents[r_fly, c_fly] * (0.5 / (intensiveComponents[r_fly, c_fly].sum(dim=-1, keepdim=True)))

        # Check and correct for below zero CaO
        PosS = torch.sum(
            intensiveComponents[:, label_indices_comp['orthopyroxene'][:5]], dim=-1
        )  # Sumterm
        NegS = (
            intensiveComponents[:, label_indices_comp['orthopyroxene'][5]] * 2
        ) + intensiveComponents[:, label_indices_comp['orthopyroxene'][6]]

        illegal = NegS > PosS

        if illegal.any():
            print(PosS[illegal])
            print(NegS[illegal])
            print(torch.where(illegal))
            denom = (
                2 * intensiveComponents[illegal, label_indices_comp['orthopyroxene'][6]]
                + 3 * intensiveComponents[illegal, label_indices_comp['orthopyroxene'][5]]
            )
            b = 1.0 / denom
            a = NegS[illegal] / (denom * PosS[illegal])

            # Rows (batch indices) where constraint is violated
            row_idx = torch.nonzero(illegal, as_tuple=False).squeeze(-1).to(torch.int)
            print(row_idx)
            # Column indices (orthopyroxene subset)
            cols_pos = torch.tensor(label_indices_comp['orthopyroxene'][:5],
                                    device=intensiveComponents.device, dtype = torch.int)
            cols_neg = torch.tensor(label_indices_comp['orthopyroxene'][-2:],
                                    device=intensiveComponents.device, dtype = torch.int)

            # Build broadcastable index grids
            rr_pos, cc_pos = torch.meshgrid(row_idx, cols_pos, indexing="ij")
            rr_neg, cc_neg = torch.meshgrid(row_idx, cols_neg, indexing="ij")
            print(intensiveComponents[rr_pos, cc_pos])
            print(intensiveComponents[rr_neg, cc_neg])

            # Scale updates
            intensiveComponents[rr_pos, cc_pos] = a[:, None] * intensiveComponents[rr_pos, cc_pos]
            intensiveComponents[rr_neg, cc_neg] = b[:, None] * intensiveComponents[rr_neg, cc_neg]

        return intensiveComponents
    
    def polish_negative_spFe(self, intensiveComponents):
        # Check and correct for below zero FeO
        PosS = torch.sum(intensiveComponents[:, label_indices_comp['spinel'][torch.tensor([0,1,2,4], dtype = torch.int)]], dim=-1
        ) + ( intensiveComponents[:, label_indices_comp['spinel'][4]]*2.25 )# Sumterm
        NegS = intensiveComponents[:, label_indices_comp['spinel'][3]] * 19

        illegal = NegS > PosS
        if illegal.any():
            # First form indexers
            # Rows (batch indices) where constraint is violated
            row_idx = torch.nonzero(illegal, as_tuple=False).squeeze(-1).to(torch.int)
            print(row_idx)
            # Column indices (spinel subset)
            cols_pos = torch.tensor(label_indices_comp['spinel'][torch.tensor([0,1,2,4], dtype = torch.int)],
                                    device=intensiveComponents.device, dtype = torch.int)
            cols_neg = torch.tensor(label_indices_comp['spinel'][3],
                                   device=intensiveComponents.device, dtype = torch.int)

            # Build broadcastable index grids
            rr_pos, cc_pos = torch.meshgrid(row_idx, cols_pos, indexing="ij")
            rr_neg, cc_neg = torch.meshgrid(row_idx, cols_neg, indexing="ij")
            A = torch.sum(intensiveComponents[rr_pos, cc_pos], dim=-1)
            print(PosS[illegal])
           # print(NegS[illegal.unsqueeze(-1)])
            #print(torch.where(illegal))
            denom = (19*A) + PosS[illegal] 
            a = 19.0 / denom
            b = PosS[illegal] / (denom *  intensiveComponents[illegal, label_indices_comp['spinel'][3]])

            # Scale updates
            intensiveComponents[rr_pos, cc_pos] = a[:, None] * intensiveComponents[rr_pos, cc_pos]
            intensiveComponents[rr_neg, cc_neg] = b[:, None] * intensiveComponents[rr_neg, cc_neg]

        return intensiveComponents
    
    def polish_negative_spAl(self, intensiveComponents):
        # Check and correct for negative Al in spinel
        NegS = (intensiveComponents[:,label_indices_comp['spinel'][2]] * (2/3)) + (intensiveComponents[:,label_indices_comp['spinel'][4]] * (1/4))
        PosS =  (intensiveComponents[:,label_indices_comp['spinel'][1]] ) + (intensiveComponents[:,label_indices_comp['spinel'][3]])
        illegal = NegS > PosS
        if illegal.sum():  # Calculate scaling factors to zero out negative hercynite without affecting Chromite
            A =  (intensiveComponents[illegal,label_indices_comp['spinel'][2]] ) + (intensiveComponents[illegal,label_indices_comp['spinel'][4]])
            RemS = A + PosS[illegal] # equivalent to 1-chromite
            a = RemS / (A + NegS[illegal])
            b = (RemS * NegS[illegal]) / ( PosS[illegal] * (A + NegS[illegal]) )
            intensiveComponents[illegal,label_indices_comp['spinel'][2]], intensiveComponents[illegal,label_indices_comp['spinel'][4]] = a * intensiveComponents[illegal,label_indices_comp['spinel'][2]], a * intensiveComponents[illegal,label_indices_comp['spinel'][4]]
            intensiveComponents[illegal,label_indices_comp['spinel'][1]], intensiveComponents[illegal,label_indices_comp['spinel'][3]] =  b * intensiveComponents[illegal,label_indices_comp['spinel'][1]], b * intensiveComponents[illegal,label_indices_comp['spinel'][3]] 
        return intensiveComponents
    
    def polish_negative_sp(self, intensiveComponents, trial = 0):
        # Indices for spinel components
        idxs = label_indices_comp['spinel']
        i1, i2, i3, i4, i5 = idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]

        # Extract components
        c1 = intensiveComponents[:, i1]
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
            cols_a = torch.tensor([i1, i2], device=intensiveComponents.device, dtype=torch.int)
            cols_b = torch.tensor([i3, i5], device=intensiveComponents.device, dtype=torch.int)
            cols_c = torch.tensor([i4], device=intensiveComponents.device, dtype=torch.int)

            rr_a, cc_a = torch.meshgrid(row_idx, cols_a, indexing="ij")
            rr_b, cc_b = torch.meshgrid(row_idx, cols_b, indexing="ij")
            rr_c, cc_c = torch.meshgrid(row_idx, cols_c, indexing="ij")

            # Extract per-row values
            A = intensiveComponents[row_idx][:, [i1, i2]].sum(dim=-1)  # sum(c1,c2)
            B = intensiveComponents[row_idx][:, [i3, i5]].sum(dim=-1)  # sum(c3,c5)
            C = intensiveComponents[row_idx][:, i4]                    # c4

            # Terms for constraints
            L1_c1c2 = intensiveComponents[row_idx][:, [i1, i2]].sum(dim=-1)
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
                rr_e, cc_e = torch.meshgrid(row_idx, torch.tensor(label_indices_comp['spinel'], dtype = torch.int, device = intensiveComponents.device), indexing="ij")
                print('SPINEL COMPOSITIONS:')
                print(intensiveComponents[rr_e, cc_e])
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
        inf_mask = ((x[:,3:] == 0).to(torch.float32) @ self.boolTransCompToOx[self.compositionally_variable_subset].T.to(torch.float32)) != 0 #be,ec->bc 


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
                print(f"To solve 0CaO Px: {round((begin_spinel-begin_refit)*1E6)} microsec; To solve 0FeO/Al2O3 Sp: {round((time.time()-begin_spinel)*1E6)} microsec ")

            
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

        # Identify superliquidus rows: only the last head > 0.5
        superliquidus = (binary_pred[:, :-1].sum(dim=1) == 0) #& (binary_pred[:, -1] == 1)
        non_super = ~superliquidus

        if binaries is None:
            binary_inp = binary_pred
        else:
            binary_inp = binaries
        features = x  # alias for clarity

        if self.middleBrain is not None: # If there is a middle encoder, use it. Otherwise prepare first encodings and phase saturation
            CoreOutput = self.middleBrain(torch.cat([latent, binary_inp, features], dim=1))
        else: 
            CoreOutput = torch.cat([latent, binary_inp, features], dim=1)

        zero_mask = binary_inp[:, self.comp_binaries] @ self.comp_mappings  # [batch, n_components]

        if binaries is None:  # Inference
            chem_outputs = [
                head(CoreOutput, inf_mask=inf_mask[:, (self.comp_mappings[i]).to(torch.bool)])
                for i, head in enumerate(self.chem_heads)
            ]
            chem_out = torch.cat(chem_outputs, dim=1)

            # Zero out superliquidus rows
            chem_out[superliquidus] = 0.0

            # Overwrite liquid component columns with feature composition
            liq_idx = torch.tensor(label_indices_comp['melts-liquid'], device=chem_out.device)
            #chem_out[superliquidus][:, liq_idx] = features[superliquidus, 3:]
            chem_out[superliquidus, -len(liq_idx):] = features[superliquidus, 3:]

            print(f"NANs in chem_out after superliquidus assignment: {torch.isnan(chem_out).sum()}")

            if not NN_only and non_super.any():
                chem_out[non_super] = self.polish_negative_px(chem_out[non_super])
                chem_out[non_super] = self.polish_negative_sp(chem_out[non_super])

        else:  # Training
            chem_outputs = [
                head(CoreOutput, train_inf_mask=inf_mask[:, (self.comp_mappings[i]).to(torch.bool)])
                for i, head in enumerate(self.chem_heads)
            ]
            chem_out = torch.cat(chem_outputs, dim=1)

            # Overwrite liquid component columns with feature composition
            liq_idx = torch.tensor(label_indices_comp['melts-liquid'], device=chem_out.device)
            chem_out[superliquidus][:, liq_idx] = features[superliquidus, 3:]


        # Compute phase properties
        phaseMass, reconBulk, componentMoles, phaseProportions = self.forward_phase_moles(
            CoreOutput, binary_mask=binary_pred.detach(), intensiveComponents=chem_out, details_out=True
        )

        # Assign direct values for superliquidus rows
        liq_idx_phase = torch.tensor(label_indices['melts-liquid'], device=chem_out.device)

        reconBulk[superliquidus] = features[superliquidus, 3:]
        componentMoles[superliquidus][:, liq_idx_phase] = features[superliquidus, 3:]
        phaseProportions[superliquidus][:, liq_idx_phase] = features[superliquidus, 3:]
        reconBulk[superliquidus] = features[superliquidus, 3:]
        phaseMass[superliquidus, -1] = 1.0


        if binaries is None:
            if detailed:
                return likelihoods, chem_out*zero_mask, phaseMass, reconBulk, componentMoles, phaseProportions # Inference with residual fitting
            else:
                return likelihoods, chem_out*zero_mask, phaseMass, reconBulk # Inference
        else:
            return logits, chem_out*zero_mask, zero_mask, phaseMass, reconBulk # Training, return zero mask for loss masking of intensive chemistries"""


class CombinedNetwork(nn.Module):
    """
    A combined class that wraps both TunableModel and MidLevelNetwork objects.
    
    This class provides a unified interface to both the base TunableModel
    and the extended MidLevelNetwork functionality, allowing flexible use
    of either component as needed.
    
    Parameters
    ----------
    encoderLayerUp : int, default=0
        Number of layers to expand encodings
    encoderLayerDown : int, default=0
        Number of layers to downscale the encodings
    middleLayerUp : int, default=0
        Number of middle layers to expand
    middleLayerDown : int, default=0
        Number of middle layers to compress
    low_regularization : str, default='none'
        Regularization for encoder layers
    high_regularization : str, default='none'
        Regularization for middle layers
    activation_leak : float, default=0.05
        Leak parameter for LeakyReLU activation
    lowWD : float, default=0
        Weight decay for lower model layers
    highWD : float, default=0
        Weight decay for upper model layers
    noise : float, default=0
        Noise level for training
    description : str, default=''
        Description of the model's target
    """
    def __init__(self, encoderLayerUp=0, encoderLayerDown=0,
                 middleLayerUp=0, middleLayerDown=0,
                 low_regularization='none', high_regularization='none', 
                 activation_leak=0.05,
                 lowWD=0, highWD=0, noise=0, description=''):
        super().__init__()
        
        # Create the base TunableModel
        self.tunable_model = TunableModel(
            encoderLayerUp=encoderLayerUp,
            encoderLayerDown=encoderLayerDown,
            middleLayerUp=middleLayerUp,
            middleLayerDown=middleLayerDown,
            low_regularization=low_regularization,
            high_regularization=high_regularization,
            activation_leak=activation_leak
        )
        
        # Create the MidLevelNetwork (which inherits from TunableModel)
        self.midlevel_network = MidLevelNetwork(
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
            description=description
        )
        
        # Store configuration
        self.config = self.midlevel_network.config
        
    def forward(self, x, binaries=None, detailed=False, NN_only=False):
        """
        Forward pass using the MidLevelNetwork.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        binaries : torch.Tensor or None
            Binary phase labels (for training)
        detailed : bool, default=False
            Whether to return detailed outputs
        NN_only : bool, default=False
            Whether to skip physics polishing
            
        Returns
        -------
        Output from MidLevelNetwork.forward()
        """
        return self.midlevel_network.forward(x, binaries=binaries, detailed=detailed, NN_only=NN_only)
    
    def forward_binaries(self, x):
        """
        Forward pass for binary predictions only.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            Binary saturation logits
        """
        return self.midlevel_network.forward_binaries(x)
    
    def forward_phase_moles(self, latentx, binary_mask, intensiveComponents, details_out=False):
        """
        Predict phase moles and reconstruct bulk composition.
        
        Parameters
        ----------
        latentx : torch.Tensor
            Latent representation
        binary_mask : torch.Tensor
            Binary phase mask
        intensiveComponents : torch.Tensor
            Intensive component predictions
        details_out : bool, default=False
            Whether to return detailed outputs
            
        Returns
        -------
        Output from MidLevelNetwork.forward_phase_moles()
        """
        return self.midlevel_network.forward_phase_moles(
            latentx, binary_mask, intensiveComponents, details_out=details_out
        )
    
    def save(self, DictFilePath):
        """
        Save model state and configuration.
        
        Parameters
        ----------
        DictFilePath : str
            Path to save the model
        """
        self.midlevel_network.save(DictFilePath)
    
    @property
    def encoder(self):
        """Access to the encoder from MidLevelNetwork."""
        return self.midlevel_network.encoder
    
    @property
    def sat_head(self):
        """Access to saturation heads from MidLevelNetwork."""
        return self.midlevel_network.sat_head
    
    @property
    def chem_heads(self):
        """Access to chemical heads from MidLevelNetwork."""
        return self.midlevel_network.chem_heads
    
    @property
    def middleBrain(self):
        """Access to middle brain from MidLevelNetwork."""
        return self.midlevel_network.middleBrain
    
    @property
    def mole_head(self):
        """Access to mole head from MidLevelNetwork."""
        return self.midlevel_network.mole_head


