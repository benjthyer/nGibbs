"""
Neural Network Emulator for MELTS.

Contains the NN_MELTS class which wraps the neural network model and provides
high-level interfaces for phase prediction, mass balancing, and fractional crystallization.
"""

import time
import numpy as np
import torch

# Import constants and mappings from config
from ..config import (
    MELTS_indices,
    Elkeys,
    compToOx,
    oxToEl,
    MM,
    Minv,
    Mtot,
    label_indices,
    label_indices_comp,
    mass_phasedict,
    phaseToCompMap,
    variedToAllComp,
    oxide_dict,
    boolTransCompToOx,
    compositionally_variable_binaries,
    fixed_phaseToCompMap,
)

# Import utility functions
from ..utils.math_utils import QFM_fO2, Fe2O3_FeO_ratio

# Import Normalizer from parser
from ..data.parser import Normalizer

from ..engine.NN import MidLevelNetwork
# NOTE: MidLevelNetwork needs to be imported from the models module or Legacy
# For now, this will need to be imported separately:
# from ..engine.Legacy.nnMELTS import MidLevelNetwork
# or from a future models module

# Default normalization tensors
# These should ideally be loaded from a saved model or configuration file
PTfO2min = torch.tensor([1, 700, -5], device='cpu', dtype=torch.float)
PTfO2max = torch.tensor([10000, 2000, 5], device='cpu', dtype=torch.float)
min_tensor = torch.zeros(len(Elkeys) + 3, device='cpu', dtype=torch.float)
min_tensor[:3] = PTfO2min
range_tensor = torch.ones(len(Elkeys) + 3, device='cpu', dtype=torch.float)
range_tensor[:3] = PTfO2max - PTfO2min


class NN_MELTS:
    """
    Neural Network Emulator of MELTS.
    
    Holds the neural network model for binary phase saturation and mass/chemistry prediction.
    All inputs are elemental molar, normalized to total = 1.
    """
    
    def __init__(self, Model, min_tensor=min_tensor, range_tensor=range_tensor, cuda=False):
        """
        Initialize the NN_MELTS emulator.
        
        Parameters:
        -----------
        Model : torch.nn.Module
            The neural network model (typically MidLevelNetwork)
        min_tensor : torch.Tensor, optional
            Minimum values for normalization (default: module-level min_tensor)
        range_tensor : torch.Tensor, optional
            Range values for normalization (default: module-level range_tensor)
        cuda : bool, default=False
            Whether to use CUDA/GPU
        """
        if cuda:
            self.model = Model.eval().cuda()
            self.phaseToCompMap = torch.tensor(phaseToCompMap, dtype=torch.float, device='cuda')
            self.variedToAllComp = torch.tensor(variedToAllComp, dtype=torch.float, device='cuda')
            self.fixed_phaseToCompMap = torch.tensor(fixed_phaseToCompMap, dtype=torch.float, device='cuda')
            self.dev = 'cuda'
            self.compToOx = torch.tensor(compToOx, dtype=torch.float, device='cuda')
            self.boolTransCompToOx = torch.tensor(boolTransCompToOx, dtype=torch.float, device='cuda')
            self.oxToEl = torch.tensor(oxToEl, dtype=torch.float, device='cuda')
            self.elToOx = torch.linalg.inv(self.oxToEl[:len(Elkeys)])  # For FeOt only
            self.Minv = torch.tensor(Minv, dtype=torch.float, device='cuda')
            self.MM = torch.tensor(MM, dtype=torch.float, device='cuda')
            self.Mtot = torch.tensor(Mtot, dtype=torch.float, device='cuda').flatten()
        else:
            self.model = Model.eval().cpu()
            self.phaseToCompMap = torch.tensor(phaseToCompMap, dtype=torch.float, device='cpu')
            self.variedToAllComp = torch.tensor(variedToAllComp, dtype=torch.float, device='cpu')
            self.fixed_phaseToCompMap = torch.tensor(fixed_phaseToCompMap, dtype=torch.float, device='cpu')
            self.dev = 'cpu'
            self.compToOx = torch.tensor(compToOx, dtype=torch.float)
            self.boolTransCompToOx = torch.tensor(boolTransCompToOx, dtype=torch.float, device='cpu')
            self.oxToEl = torch.tensor(oxToEl, dtype=torch.float)
            self.elToOx = torch.linalg.inv(self.oxToEl[:len(Elkeys)])  # For FeOt only
            self.Minv = torch.tensor(Minv, dtype=torch.float)
            self.MM = torch.tensor(MM, dtype=torch.float)
            self.Mtot = torch.tensor(Mtot, dtype=torch.float).flatten()
        
        self.compToEl = self.compToOx @ self.oxToEl
        self.norm_features = Normalizer(min_tensor=min_tensor, range_tensor=range_tensor, cuda=cuda)

    def convertOxToMol(self, features, convert=True):
        """
        Convert oxide weight percent to elemental moles.
        
        Parameters:
        -----------
        features : torch.Tensor
            Input features with [P, T, logfO2, ...oxides...]
        convert : bool, default=True
            Whether to perform conversion
            
        Returns:
        --------
        torch.Tensor
            Features with oxides converted to elemental moles
        """
        if convert:
            colsize = features.shape[1] - 3
            unclosed = (features[:, 3:] @ self.Minv[:colsize, :colsize]) @ self.oxToEl[:colsize]
            closedmoles = unclosed / unclosed.sum(dim=1, keepdim=True)
            return torch.cat([features[:, :3], closedmoles], dim=1)
        else:
            return features

    def forward_binary(self, features, normalize=True, WtPercent=True):
        """
        Outputs probability of phase saturation.
        
        Still does not explicitly prevent impossible phases (e.g. apatite when no PO4 present).
        
        Parameters:
        -----------
        features : torch.Tensor
            Input features
        normalize : bool, default=True
            Whether to normalize features
        WtPercent : bool, default=True
            Whether input is in weight percent (needs conversion)
            
        Returns:
        --------
        torch.Tensor
            Phase saturation probabilities (sigmoid output)
        """
        if normalize:
            unNormed = features.clone()
            feat_input = self.norm_features.norm(self.convertOxToMol(unNormed, convert=WtPercent))
        else:
            feat_input = self.convertOxToMol(features.clone(), convert=WtPercent)
        print(feat_input)
        with torch.no_grad():
            return torch.sigmoid(self.model.forward_binaries(feat_input))

    def forward(self, features, Normalize=True, WtPercent=True, comp_table_out='oxides'):
        """
        Forward pass through the model with mass balancing.
        
        Parameters:
        -----------
        features : torch.Tensor
            Input features
        Normalize : bool, default=True
            Whether to normalize features
        WtPercent : bool, default=True
            Whether input is in weight percent
        comp_table_out : str, default='oxides'
            Output format: 'oxides', 'comps', 'components', or None
            
        Returns:
        --------
        tuple
            (transcomponent_hat, massTens) - transformed components and mass tensors
        """
        if Normalize:
            norm_features = self.norm_features.norm(self.convertOxToMol(features, convert=WtPercent))
        else:
            norm_features = self.convertOxToMol(features, convert=WtPercent)
        with torch.no_grad():
            likelihoods, chem_out, phaseMoles, reconBulk, componentMoles, phaseProportions = self.model.forward(
                norm_features, detailed=True
            )
            transcomponent_hat, massTens = self.polish_masses( # Tunable parameters. 
                phaseMoles, reconBulk, componentMoles, phaseProportions,
                features=norm_features, optimize_masses=False, protect_opx=True,
                comp_table_out=comp_table_out, output_componentMoles=False
            )
            return transcomponent_hat, massTens

    def Iron_Speciator(self, oxides, Normedfeatures):
        """
        Speciate iron between FeO and Fe2O3 based on fO2.
        
        Parameters:
        -----------
        oxides : torch.Tensor
            Tensor of size (n, O) where columns are liquid molar oxides (except for Fe2O3)
        Normedfeatures : torch.Tensor
            Normalized features (assumed normalized)
            
        Returns:
        --------
        torch.Tensor
            Oxides with Fe2O3 column added
        """
        features = self.norm_features.denorm(Normedfeatures)[:, :3]
        unNormed = oxides.clone()[:, :len(Elkeys)]  # Ensure we don't grab the potentially empty ferric column
        
        fO2_composition_Nos = torch.tensor(
            [oxide_dict[ox] for ox in ['Al2O3', 'FeO', 'CaO', 'Na2O', 'K2O']],
            device=self.dev
        )
        fO2_composition_ind = torch.zeros(len(Elkeys), device=self.dev).to(torch.bool)
        fO2_composition_ind[fO2_composition_Nos] = True
        
        row_sums = unNormed.sum(dim=1, keepdim=True)
        nonzero_mask = row_sums != 0
        row_sums[~nonzero_mask] = 1.0
        temp_renorm = unNormed * (1 / row_sums)
        temp_renorm[~nonzero_mask.expand_as(unNormed)] = 0.0
        
        # Get ferric/ferrous ratio
        IronR = Fe2O3_FeO_ratio(
            fO2=10**(QFM_fO2(
                K=features[nonzero_mask.flatten()][:, 1] + 273,
                P=features[nonzero_mask.flatten()][:, 0],
                use_torch=True
            ) + features[nonzero_mask.flatten()][:, 2]),
            T=features[nonzero_mask.flatten()][:, 1] + 273,
            P=1e5 * features[nonzero_mask.flatten()][:, 0],
            composition=temp_renorm[nonzero_mask.flatten()][:, fO2_composition_ind],
            use_torch=True,
            device=self.dev
        )
        
        ferricPerTot = 1 / (2 + (1 / IronR))
        ferrousPerTot = 1 / ((2 * IronR) + 1)

        ferric = torch.zeros((unNormed.size()[0], 1), device=self.dev, dtype=torch.float32)
        idx = nonzero_mask.flatten().nonzero(as_tuple=True)[0]

        ferric[idx, 0] = unNormed[idx, oxide_dict['FeO']] * ferricPerTot
        unNormed[idx, oxide_dict['FeO']] *= ferrousPerTot

        unNormed_out = torch.cat([unNormed, ferric], dim=1)
        return unNormed_out

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

        print(mask.size())
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
        print(A_masked)
        print(b_masked)
        print(mask)
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
        
        print(f'Descent Time: {time.time() - inner_start}')

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

        print(f"Clamp descent time: {time.time() - start:.4f}s")
        if pass_data:
            Ac = Ac.to(self.dev)

        return Ac

    def make_phase_tables(self, newComps, compToOx, MM, compPhaseMap, features, out='oxides', eps=1e-12):
        """
        Compute phase oxide wt% tables and phase mass fractions.

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
        phaseComps = newComps.unsqueeze(-1) * compPhaseMap  # (B, C, P)

        # Convert to oxides per phase (plug in iron speciator). Moles, then grams
        phaseOxMolar = torch.einsum("bcp,co->bpo", phaseComps, compToOx)
        print(phaseOxMolar[:3, -1])
        print(features[:3])
        liqWithFerric = self.Iron_Speciator(
            oxides=phaseOxMolar[:, -1].to(self.dev),
            Normedfeatures=features.to(self.dev)
        )
        phaseOxMolar[:, -1] = liqWithFerric
        print(phaseOxMolar[:3, -1])

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
            return phaseOxWt[:, torch.tensor(compositionally_variable_binaries, dtype=torch.bool)], phaseMassNorm

        elif out in ['comps', 'components']:
            # Returns intensive, chemically variable components
            phasesums = phaseComps.sum(dim=1, keepdim=True)  # (B, 1, P)
            phaseIntensive = phaseComps / (phasesums + 1E-12)  # (B, C, P)
            print(phaseIntensive.size())
            systemComps = torch.einsum('bcp,pc->bc', phaseIntensive, compPhaseMap.T)  # (B, C)
            return systemComps, phaseMassNorm

        else:
            return phaseMassNorm

    def retrieveMassesFast(self, components, features, binaries, descent=False, pinv=False, verbose=False):
        """
        Retrieve phase masses from components and binaries.
        
        Parameters:
        -----------
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
            
        Returns:
        --------
        tuple
            (compTens, massTens) - component and mass tensors
        """
        if self.dev == 'cuda' and not descent:
            fundev = 'cpu'
            print('converting...')
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
        ncomps = label_indices['melts-liquid'][-1] + 1
        nphases = mass_phasedict['melts-liquid'] + 1
        bulk = features[:, 3:].clone()

        # Organize phaseToComp Matrix w/ NN output
        phaseToComp = torch.zeros((nrows, nphases, ncomps), device=fundev)  # (B, P, C)
        phaseToCompMap = torch.zeros((nphases, ncomps), device=fundev)  # (P, C)

        for phase, binary_ind in mass_phasedict.items():
            phaseToCompMap[binary_ind, label_indices[phase]] = 1
            if len(label_indices[phase]) > 1:
                phaseToComp[:, binary_ind, label_indices[phase]] = components[:, label_indices_comp[phase]]
            else:
                phaseToComp[:, binary_ind, label_indices[phase]] = binaries[:, binary_ind].unsqueeze(-1)

        print(self.compToOx.size())
        print(self.compToOx.device)
        
        # Calculate phaseToEl/Ox matrix
        phaseToOx = torch.einsum('bpc,co->bpo', phaseToComp, self.compToOx)  # (B, P, O)
        phaseToEl = torch.einsum('bpo,oe->bpe', phaseToOx, self.oxToEl)  # (B, P, E)

        A = phaseToEl.transpose(1, 2)
        print(A)
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
            print(f'Time for second Descent and residual fitting: {time.time() - descend_time}')

        if self.dev == 'cuda' and not descent:
            self.compToEl = self.compToEl.to('cuda')
            self.compToOx = self.compToOx.to('cuda')
            self.oxToEl = self.oxToEl.to('cuda')

        print(f'Total Linear Algebra Time: {time.time() - start} seconds')

        compTens, massTens = self.make_phase_tables(
            newComps, self.compToOx, self.MM, compPhaseMap=phaseToCompMap,
            features=features, eps=1e-12
        )
        compTens = compTens[:, self.model.comp_binaries].detach().cpu().numpy()
        massTens = massTens.detach().cpu().numpy()
        return compTens, massTens

    def polish_negative_px(self, phaseProportions):
        """Check for below zero CaO in orthopyroxene and correct."""
        PosS = torch.sum(phaseProportions[:, label_indices_comp['orthopyroxene'][:5]], dim=-1)
        NegS = (phaseProportions[:, label_indices_comp['orthopyroxene'][5]] * 2) + phaseProportions[:, label_indices_comp['orthopyroxene'][6]]
        illegal = NegS > PosS
        if illegal.sum():
            denom = ((2 * phaseProportions[illegal, label_indices_comp['orthopyroxene'][6]]) +
                     (3 * phaseProportions[illegal, label_indices_comp['orthopyroxene'][5]]))
            b = 1 / denom
            a = NegS[illegal] / (denom * PosS[illegal])
            phaseProportions[illegal, label_indices_comp['orthopyroxene'][:5]] = a * phaseProportions[illegal, label_indices_comp['orthopyroxene'][:5]]
            phaseProportions[illegal, label_indices_comp['orthopyroxene'][-2:]] = b * phaseProportions[illegal, label_indices_comp['orthopyroxene'][-2:]]

    def polish_negative_sp(self, phaseProportions):
        """Check for negative hercynite in spinel and correct."""
        NegS = (phaseProportions[:, label_indices_comp['spinel'][2]] * (2/3)) + (phaseProportions[:, label_indices_comp['spinel'][4]] * (1/4))
        PosS = (phaseProportions[:, label_indices_comp['spinel'][1]]) + (phaseProportions[:, label_indices_comp['spinel'][3]])
        illegal = NegS > PosS
        if illegal.sum():
            A = (phaseProportions[illegal, label_indices_comp['spinel'][2]]) + (phaseProportions[illegal, label_indices_comp['spinel'][4]])
            RemS = A + PosS[illegal]
            a = RemS / (A + NegS[illegal])
            b = (RemS * NegS[illegal]) / (PosS[illegal] * (A + NegS[illegal]))
            phaseProportions[illegal, label_indices_comp['spinel'][2]], phaseProportions[illegal, label_indices_comp['spinel'][4]] = (
                a * phaseProportions[illegal, label_indices_comp['spinel'][2]],
                a * phaseProportions[illegal, label_indices_comp['spinel'][4]]
            )
            phaseProportions[illegal, label_indices_comp['spinel'][1]], phaseProportions[illegal, label_indices_comp['spinel'][3]] = (
                b * phaseProportions[illegal, label_indices_comp['spinel'][1]],
                b * phaseProportions[illegal, label_indices_comp['spinel'][3]]
            )

    def polish_masses(self, phaseMoles, reconBulk, componentMoles, phaseProportions, features,
                      optimize_masses=True, output_componentMoles=False, protect_opx=False, comp_table_out='oxides'):
        """
        Polish masses to fit bulk composition.
        
        Parameters:
        -----------
        phaseMoles : torch.Tensor
            Phase moles from model
        reconBulk : torch.Tensor
            Reconstructed bulk from model
        componentMoles : torch.Tensor
            Component moles from model
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
            
        Returns:
        --------
        tuple or torch.Tensor
            Phase tables and optionally component moles
        """
        phaseMoles = phaseMoles.to('cpu')
        reconBulk = reconBulk.to('cpu')
        componentMoles = componentMoles.to('cpu')
        phaseProportions = phaseProportions.to('cpu')
        feats = features.to('cpu')
        bulk = feats[:, 3:]
        compToEl = self.compToEl.to('cpu')
        compToOx = self.compToOx.to('cpu')
        phaseToCompMap = self.phaseToCompMap.to('cpu')
        MM = self.MM.to('cpu')

        residual = bulk - reconBulk
        if optimize_masses:
            phaseComponentMoles = componentMoles[:, None, :] * phaseToCompMap[None, :, :]  # (B, P, C)
            phaseAtomMoles = torch.einsum('bpc,ce->bpe', phaseComponentMoles, compToEl)  # (B, P, E)
            print(phaseAtomMoles[:3])
            wtDelPhaseMoles = (torch.linalg.pinv(phaseAtomMoles.transpose(1, 2)) @ residual.unsqueeze(-1)).squeeze(-1)
            DelPhaseMoles = wtDelPhaseMoles * phaseMoles
            print('Before Phase Mass adjustments, phasemoles, then residual')
            print(phaseMoles)
            print(residual)
            phaseMoles = phaseMoles + DelPhaseMoles
            componentMoles = phaseProportions * (phaseMoles @ phaseToCompMap)
            reconBulk = componentMoles @ compToEl
            residual = bulk - reconBulk
            print('After Phase Mass adjustments, phasemoles, then residual')
            print(phaseMoles)
            print(residual)

        # Solve underconstrained problem
        componentAtomMoles = componentMoles.unsqueeze(-1) * compToEl  # (B, C, E)
        if protect_opx:
            componentAtomMoles[:, label_indices['orthopyroxene']] *= 0
        print('componentAtomMoles')
        print(componentAtomMoles)
        if torch.any(torch.isnan(componentAtomMoles)):
            print(torch.where(torch.isnan(componentAtomMoles)))
        wtDelComponentMoles = (torch.linalg.pinv(componentAtomMoles.transpose(1, 2)) @ residual.unsqueeze(-1)).squeeze(-1)
        DelComponentMoles = wtDelComponentMoles * componentMoles
        componentMoles = componentMoles + DelComponentMoles

        if output_componentMoles:
            return (
                self.make_phase_tables(componentMoles, compToOx, MM, compPhaseMap=phaseToCompMap.T,
                                      features=feats, eps=1e-12, out=comp_table_out),
                componentMoles,
                wtDelComponentMoles
            )
        else:
            return self.make_phase_tables(componentMoles, compToOx, MM, compPhaseMap=phaseToCompMap.T,
                                         features=feats, eps=1e-12, out=comp_table_out)

    def find_liquidus(self, features, resolution=25):
        """Returns lowest identified superliquidus temperature between 800 and 2000 C"""
        T_test = torch.tensor(np.linspace(800, 2000, int(1200/resolution) + 1), device=self.dev)
        feat_input = np.zeros((int(1200/resolution) + 1, len(features)))
        feat_input[:] = features
        feat_input[:, 1] = T_test
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
            (N, F) where F >= 2 and column 1 is temperature
        resolution : int, default=25
            Temperature resolution
        weightOxinput : bool, default=False
            Whether input is in weight percent
            
        Returns:
        --------
        torch.Tensor
            (N) - Liquidus temperatures for each composition
        """
        if weightOxinput:
            features_batch = self.convertOxToMol(features)
        else:
            features_batch = features
        N, F = features_batch.shape
        T_test = torch.linspace(800, 2000, steps=int(1200 / resolution) + 1)
        n_temps = T_test.shape[0]

        feat_input = features_batch.unsqueeze(1).repeat(1, n_temps, 1)  # [N, T, F]
        feat_input[:, :, 1] = T_test.unsqueeze(0).repeat(N, 1)
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

    def find_mineral_cosaturation(self, features, T_initial_C, phase_cols, dt_C=1, weightOxinput=False):
        """
        Vectorized version to find mineral co-saturation temperatures.
        
        Parameters:
        -----------
        features : torch.Tensor
            (N, F) where F >= 2 and column 1 is temperature
        T_initial_C : float
            Initial temperature in Celsius
        phase_cols : list
            List of phase column indices
        dt_C : float, default=1
            Temperature step size
        weightOxinput : bool, default=False
            Whether input is in weight percent
            
        Returns:
        --------
        torch.Tensor
            (N, Pc) - Saturation temperatures for each phase
        """
        if weightOxinput:
            features_batch = self.convertOxToMol(features)
        else:
            features_batch = features

        N, F = features_batch.shape
        T_test = torch.linspace(700, T_initial_C, steps=int((T_initial_C - 700) / dt_C) + 1, device=features.device)
        n_temps = T_test.shape[0]

        feat_input = features_batch.unsqueeze(1).repeat(1, n_temps, 1)  # [N, T, F]
        feat_input[:, :, 1] = T_test.unsqueeze(0).repeat(N, 1)
        feat_input = feat_input.view(N * n_temps, F)

        with torch.no_grad():
            binaries = (self.forward_binary(feat_input) > 0.5).float()

        binaries = binaries[:, phase_cols]
        binaries = binaries.view(N, n_temps, -1)

        T_masked = T_test.view(1, n_temps, 1) * binaries
        saturation_temps, _ = T_masked.max(dim=1)

        return saturation_temps

    def fractional_crystalization(self, features, T_path, fit_residual=True, WtPercent=True):
        """
        Perform fractional crystallization simulation.
        
        Parameters:
        -----------
        features : torch.Tensor
            (nB, n_features) where:
            - [:, 0] = pressure
            - [:, 1] = temperature (to be updated each step)
            - [:, 2] = log fO2 delta QFM
            - [:, 3:] = elemental composition (normalized to 1)
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
        with torch.no_grad():
            if WtPercent:
                inp_tensor = self.convertOxToMol(features)
            else:
                inp_tensor = features

            nB = inp_tensor.size(0)
            nEl = self.oxToEl.size(1)
            nC = label_indices['melts-liquid'][-1] + 1
            nP = len(list(label_indices.keys()))
            nSteps = len(T_path)

            component_tensor = torch.zeros((nB, nC, nSteps), dtype=torch.float32, device=self.dev)
            mass_tensor = torch.zeros((nB, nP, nSteps), dtype=torch.float32, device=self.dev)

            is_alive = torch.ones(nB, dtype=torch.bool, device=self.dev)
            prev_melt_frac = torch.ones(nB, dtype=torch.float32, device=self.dev)
            active_comps = inp_tensor[:, 3:].clone().to(self.dev)

            for i, temp in enumerate(T_path):
                print(temp)
                if not is_alive.any():
                    print('All systems froze!')
                    break

                idx_alive = torch.nonzero(is_alive, as_tuple=True)[0]
                inp_batch = inp_tensor[idx_alive]
                inp_batch[:, 1] = temp
                inp_batch[:, 3:] = active_comps

                if inp_batch.numel() == 0:
                    raise RuntimeError("inp_batch is empty before normalization.")

                print(inp_batch)
                print(idx_alive)
                _, transcomponent_hat, phaseMoles, reconBulk, componentMoles, phaseProportions = self.model.forward(
                    self.norm_features.norm(inp_batch), detailed=True
                )
                
                if fit_residual:
                    massTens, componentMoles2, _ = self.polish_masses(
                        phaseMoles, reconBulk, componentMoles, phaseProportions,
                        features=self.norm_features.norm(inp_batch), optimize_masses=False,
                        protect_opx=True, comp_table_out='None', output_componentMoles=True
                    )
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


def rebuild_MELTS_model(DictFilePath, substitutions=None, low_only=False):
    """
    Loads MELTS NN with saved architecture.
    
    Requires saved .config dictionary attribute.
    Substitutions are dictionaries with configurations to instantiate in the new model
    that were not in the old one to load. Useful for building upper model on trained lower model.
    
    Parameters:
    -----------
    DictFilePath : str
        Path to saved model checkpoint
    substitutions : dict, optional
        Configuration substitutions
    low_only : bool, default=False
        If True, only load lower model weights
        
    Returns:
    --------
    torch.nn.Module
        Loaded model
        
    Note:
    -----
    MidLevelNetwork must be imported separately, e.g.:
    from ..engine.Legacy.nnMELTS import MidLevelNetwork
    """
    # NOTE: This function requires MidLevelNetwork to be imported
    # For now, users will need to import it separately
    # TODO: Create a proper models module and import from there
    try:
        from ..engine.Legacy.nnMELTS import MidLevelNetwork
    except ImportError:
        raise ImportError(
            "MidLevelNetwork not found. Please import it from the appropriate module:\n"
            "from ..engine.Legacy.nnMELTS import MidLevelNetwork"
        )
    
    ckpt = torch.load(DictFilePath)
    configuration = ckpt['config']
    if substitutions is not None:
        for parameter, setting in substitutions.items():
            configuration[parameter] = setting

    model = MidLevelNetwork(**configuration)

    if low_only:
        model_dict = model.state_dict()
        allowed_prefixes = ["encoder.", "sat_head."]
        filtered_dict = {
            k: v for k, v in ckpt['state_dict'].items()
            if any(k.startswith(p) for p in allowed_prefixes)
        }
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
    else:
        model.load_state_dict(ckpt['state_dict'], strict=False)

    return model
