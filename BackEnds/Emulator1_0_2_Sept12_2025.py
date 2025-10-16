

"""Nonzero SiO2, Al2O3, MgO, FeO, CaO required
~Sept 12: Changed model so that regularization occurs after dropout and activation functions. Reversed
Future Organization: Make this the NN archetecture python file. Have one more higher level python 
script that loads the models into one object. 
"""

date = "Sept25"
modelname = "rhyoliteMELTS1.0.2_FullPolished"
CrDictFilePath=f'./{modelname}_{date}Cr.pt'
NoCrDictFilePath=f'./{modelname}_{date}NoCr.pt'

import numpy as np 
import random
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
import pickle
import os
import pandas as pd
import random
plt.rcParams['figure.figsize'] = [10, 7]
from matplotlib.colors import LinearSegmentedColormap
#import mpl_scatter_density # adds projection='scatter_density'
from scipy.stats import gaussian_kde
from scipy import optimize
from molmass import Formula
import csv
import re
import copy
import gc
import time
import molmass as ms
from tqdm import tqdm
from EmulatorLibrary import * # Replaces defining here. 

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, Dropout, BCELoss, Sigmoid, MSELoss
from torch.optim import Adam, SGD
#import torchvision.transforms as T #THIS ONE IS BROKEN 04/04/2025
import torch.nn as nn
import torch.nn.functional as F


PTfO2min = torch.tensor([1,700,-5], device = 'cpu', dtype = torch.float)
PTfO2max = torch.tensor([10000,2000,5], device = 'cpu', dtype = torch.float)
min_tensor = torch.zeros(len(Elkeys)+3, device = 'cpu', dtype = torch.float)
min_tensor[:3] = PTfO2min
range_tensor = torch.ones(len(Elkeys)+3, device = 'cpu', dtype = torch.float)
range_tensor[:3] = PTfO2max - PTfO2min
normf = Normalizer(min_tensor=min_tensor, range_tensor=range_tensor)

"""
from scipy.optimize import nnls

def solve_batch_nnls(bulkEl, phaseToEl):
    B, E = bulkEl.shape
    P = phaseToEl.shape[1]
    molPhase = torch.zeros((B, P), dtype=bulkEl.dtype)

    for b in range(B):
        A = phaseToEl[b].T.cpu().numpy()  # shape (E, P)
        b_vec = bulkEl[b].cpu().numpy()   # shape (E,)
        x, _ = nnls(A, b_vec)             # x: shape (P,)
        molPhase[b] = torch.tensor(x, dtype=bulkEl.dtype)
    
    return molPhase"""

## NEW SHARED ENCODER BUILD JULY 25
class PhaseHead(nn.Module):
    def __init__(self, n_components, signed=False, learn_temp=True):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 32),#, bias=False),
            #nn.BatchNorm1d(32),
            nn.LeakyReLU(0.05),
            nn.Linear(32, n_components)
        )
        self.signed = signed

        if learn_temp and signed:
            self.temp = nn.Parameter(torch.tensor(1.0))  # start at 1.0
        else:
            self.register_buffer("temp", torch.tensor(1.0))  # fixed value

    def forward(self, x, inf_mask=None, train_inf_mask=None):
        """Pass inf_mask """
        raw = self.fc(x)

        if self.signed:
            # Signed softmax with offset and temperature
            x_centered = raw - raw.mean(dim=-1, keepdim=True)
            x_scaled = self.temp * x_centered
            norm = x_scaled.abs().sum(dim=-1, keepdim=True) + 1e-6
            return (1 / raw.size(-1)) + x_scaled / norm
        else:
            if train_inf_mask is not None: # For training, push impossible values to approach zero
                raw[train_inf_mask] = -1E9 # (size b,subC), neg infinite logits for 0 after softmax, safe for gradient descent
                proportions = F.softmax(raw, dim=-1)

            elif inf_mask is not None: # For inference. Send impossible values to literal zeros
                raw[inf_mask] = -torch.inf # (size b,subC), neg infinite logits for 0 after softmax
                proportions = F.softmax(raw, dim=-1)
                proportions[torch.isnan(proportions)] = 0 # If all components are impossible, we get nans that break linear algebra. Cover these up with zeros. Does not occur during training with above method.
            
            return proportions

# New Model With Molar Abundance Output

class DualSaturationChemistry(nn.Module):
    """Neural network architecture for binary predictions of phase saturation, and for predicting intensive quantities . 
    Utilized a shared encoder that converts PTX features into latent encoding that a series of unique parallel phase heads 
    use to predict phase saturation and intensive chemistry.
    The output of the saturation model needs to be passed through a sigmoid function, which is left out here because the BCEwithlogits loss function 
    efficiently wraps the sigmoid operation into the loss function for faster training.
    9/12: NO dropout. Use AdamW optimizer for weight decay"""
    
    def __init__(self, input_dim=3+len(Elkeys), n_phases=len(list(label_indices.keys()))):
        super().__init__()
        self.n_phases = n_phases
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),# bias=False),
            #nn.BatchNorm1d(64),
            nn.LeakyReLU(0.05),
            
            nn.Linear(64, 128),# bias=False),
            #nn.BatchNorm1d(128),
            nn.LeakyReLU(0.05),
            
            nn.Linear(128, 256),# bias=False),
            #nn.BatchNorm1d(256),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),# bias=False),
            #nn.BatchNorm1d(128),
            nn.LeakyReLU(0.05),
            #nn.Dropout(0.05)
        )

        # Independent binary classification heads (1 per phase)
        self.sat_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 16),# bias=False),
                #nn.BatchNorm1d(16),
                nn.LeakyReLU(0.05),
                nn.Dropout(0.1),
                
                nn.Linear(16, 1)
            ) for _ in range(n_phases)
        ])

        self.mole_head = nn.Sequential(
            nn.Linear(128, 64),# bias=False),
            #nn.BatchNorm1d(64),
            nn.LeakyReLU(0.05),
            
            nn.Linear(64, 20), # Output to be put through softplus function
            nn.Softplus()
        )
        
        
        chem_list = []
        comp_mappingsL = []
        comp_binariesL = []
        #pure_mappingsL = []

        j = 0
        #k = 0
        for i, (label, inds) in enumerate(label_indices.items()):
            n_components = len(inds)
            if n_components > 1:
                comp_binariesL.append(i)
                comp_mappingsL = comp_mappingsL + np.repeat(j, n_components).tolist()
                """if label in ['spinel', 'orthopyroxene', 'clinopyroxene']:
                    chem_list.append(PhaseHead(n_components=n_components, signed=True))
                else:""" # No Signed layers as of 8/21/25
                chem_list.append(PhaseHead(n_components=n_components, signed=False))
                j += 1
            #else: 
                #pure_mappingsL += inds
                #k += 1

        #self.comp_binaries = torch.tensor(comp_binariesL, device = 'cpu')
        #self.comp_mappings = torch.zeros(j, len(comp_mappingsL), device = 'cpu') # Build Binary Matrix to project phases to components
        comp_mappings = torch.zeros(j, len(comp_mappingsL))
        #pure_mappings = torch.zeros(k, len(pure_mappingsL))
        self.pure_binaries_bool = torch.ones(i+1).to(bool)
        self.pure_binaries_bool[torch.tensor(comp_binariesL)] = False

        for col, row in enumerate(comp_mappingsL):
            comp_mappings[row, col] = 1
        """for col, row in enumerate(pure_mappingsL):
            pure_mappings[row, col] = 1"""
        #self.boolTransCompToOx = torch.tensor(boolTransCompToOx) # Collected compToOx from environment, not a buffer so that loading won't be affected
        #self.purePhaseToOxBool = boolTransCompToOx[torch.tensor(comp_mappingsL).to(bool)] Redo this
        self.register_buffer('boolTransCompToOx', torch.tensor(boolTransCompToOx)) 
        self.register_buffer('compositionally_variable_subset', torch.tensor(compositionally_variable_subset,dtype = int))
        self.register_buffer('comp_mappings', comp_mappings) 
        self.chem_heads = nn.ModuleList(chem_list)
        
        self.register_buffer('comp_binaries', torch.tensor(comp_binariesL)) # Use == 0 for pure binaries

        self.register_buffer('phaseToCompMap', torch.tensor(phaseToCompMap, dtype = torch.float))
        self.register_buffer('variedToAllComp', torch.tensor(variedToAllComp, dtype = torch.float))
        self.register_buffer('fixed_phaseToCompMap', torch.tensor(fixed_phaseToCompMap, dtype = torch.float))
        self.register_buffer('compToEl', torch.tensor(compToOx @ oxToEl, dtype = torch.float))



    def forward_binaries(self, x):
        """Outputs satuation logits only, to be passed through sigmoid. Useful for training with BCEwithlogits loss"""
        # Encode features
        latent = self.encoder(x)

        # Apply each head to the shared latent vector
        outputs = []
        for head in self.sat_heads:
            out = head(latent)  # shape: (batch_size, 1)
            outputs.append(out)
        #inf_mask = ((x[:,3:] == 0) @ self.boolTransCompToOx.T[self.]) != 0
        # Concatenate all outputs into shape: (batch_size, n_phases)
        return torch.cat(outputs, dim=1)
    
    def forward_chemistry(self, x, binaries):
        """Given binaries, outputs intensive phase chemistries for training chem heads"""
        # Encode features
        latent = self.encoder(x)
        
        zero_mask = binaries[:,self.comp_binaries] @ self.comp_mappings # 0 out absent components
        inf_mask = ((x[:,3:] == 0).to(torch.float32) @ self.boolTransCompToOx[self.compositionally_variable_subset].T.to(torch.float32)) != 0 #be,ec->bc 

        # Apply each head to the shared latent vector
        outputs = []
        for i, head in enumerate(self.chem_heads):
            out = head(latent, train_inf_mask = inf_mask[:,(self.comp_mappings[i]).to(torch.bool)])  # shape: (batch_size, 1) inf_mask (batch_size, components_in_phase)
            outputs.append(out)

        # Concatenate all outputs into shape: (batch_size, n_phases)
        return torch.cat(outputs, dim=1) * zero_mask, zero_mask

    def forward_phase_moles(self, latentx, binary_mask, intensiveComponents, details_out = False):
        """Predicts molar abundance of phases and reconstructs bulk composition. Let intensive components be indexed by label_indices_comp"""

        phaseMoles = self.mole_head(latentx) * binary_mask
        compMultipliers = phaseMoles @ self.phaseToCompMap #(B,C)
        intensivePhaseProportions = intensiveComponents @ self.variedToAllComp #BV, VC -> BC #NEED TO GET BINARIES AND PROPORTIONS TOGETHER IN COMPONENT FORM, RECREATE PHASETOCOMP (B,P,C).vASK IF INDEXING TO BUILD IS THE MOST EFFICIENT WAY
        phaseProportions = intensivePhaseProportions + self.fixed_phaseToCompMap # How to project? BC + 1C -> BC. Get ones where all pure phase components are
        componentMoles = phaseProportions * compMultipliers
        reconBulkUnNormed = componentMoles @ self.compToEl #(B,E)
        totals = reconBulkUnNormed.sum(dim=1)#torch.ones(reconBulkUnNormed.size()[0], device = 'cuda')   #(B) TEMP NO NORMALIZATION, LET LINEAR ALGEBRA TAKE CARE OF IT
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
        inf_mask = ((x[:,3:] == 0).to(torch.float32) @ self.boolTransCompToOx[self.compositionally_variable_subset].T.to(torch.float32)) != 0 #be,ec->bc 


        # Phase saturation logits (not yet sigmoid)
        sat_outputs = [head(latent) for head in self.sat_heads]
        logits = torch.cat(sat_outputs, dim=1)

        likelihoods = torch.sigmoid(logits)
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
            return logits, chem_out*zero_mask, zero_mask, phaseMass, reconBulk # Training, return zero mask for loss masking of intensive chemistries

GPUFullMELTS_Cr = DualSaturationChemistry().cuda()
#GPUFullMELTS_Cr.load_state_dict(torch.load(CrDictFilePath), strict = False) #New buffer

GPUFullMELTS_NoCr = DualSaturationChemistry()
#GPUFullMELTS_NoCr.load_state_dict(torch.load(NoCrDictFilePath), strict = False)

CPUFullMELTS_Cr = DualSaturationChemistry().cuda()
#CPUFullMELTS_Cr.load_state_dict(torch.load(CrDictFilePath), strict = False)

CPUFullMELTS_NoCr = DualSaturationChemistry()
#CPUFullMELTS_NoCr.load_state_dict(torch.load(NoCrDictFilePath), strict = False)

class NN_MELTS():
    """Neural Network Emulator of MELTS. Holds two models: Binary Phase Saturation and Mass Partitioning.
    All inputs are elemental molar, normalized to total = 1"""
    def __init__(self, Model, cuda = False):#ModelCr, ModelNoCr, cuda = False):#'components_resampled_normalization_stats.txt'):
    
        if cuda:
            #self.modelCr = ModelCr.eval().cuda()
            #self.modelNoCr = ModelNoCr.eval().cuda()
            self.model = Model.eval().cuda()
            self.phaseToCompMap = torch.tensor(phaseToCompMap, dtype = torch.float, device = 'cuda')
            self.variedToAllComp = torch.tensor(variedToAllComp, dtype = torch.float, device = 'cuda')
            self.fixed_phaseToCompMap = torch.tensor(fixed_phaseToCompMap, dtype = torch.float, device = 'cuda')
            self.dev = 'cuda'
            #self.compToOx = torch.tensor(compToOx[:label_indices['fluid'][-1]+1], dtype = torch.float, device = 'cuda') #Solid components only
            self.compToOx = torch.tensor(compToOx, dtype = torch.float, device = 'cuda')
            self.boolTransCompToOx = torch.tensor(boolTransCompToOx, dtype = torch.float, device = 'cuda')
            self.oxToEl = torch.tensor(oxToEl, dtype = torch.float, device = 'cuda')
            self.elToOx = torch.linalg.inv(self.oxToEl[:len(Elkeys)]) #For FeOt only
            self.Minv = torch.tensor(Minv, dtype = torch.float, device = 'cuda')
            self.MM = torch.tensor(MM, dtype = torch.float, device = 'cuda')
            self.Mtot = torch.tensor(Mtot, dtype = torch.float, device = 'cuda').flatten()
        else:
            #self.modelCr = ModelCr.eval().cpu()
            #self.modelNoCr = ModelNoCr.eval().cpu()
            self.model = Model.eval().cpu()
            self.phaseToCompMap = torch.tensor(phaseToCompMap, dtype = torch.float, device = 'cpu')
            self.variedToAllComp = torch.tensor(variedToAllComp, dtype = torch.float, device = 'cpu')
            self.fixed_phaseToCompMap = torch.tensor(fixed_phaseToCompMap, dtype = torch.float, device = 'cpu')
            self.dev = 'cpu'
            #self.compToOx = torch.tensor(compToOx[:label_indices['fluid'][-1]+1], dtype = torch.float) #Solid components only
            self.compToOx = torch.tensor(compToOx, dtype = torch.float)
            self.boolTransCompToOx = torch.tensor(boolTransCompToOx, dtype = torch.float, device = 'cpu')
            self.oxToEl = torch.tensor(oxToEl, dtype = torch.float)
            self.elToOx = torch.linalg.inv(self.oxToEl[:len(Elkeys)]) #For FeOt only.
            self.Minv = torch.tensor(Minv, dtype = torch.float)
            self.MM = torch.tensor(MM, dtype = torch.float)
            self.Mtot = torch.tensor(Mtot, dtype = torch.float).flatten()
        
        self.compToEl = self.compToOx @ self.oxToEl
        self.norm_features =  Normalizer(min_tensor=min_tensor, range_tensor=range_tensor, cuda = cuda)

    def convertOxToMol(self, features, convert = True):
        if convert:
            colsize = features.shape[1]-3
            unclosed = features[:,3:] @ self.Minv[:colsize,:colsize] @ self.oxToEl[:colsize]
            closedmoles = unclosed / unclosed.sum(dim=1, keepdim=True)
            return torch.cat([features[:,:3], closedmoles], dim = 1)
        else:
            return features 

    def forward_binary(self, features, normalize = True, WtPercent = True):
        """Outputs probability of phase saturation Needs binaries to prevent impossible phases (e.g. apatite when no PO4 present)"""
        if normalize: 
            unNormed = features.clone()
            feat_input = self.norm_features.norm(self.convertOxToMol(unNormed, convert = WtPercent))
        else:
            feat_input = self.convertOxToMol(features.clone(), convert = WtPercent)
        print(feat_input)
        with torch.no_grad():
                    #Zero out pure phases that are impossible with passed chemistry
            #zeroEl = features[3:] == 0 #(b,e)
            #pure_phase_mask = (zeroEl @ self.purePhaseToOxBool.T)  # (be,eP->bP) # Binary Phase Zero Mask
            return torch.sigmoid(self.model.forward_binaries(feat_input))
    
    
    def forward(self, features, Normalize = True, WtPercent = True, comp_table_out = 'oxides'): 
        if Normalize:
            norm_features = self.norm_features.norm(self.convertOxToMol(features, convert = WtPercent))
        else:
            norm_features = self.convertOxToMol(features, convert = WtPercent)
        with torch.no_grad():
            likelihoods, chem_out, phaseMoles, reconBulk, componentMoles, phaseProportions = self.model.forward(norm_features, detailed = True)
            transcomponent_hat, massTens = self.polish_masses(phaseMoles, reconBulk, componentMoles, phaseProportions, 
                                                                    features=norm_features, optimize_masses=False,  protect_opx=True, comp_table_out=comp_table_out,  output_componentMoles= False)
            return transcomponent_hat, massTens
            


    def Iron_Speciator(self, oxides, Normedfeatures):
        """Let oxides input be a tensor of size nxO, where columns are liquid molar oxides (except for Fe2O3)
        Features are assumed to be normalized. Output is oxides, now with Fe2O3"""
        features = self.norm_features.denorm(Normedfeatures)[:,:3]
        unNormed = oxides.clone()[:,:len(Elkeys)] # Ensure we don't grab the potentially empty ferric column
        #print(f"unNormed: {unNormed[:2]}")
        #print(f"Features: {features[3]}")
        fO2_composition_Nos = torch.tensor([oxide_dict[ox] for ox in ['Al2O3', 'FeO', 'CaO', 'Na2O', 'K2O']], device = self.dev) # For Feeding in relevant molar oxides for Kress and Carmichael, 
        fO2_composition_ind = torch.zeros(len(Elkeys), device = self.dev).to(torch.bool) # For Feeding in relevant molar oxides for Kress and Carmichael, 
        fO2_composition_ind[fO2_composition_Nos] = True
        row_sums = unNormed.sum(dim=1, keepdim=True) # Temporary renormalization for iron speciation equation
        nonzero_mask = row_sums != 0 # Replace zeros with 1 to avoid division by zero
        row_sums[~nonzero_mask] = 1.0
        temp_renorm = unNormed * (1 / row_sums)# Normalize to one oxide mole
        temp_renorm[~nonzero_mask.expand_as(unNormed)] = 0.0 # Set zero-sum rows to 0 to be extra careful
        #print(f"Temp renorm: {temp_renorm}")
        #print(f"Composition input: {temp_renorm[nonzero_mask.flatten()][:,fO2_composition_ind]}")
        # Get ferric/ferrous 
        IronR = Fe2O3_FeO_ratio(fO2=10**(QFM_fO2(K = features[nonzero_mask.flatten()][:,1]+273, P = features[nonzero_mask.flatten()][:,0], use_torch = True)+features[nonzero_mask.flatten()][:,2]), 
                                T=features[nonzero_mask.flatten()][:,1]+273, P=1e5*features[nonzero_mask.flatten()][:,0], composition=temp_renorm[nonzero_mask.flatten()][:,fO2_composition_ind], use_torch = True, device = self.dev)
        #print(f"IronR: {IronR}")
        ferricPerTot = 1/(2+(1/IronR))
        ferrousPerTot = 1/((2*IronR)+1)

        ferric = torch.zeros((unNormed.size()[0],1), device = self.dev, dtype = torch.float32) # Placeholder column to recieve ferric iron
        # Get indices where the mask is True
        idx = nonzero_mask.flatten().nonzero(as_tuple=True)[0]

        # Modify ferric[:, 0]
        ferric[idx, 0] = unNormed[idx, oxide_dict['FeO']] * ferricPerTot
        #print(f"Ferric: {ferric}")
        # Modify unNormed[:, FeO] in-place
        unNormed[idx, oxide_dict['FeO']] *= ferrousPerTot

        unNormed_out = torch.cat([unNormed,ferric], dim = 1) # moles oxides with Fe2O3
        #print(f"Out unNormed: {unNormed_out[:2]}")
        return unNormed_out

    #Add Only phasewise wt% Conversion,
    #nnLS slow
    #fast LS
    #Joint Optimization
    
    def batched_lstsq_masked(self, A, b, mask=None, rcond=1e-6):
        """
        Batched least squares with optional masking.
        A: (B, E, P)
        b: (B, E) or (B, E, 1)
        mask: (B, E) boolean, True=keep row, False=mask row
        Returns: x (B, P)
        """
        if b.ndim == 2:
            b = b.unsqueeze(-1)  # (B, E, 1)

        if mask is None:
            mask = (A.abs().sum(dim=2, keepdim=True) > 0).float()
            
        #mask = mask.unsqueeze(-1).to(A.dtype)  # (B, E, 1)
        print(mask.size())
        A = A * mask   # zero out masked rows
        b = b * mask   # zero out masked targets

        # Batched least squares on GPU
        sol = torch.linalg.lstsq(A, b, rcond=rcond).solution  # (B, P, 1)
        return sol.squeeze(-1)

    def masked_pinv_no_cf(self, A, b, rcond=1e-6):
        # A: (E, P), b: (E,)
        mask = (A.abs().sum(dim=2, keepdim=True) > 0).float()  # (E,1)
        A_masked = A * mask      # zero out invalid rows
        b_masked = b * mask.squeeze(-1)
        print(A_masked)
        print(b_masked)
        print(mask)
        pinvA = torch.linalg.pinv(A_masked, rcond=rcond)  # (P, E)
        return torch.einsum('bce,be->bc', pinvA, b_masked)

    def clamp_descent(self, Ac, b, x0c, steps=10, lr=1e-2, device="cuda"):
        """Projected gradient descent to push solution nonnegative.
        A: (B, E, P)
        b: (B, E)
        x0: (B, P)
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

        #print(f"Ac size: {Ac.size()}")
        #print(f"bc size: {bc.size()}")
        #print(f"x size: {x.size()}")
        #print((Ac @ x.unsqueeze(-1)).size())
        #print(f'Cuda handoff: {time.time()-inner_start}')
        for _ in range(steps):
            residual = b - (Ac @ x.unsqueeze(-1))  # (B, E, 1) bec,bc0->be0 or bep,bp0->be0
            grad = -(Ac.transpose(1, 2) @ residual).squeeze(-1)  # (B, P) bpe,be0->bp0
            x = x - (lr * grad)
            x = x.clamp_min(0.0)  # nonnegativity projection
        print(f'Descent Time: {time.time()-inner_start}')
        
        if pass_data:
            x = x.to(self.dev)

        return x
    
    def clamp_descent_newcomps(self, b, compToEl, newComps0, steps=10, lr=1e-2, device="cuda"):
        """
        Projected gradient descent to fit compositions (newComps) nonnegative.

        Args:
        bc        (B, E): bulk elements
        compToEl  (C, E): component -> element map
        newComps0 (B, C): initial guess of components
        steps: gradient descent steps
        lr: learning rate
        device: 'cuda' or 'cpu'

        Returns:
        newComps (B, C) nonnegative solution
        """

        start = time.time()
        if self.dev != device:
            pass_data = True
            bc = b.clone().to(device)                          # (B, E)
            x = compToEl.clone().to(device)                     # (C, E)
            Ac = newComps0.clone().to(device)           # (B, C)
        else:
            pass_data = False
            bc = b.clone()                        # (B, E)
            x = compToEl                 # (C, E)
            Ac = newComps0.clone() 

        zero_mask = torch.ones_like(Ac, device = device)
        zero_mask[Ac == 0] = 0
        
    
        for _ in range(steps):
            # residual = bc - (Ac @ x)   -> (B, E)
            residual = bc - (Ac @ x)

            # grad wrt Ac:   d||r||²/dAc = -2 * residual @ x^T
            grad = -(residual @ x.T)   # (B, C)
            grad *= zero_mask
            Ac = Ac - lr * grad
            Ac = Ac.clamp_min(0.0)     # enforce nonnegativity

        print(f"Clamp descent time: {time.time()-start:.4f}s")
        if pass_data:
            Ac = Ac.to(self.dev)

        return Ac

    def make_phase_tables(self, newComps, compToOx, MM, compPhaseMap, features, out = 'oxides', eps=1e-12):
        """
        Compute phase oxide wt% tables and phase mass fractions.

        newComps: (B, C)   - component abundances (moles)
        compToOx: (C, O)   - component->oxide stoichiometry
        MM:        (O, O)  - diagonal matrix of oxide molar masses
        compPhaseMap: (C, P) - component->phase membership (binary)

        Returns:
        phaseOxWt: (B, P, O)  - phase oxide wt% (normalized per phase)
        phaseMassNorm: (B, P) - phase masses normalized to 100% systemwide
        """
        #table_time = time.time()
        #B, C = newComps.shape
        #C2O = compToOx @ MM  # (C, O), grams oxide per mole of component

        # 1. Component -> Oxide conversion
        #oxideMass = newComps @ C2O  # (B, O)

        # 2. Expand phase map so we can split oxides into phases
        # compPhaseMap: (C, P), newComps: (B, C)
        phaseComps = newComps.unsqueeze(-1) * compPhaseMap # (B, C, P) # Components owned by phases

        """# 3. Convert to oxides per phase
        # (B,C,P) @ (C,O) -> (B,P,O)
        phaseOxMass = torch.einsum("bcp,co->bpo", compMass, C2O)"""

        # 3. Convert to oxides per phase (plug in iron speciator). Moles, then grams
        # (B,C,P) @ (C,O) -> (B,P,O)
        phaseOxMolar = torch.einsum("bcp,co->bpo", phaseComps, compToOx)
        print(phaseOxMolar[:3,-1])
        print(features[:3])
        liqWithFerric = self.Iron_Speciator(oxides=phaseOxMolar[:,-1].to(self.dev), Normedfeatures=features.to(self.dev))
        phaseOxMolar[:,-1] = liqWithFerric
        print(phaseOxMolar[:3,-1])


        phaseOxMass = torch.einsum("bpo,oo->bpo", phaseOxMolar, MM)

        #print(phaseOxWt[:3,-1])
        # 5. Compute total phase masses
        phaseMass = phaseOxMass.sum(dim=-1)  # (B,P)

        # 6. Normalize systemwide to 100%
        systemTotal = phaseMass.sum(dim=-1, keepdim=True)  # (B,1)
        phaseMassNorm = 100.0 * phaseMass / (systemTotal + eps)

        if out == 'oxides':
            # 4. Normalize oxide masses within each phase to 100%
            phaseSums = phaseOxMass.sum(dim=-1, keepdim=True)  # (B,P,1)
            
            phaseOxWt = 100.0 * phaseOxMass / (phaseSums + eps)

            return phaseOxWt[:,torch.tensor(compositionally_variable_binaries, dtype = torch.bool)], phaseMassNorm
        
        elif out in ['comps','components']:
            #Returns intensive, chemically variable components
                        # phaseComps: (B, C, P)
            # phaseToCompMap: (P, C)
            # 1. Normalize to intensive compositions
            phasesums =phaseComps.sum(dim=1, keepdim=True)  # (B, 1, P)
            phaseIntensive = phaseComps / (phasesums+1E-12)  # (B, C, P)

            # 2. Condense to system components
            print(phaseIntensive.size())
            systemComps = systemComps = torch.einsum('bcp,pc->bc', phaseIntensive, compPhaseMap.T)
  # (B, C)

            return systemComps, phaseMassNorm
        
        else:
            return phaseMassNorm


        

    def retrieveMassesFast(self, components, features, binaries, descent = False, pinv = False, verbose = False):
        """components is intensive component matrix. Binaries is saturation. No pure phases in component argument. Binaries""" 
        #NEED BINARY AND COMPOSITIONAL INDEXING MATRICES FOR BETTER SPEED. NEED one hots for: binaryToCompI, varyToCompI, compToPhaseI
        
        if self.dev == 'cuda' and not descent:
            fundev = 'cpu'
            print('converting...')
            self.compToEl=self.compToEl.to('cpu')
            self.compToOx=self.compToOx.to('cpu')
            self.oxToEl=self.oxToEl.to('cpu')
            components.to('cpu')
            features.to('cpu')
            binaries.to('cpu')
        else:
            fundev = self.dev

        
        start = time.time()
        nrows = components.size()[0]
        ncomps = label_indices['melts-liquid'][-1]+1
        nphases = mass_phasedict['melts-liquid']+1
        bulk = features[:,3:].clone()

        #Organize phaseToComp Matrix w/ NN output. 0s when absent, 1s, for present pure phases, and component fractions for present compositionally variable phases
        #NEXT INDEX WITH ONE-HOT MATRICES
        phaseToComp = torch.zeros((nrows,nphases,ncomps), device = fundev) #(B,P,C) Includes composition/phase present info
        phaseToCompMap = torch.zeros((nphases,ncomps), device = fundev) #(P,C) General

        for phase, binary_ind in mass_phasedict.items():
            phaseToCompMap[binary_ind, label_indices[phase]] = 1
            if len(label_indices[phase]) > 1:
                phaseToComp[:, binary_ind, label_indices[phase]] = components[:,label_indices_comp[phase]]

            else:
                phaseToComp[:, binary_ind, label_indices[phase]] = binaries[:, binary_ind].unsqueeze(-1)


        print(self.compToOx.size())
        print(self.compToOx.device)
        #Calculate phaseToEl/Ox matrix
        phaseToOx = torch.einsum('bpc,co->bpo', phaseToComp, self.compToOx)  # (B, P, O)
        phaseToEl = torch.einsum('bpo,oe->bpe', phaseToOx, self.oxToEl)  # (B, P, E)
       
        #molPhase = torch.linalg.lstsq(phaseToEl.transpose(1, 2), bulk.unsqueeze(-1)).solution.squeeze(-1)  # bep,bp0->be0 Overconstrained, slightly inconsistent. Will have residual
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

        # Now we relax the constant composition constraint to deal with the small residuals, but we weight perterbations to more common components
        residuals = (phaseToEl.transpose(1, 2) @ molPhase.unsqueeze(-1)).squeeze(-1) - bulk # bep bp0 -> be0 (b,e)
  
        #Nomenclature from Asimow and Ghiorso, 1998
        Msol = torch.diag_embed(torch.einsum('bp,bpc->bc', molPhase, phaseToComp)) # (b,c) -> (b,c,c) #Batchwise square diagonal matrix of system components, used as weights

        # Build weighted matrix
        A2 = (Msol @ self.compToEl).transpose(1,2)   # (B, E, C)
        res = residuals                              # (B, E)

        # Pseudoinverse: (B, C, E)
        #pinvA2 = torch.linalg.pinv(A2, rcond=1e-6)

        # Solve: (B, C, E) @ (B, E) -> (B, C)
        if pinv:
            wtDelComp = self.masked_pinv_no_cf(A2, res)
        else:
            wtDelComp = self.batched_lstsq_masked(A2, res)

        delComp =  torch.einsum('bcc,bc->bc', Msol, wtDelComp).squeeze(-1) #(b,c)
        oldComps = torch.einsum('bcp,bpZ->bcZ', phaseToComp.transpose(1,2), molPhase.unsqueeze(-1)).squeeze(-1)
        newComps = oldComps - delComp

        if descent:
            descend_time = time.time()
            """If doing gradient descent, need to refit residuals afterwards."""
            newComps1 = self.clamp_descent_newcomps(bulk, self.compToEl, newComps, steps=20, lr=1e-3, device=self.dev)
            newBulk1 = newComps1 @ self.compToEl
            res1 = newBulk1 - bulk

            Msol1 = torch.diag_embed(newComps1)
            A3 = (Msol1 @ self.compToEl).transpose(1,2)   # (B, E, C)
            if pinv:
                wtDelComp1 = self.masked_pinv_no_cf(A3, res1)
            else:
                wtDelComp1 = self.batched_lstsq_masked(A3, res1)
            DelComp1 =  torch.einsum('bcc,bc->bc', Msol1, wtDelComp1).squeeze(-1) #(b,c)
            newComps = newComps1 - DelComp1
            print(f'Time for second Descent and residual fitting: {time.time()-descend_time}')
        
        if self.dev == 'cuda' and not descent:
            self.compToEl=self.compToEl.to('cuda')
            self.compToEl=self.compToOx.to('cuda')
            self.compToEl=self.oxToEl.to('cuda')
        
        print(f'Total Linear Algebra Time: {time.time()-start} seconds')

        compTens, massTens = self.make_phase_tables(newComps, self.compToOx, self.MM, compPhaseMap=phaseToCompMap, features=features, eps=1e-12)
        compTens, massTens = compTens[:,self.model.comp_binaries].detach().cpu().numpy(), massTens.detach().cpu().numpy()
        return compTens, massTens
    
    def polish_negative_px(self, phaseProportions):
        # Check for below zero CaO
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
    
    def polish_negative_sp(self, phaseProportions):
        # Check for negative hercynite
        NegS = (phaseProportions[:,label_indices_comp['spinel'][2]] * (2/3)) + (phaseProportions[:,label_indices_comp['spinel'][4]] * (1/4))
        PosS =  (phaseProportions[:,label_indices_comp['spinel'][1]] ) + (phaseProportions[:,label_indices_comp['spinel'][3]])
        illegal = NegS > PosS
        if illegal.sum():  # Calculate scaling factors to zero out negative hercynite without affecting Chromite
            A =  (phaseProportions[illegal,label_indices_comp['spinel'][2]] ) + (phaseProportions[illegal,label_indices_comp['spinel'][4]])
            RemS = A + PosS[illegal] # equivalent to 1-chromite
            a = RemS / (A + NegS[illegal])
            b = (RemS * NegS[illegal]) / ( PosS[illegal] * (A + NegS[illegal]) )
            phaseProportions[illegal,label_indices_comp['spinel'][2]], phaseProportions[illegal,label_indices_comp['spinel'][4]] = a * phaseProportions[illegal,label_indices_comp['spinel'][2]], a * phaseProportions[illegal,label_indices_comp['spinel'][4]]
            phaseProportions[illegal,label_indices_comp['spinel'][1]], phaseProportions[illegal,label_indices_comp['spinel'][3]] =  b * phaseProportions[illegal,label_indices_comp['spinel'][1]], b * phaseProportions[illegal,label_indices_comp['spinel'][3]] 







    
    def polish_masses(self, phaseMoles, reconBulk, componentMoles, phaseProportions, features, optimize_masses = True, output_componentMoles= False, protect_opx = True, comp_table_out = 'oxides'):
        """components is an EXTENSIVE component matrix. Binaries is saturation. No pure phases in component argument. phaseMoles, 
        reconBulk, componentMoles, and phaseProportions are the outputs of *self.model.forward_masses(... details_out = True)
        IF YOU USE A GPU MODEL, THE OUTPUT WILL BE ON THE CPU, BECAUSE PSUEDO INVERSE IS OPTIMIZED FOR CPU
        optimize_masses adjusts masses of phases first to minimize the residual without changing the composition of any phases, so 
        it may lead to a better solution. However, It will nearly double the computation time.
        protect_olivine zeros out weights for olivine during residual fitting of components to maintain original chemistry predictions""" 
        #NEED BINARY AND COMPOSITIONAL INDEXING MATRICES FOR BETTER SPEED. NEED one hots for: binaryToCompI, varyToCompI, compToPhaseI

        phaseMoles, reconBulk, componentMoles, phaseProportions =  phaseMoles.to('cpu'), reconBulk.to('cpu'), componentMoles.to('cpu'), phaseProportions.to('cpu')
        feats = features.to('cpu')
        bulk = feats[:,3:]
        compToEl = self.compToEl.to('cpu')
        compToOx = self.compToOx.to('cpu')
        #oxToEl = self.oxToEl.to('cpu')
        phaseToCompMap = self.phaseToCompMap.to('cpu')
        MM = self.MM.to('cpu')

        
        """start = time.time()
        nrows = components.size()[0]
        ncomps = label_indices['melts-liquid'][-1]+1
        nphases = mass_phasedict['melts-liquid']+1"""
        

        residual = bulk - reconBulk
        if optimize_masses: # Initial overconstrained fit adjusting masses to help us down the line. This get us closer without changing chemical predictions, at the expense of time
            phaseComponentMoles = componentMoles[:, None, :] * phaseToCompMap[None, :, :] # (B, P, C) Extensive (weighted) Components Separated by phase
            phaseAtomMoles = torch.einsum('bpc,ce->bpe', phaseComponentMoles, compToEl) #(B,P,E)
            print(phaseAtomMoles[:3])
            wtDelPhaseMoles = (torch.linalg.pinv(phaseAtomMoles.transpose(1,2)) @ residual.unsqueeze(-1)).squeeze(-1) # (B,P) # Fit for delta multipliers for phases
            DelPhaseMoles = wtDelPhaseMoles * phaseMoles # Weights to Absolutes
            print('Before Phase Mass adjustments, phasemoles, then residual')
            print(phaseMoles)
            print(residual)
            phaseMoles = phaseMoles + DelPhaseMoles # Update Masses
            componentMoles = phaseProportions * (phaseMoles @ phaseToCompMap) # Update Extensive Component Moles.
            reconBulk = componentMoles @ compToEl # Update new reconstructed bulk
            residual = bulk - reconBulk # Update Residual
            print('After Phase Mass adjustments, phasemoles, then residual')
            print(phaseMoles)
            print(residual)
        
        # Now solve underconstrained problem to force the sum to bulk while doing the least damage to the NN solution, weight by abundance
        componentAtomMoles = componentMoles.unsqueeze(-1) * compToEl #(B,C,E)
        if protect_opx:
            componentAtomMoles[:,label_indices['orthopyroxene']] *= 0 # Zero Out OPX Weights
        wtDelComponentMoles = (torch.linalg.pinv(componentAtomMoles.transpose(1,2)) @ residual.unsqueeze(-1)).squeeze(-1) # (B,C) # Fit for Delta multipliers for components
        DelComponentMoles = wtDelComponentMoles * componentMoles # Weights to absolutes
        componentMoles = componentMoles + DelComponentMoles # Fix components to fit bulk perfectly"""
        if output_componentMoles:
            return self.make_phase_tables(componentMoles, compToOx, MM, compPhaseMap=phaseToCompMap.T, features=feats, eps=1e-12, out = comp_table_out), componentMoles, wtDelComponentMoles # Output as normalized composition and mass tables
        else:
            return self.make_phase_tables(componentMoles, compToOx, MM, compPhaseMap=phaseToCompMap.T, features=feats, eps=1e-12, out = comp_table_out) # Output as normalized composition and mass tables

        

        



        """x = (torch.linalg.pinv(A) @ b).squeeze(-1)

        #Organize phaseToComp Matrix w/ NN output. 0s when absent, 1s, for present pure phases, and component fractions for present compositionally variable phases
        #NEXT INDEX WITH ONE-HOT MATRICES
        phaseToComp = torch.zeros((nrows,nphases,ncomps), device = fundev) #(B,P,C) Includes composition/phase present info
        phaseToCompMap = torch.zeros((nphases,ncomps), device = fundev) #(P,C) General

        for phase, binary_ind in mass_phasedict.items():
            phaseToCompMap[binary_ind, label_indices[phase]] = 1
            if len(label_indices[phase]) > 1:
                phaseToComp[:, binary_ind, label_indices[phase]] = components[:,label_indices_comp[phase]]

            else:
                phaseToComp[:, binary_ind, label_indices[phase]] = binaries[:, binary_ind].unsqueeze(-1)


        print(self.compToOx.size())
        print(self.compToOx.device)
        #Calculate phaseToEl/Ox matrix
        phaseToOx = torch.einsum('bpc,co->bpo', phaseToComp, self.compToOx)  # (B, P, O)
        phaseToEl = torch.einsum('bpo,oe->bpe', phaseToOx, self.oxToEl)  # (B, P, E)
       
        #molPhase = torch.linalg.lstsq(phaseToEl.transpose(1, 2), bulk.unsqueeze(-1)).solution.squeeze(-1)  # bep,bp0->be0 Overconstrained, slightly inconsistent. Will have residual
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

        # Now we relax the constant composition constraint to deal with the small residuals, but we weight perterbations to more common components
        residuals = (phaseToEl.transpose(1, 2) @ molPhase.unsqueeze(-1)).squeeze(-1) - bulk # bep bp0 -> be0 (b,e)
  
        #Nomenclature from Asimow and Ghiorso, 1998
        Msol = torch.diag_embed(torch.einsum('bp,bpc->bc', molPhase, phaseToComp)) # (b,c) -> (b,c,c) #Batchwise square diagonal matrix of system components, used as weights

        # Build weighted matrix
        A2 = (Msol @ self.compToEl).transpose(1,2)   # (B, E, C)
        res = residuals                              # (B, E)

        # Pseudoinverse: (B, C, E)
        #pinvA2 = torch.linalg.pinv(A2, rcond=1e-6)

        # Solve: (B, C, E) @ (B, E) -> (B, C)
        if pinv:
            wtDelComp = self.masked_pinv_no_cf(A2, res)
        else:
            wtDelComp = self.batched_lstsq_masked(A2, res)

        delComp =  torch.einsum('bcc,bc->bc', Msol, wtDelComp).squeeze(-1) #(b,c)
        oldComps = torch.einsum('bcp,bpZ->bcZ', phaseToComp.transpose(1,2), molPhase.unsqueeze(-1)).squeeze(-1)
        newComps = oldComps - delComp

        if descent:
            descend_time = time.time()
            #If doing gradient descent, need to refit residuals afterwards.
            newComps1 = self.clamp_descent_newcomps(bulk, self.compToEl, newComps, steps=20, lr=1e-3, device=self.dev)
            newBulk1 = newComps1 @ self.compToEl
            res1 = newBulk1 - bulk

            Msol1 = torch.diag_embed(newComps1)
            A3 = (Msol1 @ self.compToEl).transpose(1,2)   # (B, E, C)
            if pinv:
                wtDelComp1 = self.masked_pinv_no_cf(A3, res1)
            else:
                wtDelComp1 = self.batched_lstsq_masked(A3, res1)
            DelComp1 =  torch.einsum('bcc,bc->bc', Msol1, wtDelComp1).squeeze(-1) #(b,c)
            newComps = newComps1 - DelComp1
            print(f'Time for second Descent and residual fitting: {time.time()-descend_time}')
        
        if self.dev == 'cuda' and not descent:
            self.compToEl=self.compToEl.to('cuda')
            self.compToEl=self.compToOx.to('cuda')
            self.compToEl=self.oxToEl.to('cuda')
        
        print(f'Total Linear Algebra Time: {time.time()-start} seconds')

        compTens, massTens = self.make_phase_tables(newComps, self.compToOx, self.MM, compPhaseMap=phaseToCompMap, features=features, eps=1e-12)
        compTens, massTens = compTens[:,self.model.comp_binaries].detach().cpu().numpy(), massTens.detach().cpu().numpy()
        return compTens, massTens"""


        """Need batchwise oxides per component (bco) which can get molar

        newMolPhase = newComps 

        ## Okay solve (Msol@compToEl)x = residuals , multiply x by Msol to get vector of component deltas to get rid of residual. Recalc phase abundances. 

        molOx = torch.einsum('bp,bpo->bpo', molPhase, phaseToOx) # Weighted mole oxides per phase

        gOx = molOx @ self.MM # broadcast over last dim (B, P, O) #grams oxidews per phase

        # Step 4: Normalize to weight percent
        total_gOx = gOx.sum(dim=-1)  # (B, P, 1)
        compTens = 100 * gOx / total_gOx.unsqueeze(-1)  # (B, P, O)
        
        batch_mass = total_gOx.sum(dim=-1, keepdim = True)

        massTens = 100 * total_gOx / batch_mass

        return massTens, compTens"""
    
        """ # Step 1: Moles of components in each phase
        molComp = torch.einsum('bp,bpc->bpc', molPhase, phaseToComp)  # (B, P, C)

        # Step 2: Moles of oxides
        molOx = torch.einsum('bpc,co->bpo', molComp, CompToOx)  # (B, P, O)

        # Step 3: Convert to grams — use only diagonal of MM
        molar_masses = torch.diagonal(MM)  # (O,)
        gOx = molOx * molar_masses  # broadcast over last dim (B, P, O)

        # Step 4: Normalize to weight percent
        total_gOx = gOx.sum(dim=-1, keepdim=True)  # (B, P, 1)
        compTens = 100 * gOx / total_gOx  # (B, P, O)

        
       

        massTens = torch.zeros((nrows, len(all_phases)), device = self.dev)
        compTens = torch.zeros((nrows, len(compositionally_variable_phases), len(Oxides)), device =self.dev)
        
        for phase in all_phases:
            if phase != 'melts-liquid':
                massTens[:,mass_phasedict[phase]] = components[:,label_indices[phase]] @ self.compToOx[label_indices[phase]] @ self.Mtot # Not Normalized!
                if phase in compositionally_variable_phases: #Variable componsition. 
                    unNormed = components[:,label_indices[phase]] @ self.compToOx[label_indices[phase]] @ self.MM
                    row_sums = unNormed.sum(dim=1, keepdim=True)
                    nonzero_mask = (row_sums != 0) # Replace zeros with 1 to avoid division by zero
                    row_sums[~nonzero_mask] = 1.0
                    renormed = unNormed * (100.0 / row_sums)# Normalize to 100 wt%
                    renormed[~nonzero_mask.expand_as(unNormed)] = 0.0 # Set zero-sum rows to 0 to be extra careful
                    compTens[:, comp_phasedict[phase], :] = renormed 
            else: 
                #For the liquid, We go to moles oxide and then speciate the iron between ferric and ferrous based on fO2.
                #print(f"Element components: {components[:,label_indices[phase]]}")
                unNormed = components[:,label_indices[phase]] @ self.elToOx # mole oxides, no ferric iron
                unNormed = self.Iron_Speciator(unNormed, features)  # mole oxides w/ ferric iron

                massTens[:,mass_phasedict[phase]] = unNormed @ self.Mtot # Not Normalized!
        
                unNormedWt = (unNormed @ self.MM) 
                row_sums = unNormedWt.sum(dim=1, keepdim=True) # Now Renormalizing for intensive mass: 100 wt%
                #print(f"Row sums: {row_sums[:2]}")
                nonzero_mask = row_sums != 0
                #print(f"Nonzero mask: {nonzero_mask}")
                renormed = unNormedWt * (100.0 / row_sums)# Normalize to 100 wt%
                renormed[~nonzero_mask.expand_as(unNormed)] = 0.0 # Set zero-sum rows to 0 to be extra careful
                #print(f"renormed to 100 after speciation: {renormed}")
                compTens[:, comp_phasedict[phase], :] = renormed #Finally
        
        row_sums = massTens.sum(dim=1, keepdim=True) # Normalize Mass Table to 100 wt
        massTens = massTens * (100.0 / row_sums)
        
        return massTens, compTens"""

        
    def find_liquidus(self, features, resolution = 25):
        """Returns lowest identified superliquidus temperature between 800 and 2000 C"""
        T_test = torch.tensor(np.linspace(800,2000,int(1200/resolution)+1), device = self.dev)
        feat_input = np.zeros((int(1200/resolution)+1,len(features)))
        feat_input[:] = features
        feat_input[:,1] = T_test
        binaries = self.forward_binary(torch.tensor(feat_input, device = self.dev)>0.5).float()
        liquids = binaries[:,:-1].sum(dim = 1) == 0
        lowL = torch.where(liquids)[0]
        if len(lowL):
            temp = T_test[lowL[0]]
        else:
            temp = 2000
        return temp
    
    def find_liquidi(self, features, resolution=25, weightOxinput = False):
        """Vectorized version.
        features_batch: torch.Tensor of shape [N, F], where F >= 2 and column 1 is temperature.
        Returns: torch.Tensor of shape [N] with liquidus temperatures for each composition.
        """
        if weightOxinput:
            features_batch = self.convertOxToMol(features)
            #print(features_batch)
        else:
            features_batch = features
        N, F = features_batch.shape
        T_test = torch.linspace(800, 2000, steps=int(1200 / resolution) + 1)  # [T]
        n_temps = T_test.shape[0]

        # Expand the feature set across all T values for each sample
        feat_input = features_batch.unsqueeze(1).repeat(1, n_temps, 1)  # [N, T, F]
        feat_input[:, :, 1] = T_test.unsqueeze(0).repeat(N, 1)  # Fill temperature column
        feat_input = feat_input.view(N * n_temps, F)  # [N*T, F]

        with torch.no_grad():
            binaries = (self.forward_binary(feat_input) > 0.5).float()  # [N*T, P]
        
        # Check for fully liquid (no solid phases)
        fully_liquid = binaries[:, :-1].sum(dim=1) == 0  # [N*T]
        fully_liquid = fully_liquid.view(N, n_temps)  # [N, T]

        # Find first fully liquid temperature for each composition
        first_liquid_idx = torch.argmax(fully_liquid.to(torch.int), dim=1)
        has_liquid = fully_liquid.any(dim=1)  # [N]

        # Convert indices to temperatures
        liquidus_temperatures = T_test[first_liquid_idx]  # [N]
        liquidus_temperatures[~has_liquid] = 2000.0  # Default max if none fully liquid

        return liquidus_temperatures
    


    def find_mineral_cosaturation(self, features, T_initial_C, phase_cols, dt_C=1, weightOxinput=False):
        """Vectorized version.
        features: torch.Tensor of shape [N, F], where F >= 2 and column 1 is temperature.
        phase_cols: list of phase column indices (integers).
        Returns: torch.Tensor of shape [N, Pc] with saturation temperatures for each phase.
        """
        if weightOxinput:
            features_batch = self.convertOxToMol(features)
        else:
            features_batch = features

        N, F = features_batch.shape
        T_test = torch.linspace(700, T_initial_C, steps=int((T_initial_C - 700) / dt_C) + 1, device=features.device)  # [T]
        n_temps = T_test.shape[0]

        # Expand the feature set across all T values for each sample
        feat_input = features_batch.unsqueeze(1).repeat(1, n_temps, 1)  # [N, T, F]
        feat_input[:, :, 1] = T_test.unsqueeze(0).repeat(N, 1)  # Fill temperature column
        feat_input = feat_input.view(N * n_temps, F)  # [N*T, F]

        with torch.no_grad():
            binaries = (self.forward_binary(feat_input) > 0.5).float()  # [N*T, P]

        binaries = binaries[:, phase_cols]  # [N*T, Pc]
        binaries = binaries.view(N, n_temps, -1)  # [N, T, Pc]

        # Mask temperatures where the phase is present
        T_masked = T_test.view(1, n_temps, 1) * binaries  # [N, T, Pc]

        # Find the highest temperature where the binary == 1
        saturation_temps, _ = T_masked.max(dim=1)  # [N, Pc]

        return saturation_temps
    


    def fractional_crystalization(self, features, T_path, fit_residual = True, WtPercent=True):
        """
        Args:
            inp_tensor: Tensor of shape (nB, n_features), where
                        - [:,0] = pressure
                        - [:,1] = temperature (to be updated each step)
                        - [:,2] = log fo2 delta QFM
                        - [:,3:] = elemental composition (normalized to 1)
            T_path: Sequence of temperatures to iterate through (list or 1D tensor)
        
        Returns:
            melt_fracs: (nB, nSteps)
            active_mask: (nB, nSteps)
            Can also skip residual fitting for faster calculation that doesn't satisfy that mass balance. 
        """
        with torch.no_grad():

            if WtPercent:
                inp_tensor = self.convertOxToMol(features)
            else:
                inp_tensor = features

            nB = inp_tensor.size(0)     # number of bulk compositions
            nEl = self.oxToEl.size(1)
            nC = label_indices['melts-liquid'][-1]+1 # number of components in chemically variable phases
            nP = len(list(label_indices.keys())) # Number of phases
            nSteps = len(T_path)

            

            # Init outputs 
            component_tensor = torch.zeros((nB, nC, nSteps), dtype=torch.float32, device=self.dev)
            mass_tensor = torch.zeros((nB, nP, nSteps), dtype=torch.float32, device=self.dev) # Let this mass tensor carry information about the total mass too. Total mass equals previous step's liquid mass.

            # Track which bulk comps are still active
            is_alive = torch.ones(nB, dtype=torch.bool, device=self.dev)
            prev_melt_frac = torch.ones(nB, dtype=torch.float32, device=self.dev)

            # Keep full copy of evolving liquid compositions
            active_comps = inp_tensor[:, 3:].clone().to(self.dev)
            #print(active_comps)
            for i, temp in enumerate(T_path):
                print(temp)
                if not is_alive.any():
                    print('All systems froze!')
                    break  # all systems frozen
                #print(f"Still alive: {is_alive.sum()}")
                #print(active_comps)
                # Get indices of still-active systems
                idx_alive = torch.nonzero(is_alive, as_tuple=True)[0]
                #current_comps = active_comps[idx_alive]

                # Formulate step input of alive compositions
                inp_batch = inp_tensor[idx_alive]
                inp_batch[:,1] = temp # Perhaps could be restructed to interface with any model where second variable is changed during fractionation
                inp_batch[:,3:] = active_comps
                #print(f"INPUT BATCH: {inp_batch}")
                #print(f"inp_batch shape {inp_batch.size()}")
                #print(f"active_comps shape: {active_comps.size()}")

                # Add sanity checks before norm call
                if inp_batch.numel() == 0:
                    raise RuntimeError("inp_batch is empty before normalization.")

                #if inp_batch.shape[1] < self.miner.shape[0]:
                #    raise RuntimeError(f"Expected input with at least {self.miner.shape[0]} columns, got {inp_batch.shape[1]}")


                # Run equilibrium calculation
                #cycle_components, solidOx = self.forward(inp_batch, solidOxOut=True)
                print(inp_batch)
                print(idx_alive)
                _, transcomponent_hat, phaseMoles, reconBulk, componentMoles, phaseProportions = self.model.forward(self.norm_features.norm(inp_batch), detailed=True)#, Normalize = False) # Running NN!
                if fit_residual:
                    #try:
                    massTens, componentMoles2, _ = self.polish_masses(phaseMoles, reconBulk, componentMoles, phaseProportions, 
                                                                        features=self.norm_features.norm(inp_batch), optimize_masses=False,  protect_opx=True, comp_table_out='None',  output_componentMoles= True)
                    """except:
                        print(f"Nonfinite phaseMoles: {torch.isinf(phaseMoles).sum()}")
                        print(f"Nonfinite componentMoles: {torch.isinf(componentMoles).sum()}")
                        print(f"Nonfinite phaseProportions: {torch.isinf(phaseProportions).sum()}")
                        print(f"Nonfinite reconBulk: {torch.isinf(reconBulk).sum()}")


                              
                        print("ABORTING EARLY; PROBLEM WITH RESIDUAL FITTING.")
                        break"""
                    component_tensor[idx_alive,:,i] = componentMoles2
                    mass_tensor[idx_alive,:,i] = massTens * prev_melt_frac[idx_alive,None]
                    
                else:
                    massTens = self.make_phase_tables(newComps=componentMoles, compToOx=self.compToOx, MM=self.MM, compPhaseMap=self.phaseToCompMap.T, features=self.norm_features.norm(inp_batch), eps=1e-12, 
                                                    out = None) # Hopefully this runs on GPU fine
                    component_tensor[idx_alive,:,i] = componentMoles
                    mass_tensor[idx_alive,:,i] = massTens * prev_melt_frac[idx_alive,None]
                    new_liquid_el = transcomponent_hat[:,-nEl].clone()


                #inst_melt_frac = massTens[:,mass_phasedict['melts-liquid']] # by mass
                #print(cycle_components.size())
                #new_liquid_el = cycle_components[:, -nEl:]
                #print(cycle_components.size())

                # Estimate melt oxides
                #liquidOx_guess = new_liquid_el @ self.elToOx  # (n_alive, nOx)
                #liquidOx = self.Iron_Speciator(liquidOx_guess, inp_batch)

                # Melt fraction by mass
                #melt_mass = (liquidOx @ self.Mtot).squeeze(-1)
                #total_mass = (liquidOx + solidOx) @ self.Mtot
                #otal_mass = total_mass.squeeze(-1)
                #inst_melt_frac = melt_mass / total_mass

                # Update melt tracking
                #curr_melt_frac = inst_melt_frac * prev_melt_frac[idx_alive]
                prev_melt_frac = (mass_tensor[:,-1,i].clone())/100 # between 0 and 1

                # Get Boolean for freezing
                is_alive = prev_melt_frac > 0.005 #0.5 wt percent liquid

                #update composition for next step
                new_liquid_el = component_tensor[:,-nEl:,i].clone()
                active_comps = new_liquid_el[is_alive].clone()

            return component_tensor, mass_tensor
    
Emulator102GPU_Cr = NN_MELTS(GPUFullMELTS_Cr, cuda = True)
Emulator102GPU_NoCr = NN_MELTS(GPUFullMELTS_NoCr, cuda = True)

Emulator102CPU_Cr = NN_MELTS(CPUFullMELTS_Cr, cuda = False)
Emulator102CPU_NoCr = NN_MELTS(CPUFullMELTS_NoCr, cuda = False)

"""class fxout():
    def __init__(self, extCompTens):
        Input is extensive component tensor of shape (B,V,S)
        B: Number of parallel compositions/conditions
        V: All Variable components"""