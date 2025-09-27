import ensemble_MELTSV2 # The essential ensemble MELTS functions
import numpy as np
import random
import pickle
import os   
import string
import time

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, Dropout, BCELoss, Sigmoid, MSELoss
from torch.optim import Adam, SGD
#import torchvision.transforms as T #THIS ONE IS BROKEN 04/04/2025
import torch.nn as nn

import MELTSEmulator


#PhaseSatNN = MELTSEmulator.PhaseSaturationHybridNetOFP(input_dim=15).cpu().eval()
#PhaseSatNN.load_state_dict(torch.load('rhyoliteMELTS1.0.2_BinaryPhaseSatOFP_May22.pt', map_location = torch.device('cpu')))
#MassPartitionNN = MELTSEmulator.MolePartitioningNetOFP().cpu().eval()
#MassPartitionNN.load_state_dict(torch.load('rhyoliteMELTS1.0.2_MolePartitionOFP_May22.pt', map_location = torch.device('cpu')))
#Emulator = MELTSEmulator.NN_MELTS(PhaseSatNN, MassPartitionNN)

allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet','nepheline','leucite','biotite','rhm-oxide','alloy-solid','apatite','whitlockite','quartz','tridymite','cristobalite','muscovite','fluid','liquid']


def random_char(y):
       return ''.join(random.choice(string.ascii_letters) for x in range(y))

def pull_number(string):
    string_number = ''
    for char in string:
        if char in '1234567890.-':
            string_number += char
    try:
        return float(string_number)
    except: 
        return np.nan

#chem_ind = ensemble_MELTS.chem_ind
phase_ind = pickle.load(open('PhaseDict.pkl', 'rb'))
#LEPR = pickle.load(open('liquids2.pkl', 'rb')) 

Out_Folder = 'GEOROC_SIMS'

alphaMELTSLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alphamelts')
# Location to where to put the computed files.
EnsembleLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)), Out_Folder)

if Out_Folder not in os.listdir():
    os.makedirs(EnsembleLocation)

#keys = ['Pressure', 'Temperature', 'fO2', 'SiO2', 'Al2O3', 'CaO', 'MgO', 'Na2O', 'K2O', 'Fe2O3', 'FeO', 'TiO2', 'MnO', 'Cr2O3', 'NiO', 'CoO', 'P2O5', 'H2O']
keys = ['Pressure', 'Temperature', 'fO2', 'SiO2','TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'MnO', 'H2O', 'Cr2O3', 'NiO']

key_dict = {}
for i, k in enumerate(keys):
    key_dict[k] = i
#print(LEPR[:,0,2])



# NOW USE ONLY COMPOSITIONS ALREADY USED, SAVE VALIDATION SET
#GEOROC = np.genfromtxt('GEOROC_WHOLEROCK.csv', delimiter=',',skip_header=1)
#full_indices = np.genfromtxt('GEOROC_WHOLEROCK.csv', delimiter=',',skip_header=1, dtype = str)[:,0]

#GEOROC = np.genfromtxtsource('GEOROC_training.csv', delimiter=',',skip_header=1)
#full_indices = np.genfromtxt('GEOROC_training.csv', delimiter=',',skip_header=1, dtype = str)[:,0]

"""mafics = GEOROC[:,5]>=5 # MgO above 5
full_indices = full_indices[mafics]

GEOROC = GEOROC[mafics] ### TEMP: MAFICS ONLY TO BALANCE DATASET"""

#print(GEOROC)

simcycle = 50

def alphaMELTScompress(output_file, iter = 750, fxtal = False):

    def PTF_initialize(conditions, length = int(40)): # ONLY CRUSTAL UP TO 1.5 GPa !!!!
        out_array = np.zeros((length, np.shape(conditions)[1] + 3))
        out_array[:,0] = np.random.uniform(1,21, size = length)#COMPRESSION
        #out_array[:,0] = np.random.uniform(1,15000, size = length)#COOLING # Pressure in bars
        #out_array[:,0] = np.concatenate((np.random.uniform(1,15000, size = int(length/2)), np.random.uniform(3000,15000, size = int(length/2)))) # Pressure in bars
        #out_array[:,1] = 1000 # T in K, Initial T. if end = True for forward ensemble function, minimum temp = this number, max 2000.
        out_array[:,2] = np.random.uniform(-5, 5, size = length) # fo2 in log offset from FMQ
        out_array[:,3:] = conditions
        #temps = Emulator.find_liquidi(torch.tensor(out_array, dtype = torch.float),resolution=25, weightOxinput = True).detach().numpy()#COOLING
        #print(temps)
        #out_array[:,1] = temps + np.array([10,11,12,13,14,15,16,17,18,19,20,21])#COOLING
        out_array[:,1] = np.random.uniform(700,2000, size = length)#COMPRESSION
        return out_array
    j = 0
    while j < iter: #For 12 sims, ~1200 assemblages or ~1kb per iteration (Can't be right... Maybe 100 kb?) (total reps train 1750, 1250 Validation)
        choices = np.random.randint(np.shape(GEOROC)[0], size=simcycle, dtype=int)
        compositions = GEOROC[choices, 1:]
        indices = full_indices[choices]
        print(compositions)
        #print(compositions[0])
        anhydrous = np.random.randint(simcycle, high=None, size=int(simcycle/4))
        #NoMnO = np.random.randint(simcycle, high=None, size=int(simcycle/3))
        #NoNiO = np.random.randint(simcycle, high=None, size=int(simcycle/3))
        compositions[anhydrous,-3] = 0
        too_wet = compositions[:,-3] > 5
        compositions[too_wet,-3] = 5
        #smallMnO = compositions[:,-4] < 0.1
        #compositions[smallMnO,-4] = 0 # Small MnO to zero
        #compositions[NoMnO,-4] = 0 # 1/3 MnO to zero
        #compositions[NoNiO,-1] = 0 # 1/3 NiO to zero
        compositions[:,-4] = 0 #MnO to zero
        compositions[:,-1] = 0 #NiO to zero
        #compositions[~smallMnO,-1] += 0.005 #if manganese in bulk comp, sprinkle in some nickel so olivine will precipitate
        smallCr2O = compositions[:,-2] < 0.01
        compositions[smallCr2O, -2] = 0 # SMall Cr2O3 to 0. Mostly for felsic compositions to prevent unrealistic chromites
        compositions = 100 * compositions / (np.sum(compositions, axis=1))[:, np.newaxis]  #Enforce chemical sum to 100%


        in_array = np.round(PTF_initialize(compositions, length = simcycle),2)
        #mantle = np.array(in_array[:,0] > 10000)
        #print(mantle)
        batchname = np.empty(simcycle, dtype=object)
        #batchname[mantle] = 'pMELTS'
        #batchname[~mantle] = 'Crustal'
        batchname[:] = 'Crustal' # ONLY CRUSTAL UP TO 1.5 GPa !!!!


        ensemble_MELTSV2.forward_ensemble(in_array, keys, batchname = batchname, only_phases=allowed_phases, end = 12000+in_array[:,0], fxtal = fxtal, EnsembleLocation=EnsembleLocation, WSL = True, compression=True, delta = 12000/200)

        #PTID = np.random.randint(len(batchname)) 
        for i, name in enumerate(batchname):

            batchname[i] = f"{pull_number(str(indices[i]))}:{random_char(4)}:{name}" # Put PTX Index in metadata to link double detected phases for quality control

        
        faultIDs = ensemble_MELTSV2.import_MELTS_components(EnsembleLocation=EnsembleLocation, batchname=batchname, fO2Arr=in_array[:,2], dataname = f'{output_file}.csv')
        ensemble_MELTSV2.pick_exsolution_failure(EnsembleLocation=EnsembleLocation, input_array=in_array, keys=keys, batchname=batchname, dataname = f'{output_file}_Exfail.csv', faultIDs=faultIDs)
        j += 1

def alphaMELTScooling(output_file, iter = 750, fxtal = False):

    def PTF_initialize(conditions, length = int(40)): # ONLY CRUSTAL UP TO 1.5 GPa !!!!
        out_array = np.zeros((length, np.shape(conditions)[1] + 3))
        #out_array[:,0] = np.random.uniform(1,21, size = length)#COMPRESSION
        out_array[:,0] = np.random.uniform(1,12000, size = length)#COOLING # Pressure in bars
        #out_array[:,0] = np.concatenate((np.random.uniform(1,15000, size = int(length/2)), np.random.uniform(3000,15000, size = int(length/2)))) # Pressure in bars
        #out_array[:,1] = 1000 # T in K, Initial T. if end = True for forward ensemble function, minimum temp = this number, max 2000.
        out_array[:,2] = np.random.uniform(-5, 5, size = length) # fo2 in log offset from FMQ
        out_array[:,3:] = conditions
        temps = 1925
        #temps = Emulator.find_liquidi(torch.tensor(out_array, dtype = torch.float),resolution=25, weightOxinput = True).detach().numpy()#COOLING
        #print(temps)
        out_array[:,1] = temps +20+ np.arange(length)#COOLING
        #out_array[:,1] = np.random.uniform(700,1600, size = length)#COMPRESSION
        return out_array
    j = 0
    while j < iter: #For 12 sims, ~1200 assemblages or ~1kb per iteration (Can't be right... Maybe 100 kb?) (total reps train 1750, 1250 Validation)
        choices = np.random.randint(np.shape(GEOROC)[0], size=simcycle, dtype=int)
        compositions = GEOROC[choices, 1:]
        indices = full_indices[choices]
        print(compositions)
        #print(compositions[0])
        anhydrous = np.random.randint(simcycle, high=None, size=int(simcycle/4))
        #NoMnO = np.random.randint(simcycle, high=None, size=int(simcycle/3))
        #NoNiO = np.random.randint(simcycle, high=None, size=int(simcycle/3))
        compositions[anhydrous,-3] = 0
        too_wet = compositions[:,-3] > 5
        compositions[too_wet,-3] = 5
        #smallMnO = compositions[:,-4] < 0.1
        #compositions[smallMnO,-4] = 0 # Small MnO to zero
        #compositions[NoMnO,-4] = 0 # 1/3 MnO to zero
        #compositions[NoNiO,-1] = 0 # 1/3 NiO to zero
        compositions[:,-4] = 0 #MnO to zero
        compositions[:,-1] = 0 #NiO to zero
        #compositions[~smallMnO,-1] += 0.005 #if manganese in bulk comp, sprinkle in some nickel so olivine will precipitate
        smallCr2O = compositions[:,-2] < 0.01
        compositions[smallCr2O, -2] = 0 # SMall Cr2O3 to 0. Mostly for felsic compositions to prevent unrealistic chromites
        compositions = 100 * compositions / (np.sum(compositions, axis=1))[:, np.newaxis]  #Enforce chemical sum to 100%


        in_array = np.round(PTF_initialize(compositions, length = simcycle),2)
        #mantle = np.array(in_array[:,0] > 10000)
        #print(mantle)
        batchname = np.empty(simcycle, dtype=object)
        #batchname[mantle] = 'pMELTS'
        #batchname[~mantle] = 'Crustal'
        batchname[:] = 'Crustal' # ONLY CRUSTAL UP TO 1.5 GPa !!!!


        ensemble_MELTSV2.forward_ensemble(in_array, keys, batchname = batchname, only_phases=allowed_phases, end = 700,  fxtal = fxtal, EnsembleLocation=EnsembleLocation, WSL = True, compression=False, delta = -1)

        #PTID = np.random.randint(len(batchname)) 
        for i, name in enumerate(batchname):

            batchname[i] = f"{pull_number(str(indices[i]))}:{random_char(4)}:{name}" # Put PTX Index in metadata to link double detected phases for quality control

        
        faultIDs = ensemble_MELTSV2.import_MELTS_components(EnsembleLocation=EnsembleLocation, batchname=batchname, fO2Arr=in_array[:,2], dataname = f'{output_file}.csv')
        ensemble_MELTSV2.pick_exsolution_failure(EnsembleLocation=EnsembleLocation, input_array=in_array, keys=keys, batchname=batchname, dataname = f'{output_file}_Exfail.csv', faultIDs=faultIDs)
        j += 1

#Out_file_comp_train = 'MELTS_TrainsetJuly7MnNiFree_Compression'
Out_file_cool_train = 'MELTS_TrainsetSept25FxtalCooling'
#Out_file_comp_valid = 'MELTS_ValidsetJuly7MnNiFree_Compression'
Out_file_cool_valid = 'MELTS_ValidsetSept25FxtalCooling'

"""print('Waiting...')
time.sleep(0.5*3600) # Delay by 30 min"""


GEOROC = np.genfromtxt('GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv', delimiter=',',skip_header=1)
full_indices = np.genfromtxt('GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv', delimiter=',',skip_header=1, dtype = str)[:,0]

#alphaMELTScompress(output_file=Out_file_comp_train, iter = 250)
alphaMELTScooling(output_file=Out_file_cool_train, iter = (50-30), fxtal=True)

mafics = GEOROC[:,5]>=5 # MgO above 5
full_indices = full_indices[mafics]

GEOROC = GEOROC[mafics] ### TEMP: MAFICS ONLY TO BALANCE DATASET"""

#alphaMELTScompress(output_file=Out_file_comp_train, iter = 200)
alphaMELTScooling(output_file=Out_file_cool_train, iter = 250, fxtal=True)
#alphaMELTScooling(output_file=Out_file_cool_train, iter = 1)



GEOROC = np.genfromtxt('GEOROC_PETDB_UNFILTERED_WHOLEROCK_VALIDATION.csv', delimiter=',',skip_header=1)
full_indices = np.genfromtxt('GEOROC_PETDB_UNFILTERED_WHOLEROCK_VALIDATION.csv', delimiter=',',skip_header=1, dtype = str)[:,0]

#alphaMELTScompress(output_file=Out_file_comp_valid, iter = 125)
alphaMELTScooling(output_file=Out_file_cool_valid, iter = 15, fxtal=True)


mafics = GEOROC[:,5]>=5 # MgO above 5
full_indices = full_indices[mafics]

GEOROC = GEOROC[mafics] ### TEMP: MAFICS ONLY TO BALANCE DATASET"""

#alphaMELTScompress(output_file=Out_file_comp_valid, iter = 100)
#alphaMELTScooling(output_file=Out_file_cool_valid, iter = 50, fxtal=True)
alphaMELTScooling(output_file=Out_file_cool_valid, iter = 75, fxtal=True)

#alphaMELTScooling(output_file=Out_file_cool_valid, iter = 5)
