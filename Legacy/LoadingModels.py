# Define nMELTS models
import BackEnds.nnMELTS as NN
import numpy as np
import torch

"""
Put all models in dictionaries, automate model choice based on arguments. Inputs assumed to be weight percent oxides, probably
Probably won't need to change. 
Functionalities to add:

"""

MELTSModel= '102'
CalcType = 'Fxtal'
date = 'Oct11'

DictFilePaths=[f"Models/MELTS{MELTSModel}{CalcType}{['NoCr', 'Cr'][i]}_Final_{date}.pt" for i in range(2)]

Fx102GPUFullMELTS_Cr = NN.MidLevelNetwork().cuda()
Fx102GPUFullMELTS_Cr.load_state_dict(torch.load(DictFilePaths[1])) 

Fx102GPUFullMELTS_NoCr = NN.MidLevelNetwork()
Fx102GPUFullMELTS_NoCr.load_state_dict(torch.load(DictFilePaths[0]))

Fx102CPUFullMELTS_Cr = NN.MidLevelNetwork().cuda()
Fx102CPUFullMELTS_Cr.load_state_dict(torch.load(DictFilePaths[1]), strict = False)

Fx102CPUFullMELTS_NoCr = NN.MidLevelNetwork()
Fx102CPUFullMELTS_NoCr.load_state_dict(torch.load(DictFilePaths[0]), strict = False)

FxEmulatorGPU_Cr_102 = NN.NN_MELTS(102GPUFullMELTS_Cr, cuda = True)
FxEmulatorGPU_NoCr_102 = NN.NN_MELTS(102GPUFullMELTS_NoCr, cuda = True)
FxEmulatorCPU_Cr_102 = NN.NN_MELTS(102CPUFullMELTS_Cr, cuda = False)
FxEmulatorCPU_NoCr_102 = NN.NN_MELTS(102CPUFullMELTS_NoCr, cuda = False) 