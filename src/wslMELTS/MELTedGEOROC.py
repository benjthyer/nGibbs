import ensemble_MELTSV2 # The essential ensemble MELTS functions
import numpy as np
import random
import pickle
import os   
import string
import shutil

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def move_file(src_filename, dst_dir, overwrite=False):
    """
    Move a file from the current working directory to a destination directory.

    Args:
        src_filename (str): Name of the file in the current working directory.
        dst_dir (str): Destination directory path.
        overwrite (bool): If True, overwrite any existing file with the same name.
    """
    # Ensure the source file exists
    src_path = os.path.join(os.getcwd(), src_filename)
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    # Ensure destination directory exists
    #if not os.path.exists(dst_dir):
    #   os.makedirs(dst_dir)

    # Destination file path
    dst_path = os.path.join(dst_dir, src_filename)

    # Handle overwrite
    if os.path.exists(dst_path):
        if overwrite:
            os.remove(dst_path)
        else:
            raise FileExistsError(f"Destination file already exists: {dst_path}")

    # Move the file
    shutil.move(src_path, dst_path)
    print(f"Moved '{src_filename}' to '{dst_dir}' successfully.")

from pathlib import Path
import shutil

def move_files_with_keyword(keyword, dst_dir, src_dir=None, overwrite=False):
    """
    Move all files whose names contain a given keyword from the source directory
    (or current working directory) to the destination directory.

    Args:
        keyword (str): Substring to search for in filenames.
        dst_dir (str or Path): Destination directory path.
        src_dir (str or Path, optional): Source directory path. Defaults to current working directory.
        overwrite (bool): If True, overwrite existing files in destination.
    """
    src_path = Path(src_dir) if src_dir else Path.cwd()
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    # Find all files containing the keyword in their filename
    files_to_move = [f for f in src_path.iterdir() if f.is_file() and keyword in f.name]

    if not files_to_move:
        print(f"No files containing '{keyword}' found in {src_path}.")
        return

    for file_path in files_to_move:
        dest_file = dst_path / file_path.name

        if dest_file.exists() and not overwrite:
            print(f"Skipping {file_path.name}, already exists in destination.")
            continue

        if dest_file.exists() and overwrite:
            dest_file.unlink()

        shutil.move(str(file_path), str(dest_file))
        print(f"Moved {file_path.name} -> {dst_path}")



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

#MELTSModel = '102' # 102, 110, 120, p
calctype = 'Cooling' # Isobaric: 'Cooling', 'Compression'. To add: Isentropic, Isochoric, Isenthalpic  # 'FxCryst', 'FxMelt', 'Batch'
date = 'Nov20'
#storage_directory = f'/mnt/d/Workspace/{MELTSModel}Datasets/'

# Check that arguments are valid


#batch_file = MELTSModel + 'batch'

for N, MELTSModel in enumerate(['p', '102', '120']): 
    for fractionate in ['Batch', 'FxCryst']:

        if MELTSModel == 'p':
             allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
            'rhm-oxide','alloy-solid','apatite','whitlockite','quartz','tridymite','cristobalite','fluid','liquid']
        else:
            allowed_phases = ['olivine','orthopyroxene','clinopyroxene','spinel','plagioclase','k-feldspar','garnet',
                'nepheline','leucite','biotite','rhm-oxide','alloy-solid','apatite','whitlockite','quartz','tridymite','cristobalite','muscovite','fluid','liquid']


        assert fractionate in ['Batch', 'FxCryst'], "fractionate argument must be one of ['Batch', 'FxCryst'], 'FxMelt' not yet implemented"
        assert calctype in ['Cooling', 'Compression'], "calctype argument must be one of ['Cooling', 'Compression'], isoentropic, isoenthalpic, isochroic not yet implemented"
        assert MELTSModel in ['102', '110', '120', 'p'], "MELTSModel argument must be one of ['102', '110', '120', 'p'], MAGEmin not yet implemented"

        batch_file = MELTSModel + 'Batch'
        Trainfilename = f'MELTS{MELTSModel}_Trainset{date}{fractionate}{calctype}'
        Validfilename = f'MELTS{MELTSModel}_Validset{date}{fractionate}{calctype}'

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

        simcycle = 50

        if MELTSModel == 'p':
            Prange = [8000,30000]
            end = 1000
        else:
            Prange = [1,12000]
            end = 700

        def alphaMELTScompress(output_file, iter = 750, fxtal = False):
            if iter <= 0 or iter != int(iter):
                print('Must have integer positive nonzero for iteration "iter" arguement. Substituting 1.')
                iter = int(1)
            """COMPRESSION NOT IMPLEMENTED FOR pMELTS YET; CHECK ENSEMBLEMELTS.PY"""
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
                batchname[:] = batch_file # ONLY CRUSTAL UP TO 1.5 GPa !!!!


                ensemble_MELTSV2.forward_ensemble(in_array, keys, batchname = batchname, only_phases=allowed_phases, end = 12000+in_array[:,0], fxtal = fxtal, EnsembleLocation=EnsembleLocation, WSL = True, compression=True, delta = 12000/200)

                #PTID = np.random.randint(len(batchname)) 
                for i, name in enumerate(batchname):

                    batchname[i] = f"{pull_number(str(indices[i]))}:{random_char(4)}:{name}" # Put PTX Index in metadata to link double detected phases for quality control

                
                faultIDs = ensemble_MELTSV2.import_MELTS_components(EnsembleLocation=EnsembleLocation, batchname=batchname, fO2Arr=in_array[:,2], dataname = f'{output_file}.csv')
                ensemble_MELTSV2.pick_exsolution_failure(EnsembleLocation=EnsembleLocation, input_array=in_array, keys=keys, batchname=batchname, dataname = f'{output_file}_Exfail.csv', faultIDs=faultIDs)
                j += 1

        def alphaMELTScooling(output_file, iter = 750, fxtal = False):
            if iter <= 0 or iter != int(iter):
                print('Must have integer positive nonzero for iteration "iter" arguement. Substituting 1.')
                iter = int(1)

            def PTF_initialize(conditions, length = int(40)): # ONLY CRUSTAL UP TO 1.5 GPa !!!!
                out_array = np.zeros((length, np.shape(conditions)[1] + 3))
                #out_array[:,0] = np.random.uniform(1,21, size = length)#COMPRESSION
                out_array[:,0] = np.random.uniform(*Prange, size = length)#COOLING # Pressure in bars
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

                if MELTSModel == 'p': # Cancel out many poorly handled volatiles
                    NoKP = np.random.randint(simcycle, high=None, size=int(simcycle/2))
                    compositions[NoKP, key_dict['P2O5']-3] = 0
                    compositions[NoKP, key_dict['K2O']-3] = 0


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
                batchname[:] = batch_file # ONLY CRUSTAL UP TO 1.5 GPa !!!!


                ensemble_MELTSV2.forward_ensemble(in_array, keys, batchname = batchname, only_phases=allowed_phases, end = end,  fxtal = fxtal, EnsembleLocation=EnsembleLocation, WSL = True, compression=False, delta = -1)

                #PTID = np.random.randint(len(batchname)) 
                for i, name in enumerate(batchname):

                    batchname[i] = f"{pull_number(str(indices[i]))}:{random_char(4)}:{name}" # Put PTX Index in metadata to link double detected phases for quality control

                
                faultIDs = ensemble_MELTSV2.import_MELTS_components(EnsembleLocation=EnsembleLocation, batchname=batchname, fO2Arr=in_array[:,2], dataname = f'{output_file}.csv')
                ensemble_MELTSV2.pick_exsolution_failure(EnsembleLocation=EnsembleLocation, input_array=in_array, keys=keys, batchname=batchname, dataname = f'{output_file}_Exfail.csv', faultIDs=faultIDs)
                j += 1

        #logic trees to direct dataset generation:
        if calctype == 'Cooling':
            MELTER = alphaMELTScooling
        elif calctype == 'Compression':
            MELTER = alphaMELTScompress

        total_to_run = int(160)# 300
        if MELTSModel == 'p' and fractionate == 'batch':
            mafics_to_run = int(total_to_run * 0.9)
            full_to_run = int(total_to_run * 0.1)
        elif MELTSModel == 'p':
            mafics_to_run = int(total_to_run * 0.975)
            full_to_run = int(total_to_run * 0.025)
        elif MELTSModel == '120' and fractionate == 'batch':
            mafics_to_run = int(0)
            full_to_run = total_to_run
        elif MELTSModel == '120':
            mafics_to_run = int(total_to_run * 0.2)
            full_to_run = int(total_to_run * 0.8)
        elif fractionate == 'batch':
            mafics_to_run = int(total_to_run * 0.3)
            full_to_run = int(total_to_run * 0.7)
        else:
            mafics_to_run = int(total_to_run * 0.7)
            full_to_run = int(total_to_run * 0.3)


        # Generate Training Dataset
        
        #if N != 0: #DELETE LATER
        
        GEOROC = np.genfromtxt('GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv', delimiter=',',skip_header=1)
        full_indices = np.genfromtxt('GEOROC_PETDB_UNFILTERED_WHOLEROCK_TRAIN.csv', delimiter=',',skip_header=1, dtype = str)[:,0]

        if fractionate == 'Batch':
            MELTER(output_file=Trainfilename, iter = full_to_run)
        elif fractionate == 'FxCryst':
            MELTER(output_file=Trainfilename, iter = full_to_run, fxtal = True)

        if mafics_to_run != 0:
            mafics = GEOROC[:,5]>=5 # MgO above 5
            full_indices = full_indices[mafics]

            GEOROC = GEOROC[mafics] ### TEMP: MAFICS ONLY TO BALANCE DATASET

            if fractionate == 'Batch':
                MELTER(output_file=Trainfilename, iter = mafics_to_run)
            elif fractionate == 'FxCryst':
                MELTER(output_file=Trainfilename, iter = mafics_to_run, fxtal = True)





        GEOROC = np.genfromtxt('GEOROC_PETDB_UNFILTERED_WHOLEROCK_VALIDATION.csv', delimiter=',',skip_header=1)
        full_indices = np.genfromtxt('GEOROC_PETDB_UNFILTERED_WHOLEROCK_VALIDATION.csv', delimiter=',',skip_header=1, dtype = str)[:,0]

        if fractionate == 'Batch':
            MELTER(output_file=Validfilename, iter = int(full_to_run//4))
        elif fractionate == 'FxCryst':
            MELTER(output_file=Validfilename, iter = int(full_to_run//4), fxtal = True)

        if mafics_to_run != 0:
            mafics = GEOROC[:,5]>=5 # MgO above 5
            full_indices = full_indices[mafics]

            GEOROC = GEOROC[mafics] ### TEMP: MAFICS ONLY TO BALANCE DATASET

            if fractionate == 'Batch':
                MELTER(output_file=Validfilename, iter = int(mafics_to_run//4))
            elif fractionate == 'FxCryst':
                MELTER(output_file=Validfilename, iter = int(mafics_to_run//4), fxtal = True)

#fxtal and batch crystallization 102 mafics disproportionately strongly represented in testing

# Move (DEBUG, Need pemissions in WSL)
#move_files_with_keyword(keyword = f'MELTS{MELTSModel}', dst_dir=storage_directory, src_dir=None, overwrite=False)