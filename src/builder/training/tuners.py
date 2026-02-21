import nMELTS.engine.NN as NN
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from nMELTS.utils.string_utils import pull_number
import time
from tqdm import tqdm
from torch import nn
import gc
import sys
from pathlib import Path

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from builder.training.trainer import train_Lower_MELTS, train_Upper_MELTS
from nMELTS.engine.NN import rebuild_MELTS_model
from builder.training.optimizer_factory import create_optimizer, create_scheduler, SchedulerWrapper
from copy import deepcopy

# Set up temp models directory
TEMP_MODELS_DIR = Path(__file__).parent / "temp_models"
TEMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# NO LR SCHEDULERS FOR TUNING.

# AI REVISED:
def tune_Lower_MELTS(Model, trainData=None, testData=None, lr = 1E-4, scheduler=None, scheduler_kwargs = {},
                     Param_Dict=None, Epochs=10, batch_size=1024, early_stopping_patience=5, max_N=np.inf, best_loss=None):
    """
    Tune lower binary saturation model.
    Model argument is now required.
    Returns model with best parameters, with the same weights as before.
    """

    if trainData is None or testData is None:
        raise ValueError("tune_Lower_MELTS requires trainData and testData.")
    if Model is None:
        raise ValueError("tune_Lower_MELTS requires a Model instance.")

    ml_indexer = Model.ml_indexer

    # === Default Param_Dict if none given ===
    if Param_Dict is None:
        Param_Dict = {
            'encoderLayer': [[1, 1], [1, 0], [2, 2], [2, 1], [3, 2]],
            'low_regularization': ['layernormdropout0', 'batchnormdropout0', 'dropout0'],
            'lowWD': [0, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1],
            'noise': [0, 0.01, 0.05, 0.1, 0.2]
        }

    
    allowable_keys = ['low_regularization', 'lowWD', 'encoderLayer', 'activation_leak', 'noise']
    for key in list(Param_Dict.keys()):
        assert key in allowable_keys, f"{key} not in {allowable_keys}!"

    # === Baseline training ===
    print("\n" + "=" * 80)
    print(f" BASELINE TRAINING for {Model.config['description']}")
    print("=" * 80)

    if best_loss is None:
        best_loss = train_Lower_MELTS(Model, trainData, testData, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
                                    Epochs=Epochs, lr=lr, batch_size=batch_size, early_stopping_patience=early_stopping_patience, max_N=max_N)
    results = [{'model': deepcopy(Model.config), 'loss': best_loss}]
    best_config = deepcopy(Model.config)
    best_weights = deepcopy(Model.state_dict())

    # === Begin tuning loop ===
    for parameter, trials in Param_Dict.items():
        print("\n" + "#" * 80)
        print(f" TUNING PARAMETER: {parameter}")
        print("#" * 80)

        # Ensure list type for categorical parameters
        if not isinstance(trials, list):
            try:
                trials = trials.tolist()
            except:
                trials = [trials]

        # --- Handle Encoder Layers (paired parameter) ---
        if parameter == 'encoderLayer':
            trials.append([Model.encoderLayerUp, Model.encoderLayerDown]) # ensure current state represented
            trials = np.array(trials)
            trials = np.unique(trials, axis=0)
            complexity_score = trials[:, 0] - 0.75 * trials[:, 1]
            trials = trials[np.argsort(complexity_score)]
            zero_idx = np.where(
                np.all(trials == np.array([Model.encoderLayerUp, Model.encoderLayerDown]), axis=1)
            )[0][0]

            current_idx = zero_idx + 1
            go_up = current_idx < trials.shape[0]
            go_down = True

            while 0 <= current_idx < trials.shape[0]:
                working_config = deepcopy(best_config)
                working_config['encoderLayerUp'] = trials[current_idx, 0]
                working_config['encoderLayerDown'] = trials[current_idx, 1]

                print(f"\nTesting encoderLayerUp={working_config['encoderLayerUp']}, "
                      f"encoderLayerDown={working_config['encoderLayerDown']}")

                Model = NN.MidLevelNetwork(**working_config, ml_indexer=ml_indexer)
                trial_loss = train_Lower_MELTS(Model, trainData, testData, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
                                               Epochs=Epochs, lr=lr, batch_size=batch_size, early_stopping_patience=early_stopping_patience, max_N=max_N)
                results.append({'model': deepcopy(Model.config), 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
                    best_config = deepcopy(working_config)
                    best_loss = trial_loss
                    best_weights = deepcopy(Model.state_dict())

                    if go_up:
                        go_down = False
                        current_idx += 1
                    else:
                        current_idx -= 1
                elif go_down and go_up:
                    print(f"Going Down... old i: {current_idx}, new i: {zero_idx-1}")
                    current_idx = zero_idx - 1
                    go_up = False
                else:
                    print(f"No improvement — stopping search for {parameter}.")
                    break

            Model = NN.MidLevelNetwork(**best_config, ml_indexer=ml_indexer)
            Model.load_state_dict(best_weights)

        else: 
            # Include current parameter value if missing, handled for encoderlayer case above
            current_val = getattr(Model, parameter)
            if current_val is not None and current_val not in trials:
                trials.append(current_val)

        # --- Handle Simple Continuous Hyperparameters: Weight Decay, noise ---
        if parameter in ['lowWD', 'noise']:
            best_weights_WD = deepcopy(best_weights)
            trials = sorted(set(trials))
            zero_idx = trials.index(Model.config[parameter])
            current_idx = zero_idx + 1
            go_up = current_idx < len(trials)
            go_down = True

            while 0 <= current_idx < len(trials):
                working_config = deepcopy(best_config)
                working_config[parameter] = trials[current_idx]

                print(f"\nTesting {parameter}={working_config[parameter]:.1e}")

                Model = NN.MidLevelNetwork(**working_config, ml_indexer=ml_indexer)
                Model.load_state_dict(best_weights) # Since WD is not model structure, load in best weights)
                trial_loss = train_Lower_MELTS(Model, trainData, testData, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
                                               Epochs=Epochs, lr=lr, batch_size=batch_size, early_stopping_patience=early_stopping_patience, max_N=max_N)
                results.append({'model': deepcopy(Model.config), 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
                    best_config = deepcopy(working_config)
                    best_loss = trial_loss
                    best_weights_WD = deepcopy(Model.state_dict()) # Save state dict without loading it for the next WD trial for fairness
                    if go_up:
                        go_down = False
                        current_idx += 1
                    else:
                        current_idx -= 1
                elif go_down and go_up:
                    current_idx = zero_idx - 1
                    go_up = False
                else:
                    print("No improvement — stopping search for this parameter.")
                    break

            Model = NN.MidLevelNetwork(**best_config, ml_indexer=ml_indexer)
            best_weights = best_weights_WD
            

        # --- Handle Unordered Categorical Parameters: Try every option with no early abort ---
        if parameter in ['activation_factory', 'low_regularization']:
            starting_value = getattr(Model, parameter)
            print(f"Debugging: starting categorical parameter redundancy protection: {parameter} = {starting_value}")
            for trial in trials:
                if trial == starting_value:
                    continue

                working_config = deepcopy(best_config)
                working_config[parameter] = trial

                print(f"\nTesting {parameter}='{trial}'")

                Model = NN.MidLevelNetwork(**working_config, ml_indexer=ml_indexer)
                trial_loss = train_Lower_MELTS(Model, trainData, testData, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
                                               Epochs=Epochs, lr=lr, batch_size=batch_size, early_stopping_patience=early_stopping_patience, max_N=max_N)
                results.append({'model': deepcopy(Model.config), 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
                    best_config = deepcopy(working_config)
                    best_loss = trial_loss
                    best_weights = deepcopy(Model.state_dict())

                else:
                    print(f"No improvement ({trial_loss:.4e})")

            Model = NN.MidLevelNetwork(**best_config, ml_indexer=ml_indexer)

        # === Summary for this parameter ===
        print("\n" + "-" * 80)
        print(f"Best {parameter} configuration so far:")
        for k, v in best_config.items():
            print(f"  {k}: {v}")
        print(f"-> Current best loss: {best_loss:.4e}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print(f"Best overall loss: {best_loss:.4e}")
    print("=" * 80)

    Model.load_state_dict(best_weights) # Load best model's weights

    return Model, results




def tune_Upper_MELTS(Model, trainData=None, testData=None, lr=1E-4, scheduler=None, scheduler_kwargs = {}, Param_Dict=None,
                     Epochs=10, best_loss = None, batch_size=1024, early_stopping_patience=5, binWeights=None, compWeights=None, max_N=np.inf, which_heads_to_freeze = [],
                    chem_alpha=1, mole_alpha=1, bulk_alpha=0, sat_alpha=0, amsgrad=False, eps = 1E-8):
    """
    Function to 
     lower binary saturation model. Initializes new model if none given.
    Best to generate one and give it a description.
    Returns model with best parameters, with the same weights as before.
    """

    

    if trainData is None or testData is None:
        raise ValueError("tune_Upper_MELTS requires trainData and testData.")

    if binWeights is None:
        binWeights = torch.ones(1)
    if compWeights is None:
        compWeights = torch.ones(1)

    # === Default Param_Dict if none given ===
    if Param_Dict is None:
        Param_Dict = {
            'middleLayer': [[1, 1], [2, 2], [3,3], [1, 0], [2, 1], [3, 2]],
            'high_regularization': ['batchnormdropout0', 'layernormdropout0', 'dropout0'],
            'highWD': [0, 1E-6, 1E-5, 1E-4, 1E-3],
            'noise': [0, 0.01, 0.05, 0.1, 0.2]
        }

        """#Test excluding encoder from adaptive dropout
        if 'dropout' in Model.config['low_regularization'].lower():
            Param_Dict['low_regularization'] = []
            if 'batchnorm' in Model.config['low_regularization'].lower():
                Param_Dict['low_regularization'].append('batchnorm')
            elif 'layernorm' in Model.config['low_regularization'].lower():
                Param_Dict['low_regularization'].append('layernorm')
            else:
                Param_Dict['low_regularization'].append('none')"""

    allowable_keys = ['low_regularization', 'highWD', 'lowWD', 'middleLayer', 'high_regularization', 'activation_leak', 'noise']

    for key in list(Param_Dict.keys()):
        assert key in allowable_keys, f"{key} not in {allowable_keys}!"

    # === Baseline training ===
    print("\n" + "=" * 80)
    print(f" BASELINE TRAINING for {Model.config['description']}")
    print("=" * 80)
    if best_loss is None:
        best_loss = train_Upper_MELTS(Model, trainData, testData, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
                          Epochs=Epochs, lr=lr, batch_size=batch_size, early_stopping_patience=early_stopping_patience,
                          binWeights=binWeights, compWeights=compWeights, max_N=max_N, which_heads_to_freeze = which_heads_to_freeze,
                          chem_alpha=chem_alpha, mole_alpha=mole_alpha, bulk_alpha=bulk_alpha, sat_alpha=sat_alpha, amsgrad=amsgrad, eps = eps)

    results = [{'model': deepcopy(Model.config), 'loss': best_loss}]
    torch.save({'state_dict': Model.state_dict(), 'config': Model.config}, str(TEMP_MODELS_DIR / 'Temp_Upper_Tune.pt'))

    # === Begin tuning loop ===
    for parameter, trials in Param_Dict.items():
        print("\n" + "#" * 80)
        print(f" TUNING PARAMETER: {parameter}")
        print(f"Testing: {trials}")
        print("#" * 80)

        # Ensure list type for categorical parameters
        if not isinstance(trials, list):
            try:
                trials = trials.tolist()
            except:
                trials = [trials]

        # --- Handle middleBrain Layers (paired parameter) ---
        if parameter == 'middleLayer':
            trials.append([Model.middleLayerUp, Model.middleLayerDown]) # ensure current state represented
            trials = np.array(trials)
            trials = np.unique(trials, axis=0)
            complexity_score = trials[:, 0] - 0.75 * trials[:, 1]
            trials = trials[np.argsort(complexity_score)]
            zero_idx = np.where(
                np.all(trials == np.array([Model.middleLayerUp, Model.middleLayerDown]), axis=1)
            )[0][0]

            current_idx = zero_idx + 1
            go_up = current_idx < trials.shape[0]
            go_down = True

            while 0 <= current_idx < trials.shape[0]:
                
                #working_config = deepcopy(best_config)
                substitutions = {
                    'middleLayerUp': trials[current_idx, 0],
                    'middleLayerDown': trials[current_idx, 1]
                }

                Model = rebuild_MELTS_model(str(TEMP_MODELS_DIR / 'Temp_Upper_Tune.pt'), substitutions=substitutions, low_only = True, ml_indexer=Model.ml_indexer) # Rebuild model with new middle layer config but load in best weights for lower layers for fair comparison since middle layer changes model structure but not low layer weights.

                print(Model.config)

                print(f"\nTesting middleLayerUp={substitutions['middleLayerUp']}, "
                      f"middleLayerDown={substitutions['middleLayerDown']}")

                trial_loss = train_Upper_MELTS(Model, trainData, testData, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
                                               Epochs=Epochs, lr=lr, batch_size=batch_size, early_stopping_patience=early_stopping_patience,
                                               binWeights=binWeights, compWeights=compWeights, max_N=max_N, which_heads_to_freeze=which_heads_to_freeze,
                                               chem_alpha=chem_alpha, mole_alpha=mole_alpha, bulk_alpha=bulk_alpha, sat_alpha=sat_alpha, amsgrad=amsgrad, eps = eps) # Set sat alpha to 1 for middle layer tuning to isolate middle layer effects without changing relative importance of chemistry vs mole vs bulk loss since they don't change model structure and we want fair comparison between middle layer configs.
                results.append({'model': deepcopy(Model.config), 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
                    torch.save({'state_dict': Model.state_dict(), 'config': Model.config}, str(TEMP_MODELS_DIR / 'Temp_Upper_Tune.pt'))

                    best_loss = trial_loss

                    if go_up:
                        go_down = False
                        current_idx += 1
                    else:
                        current_idx -= 1
                elif go_down and go_up:
                    print(f"Going Down... old i: {current_idx}, new i: {zero_idx-1}")
                    current_idx = zero_idx - 1
                    go_up = False
                else:
                    print(f"No improvement — stopping search for {parameter}.")
                    break

            Model = rebuild_MELTS_model(str(TEMP_MODELS_DIR / 'Temp_Upper_Tune.pt'), ml_indexer=Model.ml_indexer) # Rebuild with upper layers too
            #Model = rebuild_MELTS_model(str(TEMP_MODELS_DIR / 'Temp_Upper_Tune.pt'), low_only=True) # Rebuild with low layers only to preserve any changes from WD/noise tuning for fair comparison since they don't change model structure
            #Model.load_state_dict(torch.load(str(TEMP_MODELS_DIR / 'Temp_Upper_Tune.pt'))['state_dict']) # Load best model so far for fair WD/noise comparison since they don't change model structure



        else: 
            # Include current parameter value if missing, handled for encoderlayer case above
            current_val = getattr(Model, parameter)
            if current_val is not None and current_val not in trials:
                trials.append(current_val)

        # --- Handle Simple continuous hyperparameters: Weight Decay / noise ---
        if parameter in ['highWD', 'lowWD', 'noise']:
            
            trials = sorted(set(trials))
            zero_idx = trials.index(Model.config[parameter])
            current_idx = zero_idx + 1
            go_up = current_idx < len(trials)
            go_down = True

            while 0 <= current_idx < len(trials):
                
                substitutions = {parameter:trials[current_idx]}

                print(f"\nTesting {parameter}={substitutions[parameter]:.1e}")

                Model = rebuild_MELTS_model(str(TEMP_MODELS_DIR / 'Temp_Upper_Tune.pt'), substitutions=substitutions, ml_indexer=Model.ml_indexer)
                Model.load_state_dict(torch.load(str(TEMP_MODELS_DIR / 'Temp_Upper_Tune.pt'))['state_dict']) # Load best model so far for fair WD/noise comparison since they don't change model structure. Redundant?
                trial_loss = train_Upper_MELTS(Model, trainData, testData, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
                                               Epochs=Epochs, lr=lr, batch_size=batch_size, early_stopping_patience=early_stopping_patience,
                                               binWeights=binWeights, compWeights=compWeights, max_N=max_N, which_heads_to_freeze=which_heads_to_freeze,
                                               chem_alpha=chem_alpha, mole_alpha=mole_alpha, bulk_alpha=bulk_alpha, sat_alpha=sat_alpha, amsgrad=amsgrad, eps = eps)
                results.append({'model': deepcopy(Model.config), 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
                    torch.save({'state_dict': Model.state_dict(), 'config': Model.config}, str(TEMP_MODELS_DIR / 'Temp_Upper_Tune.pt'))
                    best_loss = trial_loss
                    if go_up:
                        go_down = False
                        current_idx += 1
                    else:
                        current_idx -= 1
                elif go_down and go_up:
                    current_idx = zero_idx - 1
                    go_up = False
                else:
                    print("No improvement — stopping search for this parameter.")
                    break

        # --- Handle Unordered Categorical Parameters ---
        if parameter in ['activation_leak', 'low_regularization', 'high_regularization']: # Actually activation leak is a continuous parameter and should be treated as such.
            starting_value = getattr(Model, parameter)
            print(f"Debugging: starting categorical parameter redundancy protection: {parameter} = {starting_value}")
            for trial in trials:
                if trial == starting_value:
                    continue

                substitutions = {parameter:trial}
                Model = rebuild_MELTS_model(str(TEMP_MODELS_DIR / 'Temp_Upper_Tune.pt'), substitutions=substitutions, low_only=True, ml_indexer=Model.ml_indexer)

                print(f"\nTesting {parameter}='{trial}'")

                trial_loss = train_Upper_MELTS(Model, trainData, testData, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs, 
                                               Epochs=Epochs, lr=lr, batch_size=batch_size, early_stopping_patience=early_stopping_patience,
                                               binWeights=binWeights, compWeights=compWeights, max_N=max_N, which_heads_to_freeze=which_heads_to_freeze,
                                               chem_alpha=chem_alpha, mole_alpha=mole_alpha, bulk_alpha=bulk_alpha, sat_alpha=sat_alpha, amsgrad=amsgrad, eps = eps) # Set sat alpha to 1 for regularization tuning to isolate regularization effects without changing relative importance of chemistry vs mole vs bulk loss since they don't change model structure and we want fair comparison between regularization configs.
                results.append({'model': deepcopy(Model.config), 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
                    torch.save({'state_dict': Model.state_dict(), 'config': Model.config}, str(TEMP_MODELS_DIR / 'Temp_Upper_Tune.pt')) 
                    best_loss = trial_loss
                
                else:
                    print(f"No improvement ({trial_loss:.4e})")

            Model = rebuild_MELTS_model(str(TEMP_MODELS_DIR / 'Temp_Upper_Tune.pt'), ml_indexer=Model.ml_indexer)


        # === Summary for this parameter ===
        print("\n" + "-" * 80)
        print(f"-> Current best loss: {best_loss:.4e}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print(f"Best overall loss: {best_loss:.4e}")
    print(f"Best config: {Model.config}")
    print("=" * 80)

    return Model, results




