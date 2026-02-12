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
from src.builder.training.trainer import train_Lower_MELTS, train_Upper_MELTS
from src.nMELTS.engine.NN import rebuild_MELTS_model
from src.builder.training.optimizer_factory import create_optimizer, create_scheduler, SchedulerWrapper
from copy import deepcopy
# NO LR SCHEDULERS FOR TUNING.

# AI REVISED:
def tune_Lower_MELTS(Model=None, lr = 1E-4, scheduler=None, scheduler_kwargs = {},Param_Dict=None, Epochs=10):
    """
    Function to 
    lower binary saturation model. Initializes new model if none given.
    Best to generate one and give it a description.
    Returns model with best parameters, with the same weights as before.
    """

    # === Default model if none given ===
    if Model is None:
        Model = NN.MidLevelNetwork(
            encoderLayerUp=1,
            encoderLayerDown=0,
            low_regularization='layernormdropout0',
            lowWD = 0#1E-5
        )

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
    best_loss = train_Lower_MELTS(Model, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs, Epochs=Epochs)
    results = [{'model': deepcopy(Model.config), 'loss': best_loss}]
    best_config = deepcopy(Model.config)
    best_weights = deepcopy(Model.state_dict())

    # === Begin tuning loop ===
    for parameter, trials in Param_Dict.items():
        print("\n" + "#" * 80)
        print(f" TUNING PARAMETER: {parameter}")
        print("#" * 80)

        # Ensure list type for categorical parameters
        if isinstance(trials, np.ndarray):
            trials = trials.tolist()

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

                Model = NN.MidLevelNetwork(**working_config)
                trial_loss = train_Lower_MELTS(Model, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs, Epochs=Epochs)
                results.append({'model': deepcopy(Model.config), 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"✅ Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
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
                    print(f"❌ No improvement — stopping search for {parameter}.")
                    break

            Model = NN.MidLevelNetwork(**best_config)
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

                Model = NN.MidLevelNetwork(**working_config)
                Model.load_state_dict(best_weights) # Since WD is not model structure, load in best weights)
                trial_loss = train_Lower_MELTS(Model, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs, Epochs=Epochs)
                results.append({'model': deepcopy(Model.config), 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"✅ Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
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
                    print("❌ No improvement — stopping search for this parameter.")
                    break

            Model = NN.MidLevelNetwork(**best_config)
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

                Model = NN.MidLevelNetwork(**working_config)
                trial_loss = train_Lower_MELTS(Model, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs, Epochs=Epochs)
                results.append({'model': deepcopy(Model.config), 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"✅ Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
                    best_config = deepcopy(working_config)
                    best_loss = trial_loss
                    best_weights = deepcopy(Model.state_dict())

                else:
                    print(f"❌ No improvement ({trial_loss:.4e})")

            Model = NN.MidLevelNetwork(**best_config)

        # === Summary for this parameter ===
        print("\n" + "-" * 80)
        print(f"Best {parameter} configuration so far:")
        for k, v in best_config.items():
            print(f"  {k}: {v}")
        print(f"→ Current best loss: {best_loss:.4e}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print(f"Best overall loss: {best_loss:.4e}")
    print("=" * 80)

    Model.load_state_dict(best_weights) # Load best model's weights

    return results




def tune_Upper_MELTS(Model, lr=1E-4, scheduler=None, scheduler_kwargs = {}, Param_Dict=None, Epochs=10, best_loss = None):
    """
    Function to 
     lower binary saturation model. Initializes new model if none given.
    Best to generate one and give it a description.
    Returns model with best parameters, with the same weights as before.
    """

    

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

    allowable_keys = ['low_regularization', 'highWD', 'middleLayer', 'high_regularization', 'activation_leak', 'noise']

    for key in list(Param_Dict.keys()):
        assert key in allowable_keys, f"{key} not in {allowable_keys}!"

    # === Baseline training ===
    print("\n" + "=" * 80)
    print(f" BASELINE TRAINING for {Model.config['description']}")
    print("=" * 80)
    if best_loss is None:
        best_loss = train_Upper_MELTS(Model, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs, Epochs=Epochs)

    results = [{'model': deepcopy(Model.config), 'loss': best_loss}]
    torch.save({'state_dict': Model.state_dict(), 'config': Model.config}, 'Models/Temp_Upper_Tune.pt')

    # === Begin tuning loop ===
    for parameter, trials in Param_Dict.items():
        print("\n" + "#" * 80)
        print(f" TUNING PARAMETER: {parameter}")
        print("#" * 80)

        # Ensure list type for categorical parameters
        if isinstance(trials, np.ndarray):
            trials = trials.tolist()

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

                Model = rebuild_MELTS_model('Models/Temp_Upper_Tune.pt', substitutions=substitutions, low_only = True)

                print(Model.config)

                print(f"\nTesting middleLayerUp={substitutions['middleLayerUp']}, "
                      f"middleLayerDown={substitutions['middleLayerDown']}")

                trial_loss = train_Upper_MELTS(Model, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs, Epochs=Epochs)
                results.append({'model': deepcopy(Model.config), 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"✅ Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
                    torch.save({'state_dict': Model.state_dict(), 'config': Model.config}, 'Models/Temp_Upper_Tune.pt')

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
                    print(f"❌ No improvement — stopping search for {parameter}.")
                    break

            Model = rebuild_MELTS_model('Models/Temp_Upper_Tune.pt') # Rebuild with upper layers too


        else: 
            # Include current parameter value if missing, handled for encoderlayer case above
            current_val = getattr(Model, parameter)
            if current_val is not None and current_val not in trials:
                trials.append(current_val)

        # --- Handle Simple continuous hyperparameters: Weight Decay / noise ---
        if parameter in ['highWD', 'noise']:
            
            changedWD = False # Track if WD/noise changes to load new model at the end. 
            trials = sorted(set(trials))
            zero_idx = trials.index(Model.config[parameter])
            current_idx = zero_idx + 1
            go_up = current_idx < len(trials)
            go_down = True

            while 0 <= current_idx < len(trials):
                
                substitutions = {parameter:trials[current_idx]}

                print(f"\nTesting {parameter}={substitutions[parameter]:.1e}")

                Model = rebuild_MELTS_model('Models/Temp_Upper_Tune.pt', substitutions=substitutions)

                trial_loss = train_Upper_MELTS(Model, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs, Epochs=Epochs)
                results.append({'model': deepcopy(Model.config), 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"✅ Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
                    torch.save({'state_dict': Model.state_dict(), 'config': Model.config}, 'Temp_Upper_TuneWD.pt')
                    changedWD = True
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
                    print("❌ No improvement — stopping search for this parameter.")
                    break

            if changedWD:
                Model = rebuild_MELTS_model('Temp_Upper_TuneWD.pt')
                torch.save({'state_dict': Model.state_dict(), 'config': Model.config}, 'Models/Temp_Upper_Tune.pt') # Replace best model with new best
            else: # Rebuild best model
                Model = rebuild_MELTS_model('Models/Temp_Upper_Tune.pt')

        # --- Handle Unordered Categorical Parameters ---
        if parameter in ['activation_leak', 'low_regularization', 'high_regularization']: # Actually activation leak is a continuous parameter and should be treated as such.
            starting_value = getattr(Model, parameter)
            print(f"Debugging: starting categorical parameter redundancy protection: {parameter} = {starting_value}")
            for trial in trials:
                if trial == starting_value:
                    continue

                substitutions = {parameter:trial}
                Model = rebuild_MELTS_model('Models/Temp_Upper_Tune.pt', substitutions=substitutions, low_only=True)

                print(f"\nTesting {parameter}='{trial}'")

                trial_loss = train_Upper_MELTS(Model, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs, Epochs=Epochs)
                results.append({'model': deepcopy(Model.config), 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"✅ Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
                    torch.save({'state_dict': Model.state_dict(), 'config': Model.config}, 'Models/Temp_Upper_Tune.pt') 
                    best_loss = trial_loss
                
                else:
                    print(f"❌ No improvement ({trial_loss:.4e})")

            Model = rebuild_MELTS_model('Models/Temp_Upper_Tune.pt')


        # === Summary for this parameter ===
        print("\n" + "-" * 80)
        print(f"→ Current best loss: {best_loss:.4e}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print(f"Best overall loss: {best_loss:.4e}")
    print("=" * 80)

    return Model, results




