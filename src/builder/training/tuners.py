import ngibbs.engine.NN as NN
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
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
from builder.training.optimizer_factory import create_optimizer, create_scheduler, SchedulerWrapper
from copy import deepcopy

# Set up temp models directory
TEMP_MODELS_DIR = Path(__file__).parent / "temp_models"
TEMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Scheduler is optional (defaults to None/{} below, i.e. flat LR) - main.py now
# passes one through when the episode's YAML configures it, so baseline and
# every per-trial train_Lower_MELTS/train_Upper_MELTS call below share it.


def _save_anchor_bundle(model, bundle_path):
    model.save(str(bundle_path))


def _load_trial_model(bundle_path, ml_indexer, substitutions=None, low_only=False,
                      load_prefixes=None, model_class=None):
    return NN.rebuild_MELTS_model(
        str(bundle_path),
        substitutions=substitutions,
        low_only=low_only,
        ml_indexer=ml_indexer,
        load_prefixes=load_prefixes,
        model_class=model_class,
    )


# --------------------------------------------------------------------------- #
#  Parameter-class dispatch
# --------------------------------------------------------------------------- #
# Three classes of tunable, per the design:
#   architectural  paired [[up, down], ...] layer counts, stored as <name>Up/<name>Down
#   numerical      continuous/ordered; try higher first, then lower, stop when it stops
#                  helping
#   categorical    unordered strings; every option is tried, no early abort
#
# Dispatch is on the parameter NAME for the architectural class ('layer' in the name)
# and on the trial VALUE TYPE for the other two, so no list of blessed hyperparameter
# names appears anywhere below. `encoderLayer`, `middleLayer` and `moleLayer` all work
# without this file knowing any of them exist, and a config a future network adds is
# tunable the day it is added.
#
# One behaviour change falls out of type dispatch: `activation_leak` used to be routed
# categorically despite being continuous (the source said so in a comment). It is now
# searched as a numerical, which is what it always should have been.


def _is_architectural(parameter):
    return 'layer' in str(parameter).lower()


def _arch_keys(parameter):
    """'middleLayer' -> ('middleLayerUp', 'middleLayerDown')."""
    return f'{parameter}Up', f'{parameter}Down'


def _is_numerical(trials):
    return all(isinstance(t, (int, float, np.integer, np.floating))
               and not isinstance(t, (bool, np.bool_)) for t in trials)


def _as_list(trials):
    if isinstance(trials, list):
        return trials
    try:
        return trials.tolist()
    except AttributeError:
        return [trials]


def _validate_param_dict(Param_Dict, config, label):
    """A trial parameter must correspond to real config keys on the model being tuned.

    This replaces the hard-coded `allowable_keys` lists. It is strictly stronger: those
    lists accepted `lowWD` while tuning a model whose config had no such key, and
    rejected any key a new architecture introduced.
    """
    for key in Param_Dict:
        if _is_architectural(key):
            missing = [k for k in _arch_keys(key) if k not in config]
            if missing:
                raise KeyError(
                    f"{label}: architectural parameter '{key}' needs config keys "
                    f"{list(_arch_keys(key))}; missing {missing}. "
                    f"Model config keys: {sorted(config)}")
        elif key not in config:
            raise KeyError(f"{label}: '{key}' is not a config key of this model. "
                           f"Model config keys: {sorted(config)}")


def _order_arch_trials(trials, current_pair):
    """Sort [[up, down], ...] by complexity and report where the current config sits.

    The current pair is appended first so the search always has an origin to walk away
    from, even when the caller's list omits it.
    """
    current_pair = [int(current_pair[0]), int(current_pair[1])]
    arr = np.array([[int(t[0]), int(t[1])] for t in _as_list(trials)] + [current_pair])
    arr = np.unique(arr, axis=0)
    arr = arr[np.argsort(arr[:, 0] - 0.66 * arr[:, 1])]
    zero_idx = int(np.where(np.all(arr == np.array(current_pair), axis=1))[0][0])
    return arr, zero_idx


def _fmt(value):
    return f"{value:.1e}" if isinstance(value, (float, np.floating)) else f"{value}"

# AI REVISED:
def tune_Lower_MELTS(Model, trainData=None, testData=None, lr = 1E-4, scheduler=None, scheduler_kwargs = {},
                     Param_Dict=None, Epochs=10, batch_size=1024, early_stopping_patience=5, max_N=np.inf, best_loss=None,
                     sweep=False, dropout_step_up=0.05, dropout_step_down=0.02, noise_step_up=0.002, noise_step_down=0.001,
                     load_prefixes=('encoder.', 'sat_head.'), device='cuda'):
    """
    Tune lower binary saturation model.
    Model argument is now required.
    Returns model with best parameters, with the same weights as before.

    Parameter classification is shared with `tune_Upper_MELTS` -- see the dispatch notes
    at the top of this file. `load_prefixes` names the state-dict prefixes carried into
    each trial; prefixes the model lacks are skipped rather than raising.
    """

    if trainData is None or testData is None:
        raise ValueError("tune_Lower_MELTS requires trainData and testData.")
    if Model is None:
        raise ValueError("tune_Lower_MELTS requires a Model instance.")

    ml_indexer = Model.ml_indexer
    model_class = type(Model)
    load_prefixes = list(load_prefixes) if load_prefixes else None

    # === Default Param_Dict if none given ===
    if Param_Dict is None:
        Param_Dict = {'encoderLayer': [[1, 1], [1, 0], [2, 2], [2, 1], [3, 2]]}
        reg_key = getattr(Model, 'lower_regularization_config_key', 'low_regularization')
        if reg_key in Model.config:
            Param_Dict[reg_key] = ['layernormdropout0', 'batchnormdropout0', 'dropout0']
        for key, grid in (('lowWD', [0, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1]),
                          ('noise', [0, 0.01, 0.05, 0.1, 0.2])):
            if key in Model.config:
                Param_Dict[key] = grid

    _validate_param_dict(Param_Dict, Model.config, 'tune_Lower_MELTS')

    # === Baseline training ===
    print("\n" + "=" * 80)
    print(f" BASELINE TRAINING for {Model.config['description']}")
    print("=" * 80)

    best_model = Model

    if best_loss is None:
        best_loss = train_Lower_MELTS(best_model, trainData, testData, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
                                    Epochs=Epochs, lr=lr, batch_size=batch_size, early_stopping_patience=early_stopping_patience, max_N=max_N, device=device,
                                               dropout_step_up=dropout_step_up, dropout_step_down=dropout_step_down,
                                               noise_step_up=noise_step_up, noise_step_down=noise_step_down)
    results = [{'model': deepcopy(best_model.config), 'loss': best_loss}]

    # === Begin tuning loop ===
    for parameter, trials in Param_Dict.items():
        print("\n" + "#" * 80)
        print(f" TUNING PARAMETER: {parameter}")
        print("#" * 80)

        anchor_bundle = TEMP_MODELS_DIR / 'Temp_Lower_Anchor.pt'
        _save_anchor_bundle(best_model, anchor_bundle)
        anchor_config = deepcopy(best_model.config)
        parameter_best_config = deepcopy(best_model.config)
        parameter_best_model = best_model
        parameter_best_loss = best_loss

        trials = _as_list(trials)

        # ---- shared trial runner (see the dispatch notes at the top of this file) ---
        def _run_trial(substitutions, label=''):
            trial_model = _load_trial_model(
                anchor_bundle,
                ml_indexer,
                substitutions=substitutions,
                low_only=True,
                load_prefixes=load_prefixes,
                model_class=model_class,
            )
            print(f"\nTesting {label}")
            trial_loss = train_Lower_MELTS(
                trial_model, trainData, testData, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
                Epochs=Epochs, lr=lr, batch_size=batch_size,
                early_stopping_patience=early_stopping_patience, max_N=max_N, device=device,
                dropout_step_up=dropout_step_up, dropout_step_down=dropout_step_down,
                noise_step_up=noise_step_up, noise_step_down=noise_step_down)
            results.append({'model': deepcopy(trial_model.config), 'loss': trial_loss})
            return trial_model, trial_loss

        def _accept(trial_model, trial_loss):
            nonlocal parameter_best_loss, parameter_best_config, parameter_best_model
            if trial_loss >= parameter_best_loss:
                return False
            print(f"Improved! Loss {trial_loss:.4e} < {parameter_best_loss:.4e}")
            parameter_best_loss = trial_loss
            parameter_best_config = deepcopy(trial_model.config)
            parameter_best_model = trial_model
            if sweep:
                _save_anchor_bundle(parameter_best_model, anchor_bundle)
            return True

        # --- ARCHITECTURAL: paired [[up, down], ...] layer counts -------------------
        if _is_architectural(parameter):
            up_key, down_key = _arch_keys(parameter)
            ordered, zero_idx = _order_arch_trials(
                trials, (anchor_config[up_key], anchor_config[down_key]))

            current_idx = zero_idx + 1
            go_up = current_idx < ordered.shape[0]
            go_down = True

            while 0 <= current_idx < ordered.shape[0]:
                substitutions = {up_key: int(ordered[current_idx, 0]),
                                 down_key: int(ordered[current_idx, 1])}
                trial_model, trial_loss = _run_trial(
                    substitutions,
                    label=f"{up_key}={substitutions[up_key]}, {down_key}={substitutions[down_key]}")

                if _accept(trial_model, trial_loss):
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
                    print(f"No improvement - stopping search for {parameter}.")
                    break

        else:
            # Include the current value so the search always has an origin.
            current_val = anchor_config.get(parameter)
            if current_val is not None and current_val not in trials:
                trials.append(current_val)

            # --- NUMERICAL: try higher first, then lower, stop on no gain -----------
            if _is_numerical(trials):
                trials = sorted(set(trials))
                zero_idx = trials.index(anchor_config[parameter])
                if (zero_idx + 1) >= len(trials) and (zero_idx - 1) < 0:
                    print(f"No alternate values to test for {parameter}.")

                for current_idx in range(zero_idx + 1, len(trials)):
                    trial_model, trial_loss = _run_trial(
                        {parameter: trials[current_idx]},
                        label=f"{parameter}={_fmt(trials[current_idx])}")
                    if not _accept(trial_model, trial_loss):
                        print("No improvement upward - switching to lower values.")
                        break

                # Always attempt lower values after finishing/rejecting upward.
                for current_idx in range(zero_idx - 1, -1, -1):
                    trial_model, trial_loss = _run_trial(
                        {parameter: trials[current_idx]},
                        label=f"{parameter}={_fmt(trials[current_idx])}")
                    if not _accept(trial_model, trial_loss):
                        print("No improvement downward - stopping lower-value search.")
                        break

            # --- CATEGORICAL: try every option, never abort early -------------------
            else:
                starting_value = anchor_config.get(parameter)
                print(f"Categorical parameter {parameter}: skipping current value {starting_value!r}")
                for trial in trials:
                    if trial == starting_value:
                        continue
                    trial_model, trial_loss = _run_trial(
                        {parameter: trial}, label=f"{parameter}={trial!r}")
                    if not _accept(trial_model, trial_loss):
                        print(f"No improvement ({trial_loss:.4e})")

        best_model = parameter_best_model

        best_loss = parameter_best_loss

        # === Summary for this parameter ===
        print("\n" + "-" * 80)
        print(f"Best {parameter} configuration so far:")
        for k, v in parameter_best_config.items():
            print(f"  {k}: {v}")
        print(f"-> Current best loss: {best_loss:.4e}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print(f"Best overall loss: {best_loss:.4e}")
    print("=" * 80)

    return best_model, results




def tune_Upper_MELTS(Model, trainData=None, testData=None, lr=1E-4, scheduler=None, scheduler_kwargs = {}, Param_Dict=None,
                     Epochs=10, best_loss = None, batch_size=1024, early_stopping_patience=5, binWeights=None, compWeights=None, max_N=np.inf, which_heads_to_freeze = [],
                    chem_alpha=1, mole_alpha=1, bulk_alpha=0, sat_alpha=0, amsgrad=False, eps = 1E-8, sweep=False,
                    dropout_step_up=0.05, dropout_step_down=0.02, noise_step_up=0.002, noise_step_down=0.001,
                    arch_load_prefixes=('encoder.', 'sat_head.'), device='cuda',
                    train_fn=None, train_fn_kwargs=None):
    """
    Tune the upper model. Initializes new model if none given.
    Best to generate one and give it a description.
    Returns model with best parameters, with the same weights as before.

    Architecture-agnostic: the trial model is rebuilt as `type(Model)`, parameters are
    classified by name/type rather than by a hard-coded list, and the objective is
    whatever `train_Upper_MELTS` computes for that class. Tuning a `ContinuousModel`'s
    `encoderLayer` / `moleLayer` / `mole_regularization` needs no change here.

    train_fn : callable, optional
        The upper training function each trial runs. Defaults to `train_Upper_MELTS`.
        Pass `train_Upper_Sobolev` to sweep against the derivative-supervised objective --
        the search logic is unchanged, because a trial is compared on whatever scalar the
        trainer returns, and both return a validation loss on their own objective.

        The one thing that does NOT survive changing this: losses are only comparable
        within a sweep. A `best_loss` seeded from a value-only episode is meaningless
        against Sobolev trials, so let the baseline retrain rather than passing one across.
    train_fn_kwargs : dict, optional
        Extra keyword arguments forwarded to `train_fn` on every trial AND on the
        baseline, so the baseline is scored on the same objective as its challengers.
    arch_load_prefixes : tuple of str
        State-dict prefixes carried across an architectural rebuild -- everything else is
        re-initialised, since changing layer counts invalidates the shapes downstream.
        The default preserves the gated model's lower half, i.e. exactly the historical
        behaviour. Prefixes absent from the trial model are skipped, not an error, so a
        model without a `sat_head` needs no override; pass e.g. `('encoder.',)` to be
        explicit, or `()` to re-initialise everything.
    """

    if trainData is None or testData is None:
        raise ValueError("tune_Upper_MELTS requires trainData and testData.")
    if Model is None:
        raise ValueError("tune_Upper_MELTS requires a Model instance.")

    model_class = type(Model)
    arch_load_prefixes = list(arch_load_prefixes) if arch_load_prefixes else None
    train_fn = train_Upper_MELTS if train_fn is None else train_fn
    train_fn_kwargs = dict(train_fn_kwargs or {})

    if binWeights is None:
        binWeights = torch.ones(1)
    if compWeights is None:
        compWeights = torch.ones(1)

    # === Default Param_Dict if none given ===
    if Param_Dict is None:
        # Built from the model's own config so the default sweep is meaningful for
        # whatever class was handed in: every architectural pair it exposes except the
        # encoder (which the lower model owns), plus the regularisation key it declares.
        Param_Dict = {}
        for key in Model.config:
            if key.endswith('Up') and _is_architectural(key[:-2]) and key[:-2] != 'encoderLayer':
                Param_Dict[key[:-2]] = [[1, 1], [2, 2], [3, 3], [1, 0], [2, 1], [3, 2]]
        reg_key = getattr(Model, 'upper_regularization_config_key', 'high_regularization')
        if reg_key in Model.config:
            Param_Dict[reg_key] = ['batchnormdropout0', 'layernormdropout0', 'dropout0']
        for key, grid in (('highWD', [0, 1E-6, 1E-5, 1E-4, 1E-3]),
                          ('noise', [0, 0.01, 0.05, 0.1, 0.2])):
            if key in Model.config:
                Param_Dict[key] = grid

        """#Test excluding encoder from adaptive dropout
        if 'dropout' in Model.config['low_regularization'].lower():
            Param_Dict['low_regularization'] = []
            if 'batchnorm' in Model.config['low_regularization'].lower():
                Param_Dict['low_regularization'].append('batchnorm')
            elif 'layernorm' in Model.config['low_regularization'].lower():
                Param_Dict['low_regularization'].append('layernorm')
            else:
                Param_Dict['low_regularization'].append('none')"""

    _validate_param_dict(Param_Dict, Model.config, 'tune_Upper_MELTS')

    # === Baseline training ===
    print("\n" + "=" * 80)
    print(f" BASELINE TRAINING for {Model.config['description']}")
    print("=" * 80)

    if best_loss is None:
        best_loss = train_fn(Model, trainData, testData, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
                          **train_fn_kwargs,
                          Epochs=Epochs, lr=lr, batch_size=batch_size, early_stopping_patience=early_stopping_patience,
                          binWeights=binWeights, compWeights=compWeights, max_N=max_N, device=device, which_heads_to_freeze = which_heads_to_freeze,
                          chem_alpha=chem_alpha, mole_alpha=mole_alpha, bulk_alpha=bulk_alpha, sat_alpha=sat_alpha, amsgrad=amsgrad, eps = eps,
                                               dropout_step_up=dropout_step_up, dropout_step_down=dropout_step_down,
                                               noise_step_up=noise_step_up, noise_step_down=noise_step_down)

    results = [{'model': deepcopy(Model.config), 'loss': best_loss}]
    best_model = Model

    # === Begin tuning loop ===
    for parameter, trials in Param_Dict.items():
        print("\n" + "#" * 80)
        print(f" TUNING PARAMETER: {parameter}")
        print(f"Testing: {trials}")
        print("#" * 80)

        anchor_bundle = TEMP_MODELS_DIR / "Temp_Upper_Anchor.pt"
        _save_anchor_bundle(best_model, anchor_bundle)

        active_config = deepcopy(best_model.config)
        parameter_best_config = deepcopy(best_model.config)
        parameter_best_model = best_model
        parameter_best_loss = best_loss

        trials = _as_list(trials)

        # ---- shared trial runner: build the model, train it, record, compare --------
        def _run_trial(substitutions, load_prefixes=None, label=''):
            trial_model = _load_trial_model(
                anchor_bundle,
                best_model.ml_indexer,
                substitutions=substitutions,
                load_prefixes=load_prefixes,
                model_class=model_class,
            )
            print(f"\nTesting {label}")
            trial_loss = train_fn(
                trial_model, trainData, testData, scheduler=scheduler, scheduler_kwargs=scheduler_kwargs,
                **train_fn_kwargs,
                Epochs=Epochs, lr=lr, batch_size=batch_size, early_stopping_patience=early_stopping_patience,
                binWeights=binWeights, compWeights=compWeights, max_N=max_N, device=device,
                which_heads_to_freeze=which_heads_to_freeze,
                chem_alpha=chem_alpha, mole_alpha=mole_alpha, bulk_alpha=bulk_alpha, sat_alpha=sat_alpha,
                amsgrad=amsgrad, eps=eps,
                dropout_step_up=dropout_step_up, dropout_step_down=dropout_step_down,
                noise_step_up=noise_step_up, noise_step_down=noise_step_down)
            results.append({'model': deepcopy(trial_model.config), 'loss': trial_loss})
            return trial_model, trial_loss

        def _accept(trial_model, trial_loss):
            nonlocal parameter_best_loss, parameter_best_config, parameter_best_model
            if trial_loss >= parameter_best_loss:
                return False
            print(f"Improved! Loss {trial_loss:.4e} < {parameter_best_loss:.4e}")
            parameter_best_loss = trial_loss
            parameter_best_config = deepcopy(trial_model.config)
            parameter_best_model = trial_model
            if sweep:
                _save_anchor_bundle(parameter_best_model, anchor_bundle)
            return True

        # --- ARCHITECTURAL: paired [[up, down], ...] layer counts -------------------
        if _is_architectural(parameter):
            up_key, down_key = _arch_keys(parameter)
            ordered, zero_idx = _order_arch_trials(
                trials, (active_config[up_key], active_config[down_key]))

            current_idx = zero_idx + 1
            go_up = current_idx < ordered.shape[0]
            go_down = True

            if not go_up and go_down:
                print(f"Going Down... old i: {current_idx}, new i: {zero_idx-1}")
                current_idx = zero_idx - 1
                go_up = False

            while 0 <= current_idx < ordered.shape[0]:
                substitutions = {up_key: int(ordered[current_idx, 0]),
                                 down_key: int(ordered[current_idx, 1])}
                trial_model, trial_loss = _run_trial(
                    substitutions,
                    load_prefixes=arch_load_prefixes,
                    label=f"{up_key}={substitutions[up_key]}, {down_key}={substitutions[down_key]}")

                if _accept(trial_model, trial_loss):
                    current_idx += 1 if go_up else -1
                elif go_down and go_up:
                    print(f"Going Down... old i: {current_idx}, new i: {zero_idx-1}")
                    current_idx = zero_idx - 1
                    go_up = False
                else:
                    print(f"No improvement - stopping search for {parameter}.")
                    break

        else:
            # Include the current value so the search always has an origin.
            current_val = active_config.get(parameter)
            if current_val is not None and current_val not in trials:
                trials.append(current_val)

            # --- NUMERICAL: ordered; try higher first, then lower, stop on no gain ---
            if _is_numerical(trials):
                trials = sorted(set(trials))
                zero_idx = trials.index(active_config[parameter])
                go_up = (zero_idx + 1) < len(trials)
                go_down = (zero_idx - 1) >= 0
                if go_up:
                    current_idx = zero_idx + 1
                elif go_down:
                    current_idx = zero_idx - 1
                else:
                    current_idx = -1
                    print(f"No alternate values to test for {parameter}.")

                while 0 <= current_idx < len(trials):
                    substitutions = {parameter: trials[current_idx]}
                    trial_model, trial_loss = _run_trial(
                        substitutions, label=f"{parameter}={_fmt(trials[current_idx])}")

                    if _accept(trial_model, trial_loss):
                        if go_up:
                            go_down = False
                            current_idx += 1
                        else:
                            current_idx -= 1
                    elif go_down and go_up:
                        current_idx = zero_idx - 1
                        go_up = False
                    else:
                        print("No improvement - stopping search for this parameter.")
                        break

            # --- CATEGORICAL: unordered; try every option, never abort early --------
            else:
                starting_value = active_config.get(parameter)
                print(f"Categorical parameter {parameter}: skipping current value {starting_value!r}")
                for trial in trials:
                    if trial == starting_value:
                        continue
                    trial_model, trial_loss = _run_trial(
                        {parameter: trial}, label=f"{parameter}={trial!r}")
                    if not _accept(trial_model, trial_loss):
                        print(f"No improvement ({trial_loss:.4e})")

        best_model = parameter_best_model
        best_loss = parameter_best_loss


        # === Summary for this parameter ===
        print("\n" + "-" * 80)
        print(f"-> Current best loss: {best_loss:.4e}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print(f"Best overall loss: {best_loss:.4e}")
    print(f"Best config: {best_model.config}")
    print("=" * 80)

    return best_model, results




