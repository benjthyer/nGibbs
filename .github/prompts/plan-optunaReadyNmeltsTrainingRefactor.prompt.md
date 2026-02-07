# Plan: Optuna-Ready nMELTS Training Refactor

**TL;DR:** Restructure training scripts into 5 modular components (dataset, model, config, optimizer, trainer), integrate Optuna as a thin wrapper, store ml_indexer in checkpoints via pickle, use a single unified `training.yaml`, and support staged training with encoder freezing. Remove all Cr logic. Split ~1100 lines from `tuners.py` into 6-8 separate files with clear responsibilities. This enables independent model/dataset inputs, configurable hyperparameters, and advanced training strategies (frozen encoders → unfrozen fine-tuning).

---

## Steps

### **Phase 1: Architecture & Module Design**

1. **Create new directory structure** in `src/builder/`:
   ```
   src/builder/training/
   ├── __init__.py
   ├── config/
   │   ├── __init__.py
   │   ├── training_config.py       (DataClass for training.yaml parsing)
   │   └── defaults.yaml            (Default hyperparameters)
   ├── data/
   │   ├── __init__.py
   │   ├── dataset.py               (PyTorch Dataset wrappers)
   │   └── loader.py                (DataLoader factory with batching strategy)
   ├── design/
   │   ├── __init__.py
   │   ├── model_factory.py          (Create/load models with ml_indexer)
   │   ├── loss_factory.py           (Compute phase-specific weighted losses)
   │   ├── constraints.py            (Physics-aware loss terms)
   │   └── optimizer_factory.py      (Create optimizers/schedulers from config)
   ├── trainer/
   │   ├── __init__.py
   │   ├── base_trainer.py           (Abstract training loop, epoch management)
   │   ├── lower_trainer.py          (Binary saturation training)
   │   └── upper_trainer.py          (Chemistry + mole prediction)
   ├── tuning/
   │   ├── __init__.py
   │   ├── optuna_objectives.py      (Optuna trial objectives for lower/upper)
   │   └── search_space.py           (Define hyperparameter search space)
   └── main.py                       (CLI entry point for training/tuning)
   ```

2. **Define unified config structure** in `config/training.yaml`:
   - `lower_model`: encoder architecture, regularization, training hyperparameters
   - `upper_model`: middle brain + heads, training hyperparameters
   - `data`: batch size, num workers, train/test split (Test data is to be taken NOT from the training bundle, 
             but from a separate validation bundle (called VALIDMELTS) to ensure no data leakage)
   - `optuna`: search space, pruner strategy, study name
   - `training_strategy`: stage definitions (frozen encoder, unfrozen, etc.)
   - Remove all Cr-specific sections

3. **Update MidLevelNetwork in `src/nMELTS/engine/NN.py`** to accept ml_indexer:
   - Add `ml_indexer` parameter to `__init__`
   - Store as `self.ml_indexer` (pickle-serializable reference)
   - Replace global config imports with indexer-sourced data
   - Update `save()` method to include `ml_indexer`

---

### **Phase 2: Modular Data & Models**

4. **Create unified dataset wrapper** in `src/builder/training/data/dataset.py`: ***NOTE: Do not change loadTrainData.py. It should still return a dictionary of arrays for now.***
   - Single `MLDataset` class accepts dictionary of arrays: features, binaries, chemistries, molarities, and ml_indexer, +/- free outputs (not subject to mass balance or transformation)
   - Constructor: `MLDataset(data_dict)` where `data_dict` has keys: `features`, `binaries`, `chemistries`, `molarities`, `ml_indexer`, `+/- free outputs`
   - Supports both 4-head modes or 5 head modes (for datasets with optional free outputs)
   - Property-based access to metadata: `n_phases`, `n_components`, etc. from ml_indexer

5. **Create model factory** in `src/builder/training/models/model_factory.py`:
   - `create_model(config, ml_indexer, device='cuda')` → MidLevelNetwork instance
   - `load_model(checkpoint_path, device='cuda')` → reconstructs model + ml_indexer from .pt file
   - Validates checkpoint ml_indexer version compatibility

6. **Refactor loss computation** in `src/builder/training/loss/loss_factory.py`: 
   - Extract hardcoded loss weights (lines 246-248 in current tuners.py) into config
   - `compute_losses(predictions, targets, ml_indexer, loss_config)` 
   - Support masking by `compositionally_variable_subset` from ml_indexer
   - Return structured dict: `{binary_loss, chemistry_loss, molar_loss, total_loss}`

7. **Create optimizer factory** in `src/builder/training/optimizer/optimizer_factory.py`:
   - `create_optimizer(model, config)` with support for Adam, SGD, AdamW from config
   - `create_scheduler(optimizer, config)` for LR scheduling
   - Isolate weight decay (`lowWD`, `highWD`) as config parameters

***NOTE: I am concerned that this is an overly seqmented/complex design. See above directory structure changes. The loss, optimizer, and models directories are merged into a "design" directory.***
---

### **Phase 3: Training Loop Refactoring**

8. **Create base trainer** in `src/builder/training/trainer/base_trainer.py`:
   - Abstract `BaseTrainer` class with:
     - `train_epoch(train_loader, optimizer, scheduler)`
     - `validate(val_loader)` with early stopping logic
     - `checkpoint management` (save best, periodic snapshots)
     - Device handling (CPU/GPU)
   - Shared validation logic from current tuners.py (lines 87-89 early stopping)
   - Return training history dict for Optuna callbacks

9. **Create lower trainer** in `src/builder/training/trainer/lower_trainer.py`:
   - `LowerTrainer(BaseTrainer)` for binary saturation prediction
   - Override `train_epoch()` to compute only binary loss
   - Use BCEWithLogitsLoss (from current forward_binaries)
   - No masking needed for binary predictions

10. **Create upper trainer** in `src/builder/training/trainer/upper_trainer.py`:
    - `UpperTrainer(BaseTrainer)` for chemistry + molar prediction
    - Constructor accepts `lower_model` parameter for encoder access
    - Support frozen/unfrozen encoder via parameter `freeze_encoder=True`
    - Override `train_epoch()` to:
      - Compute encoder latent with `lower_model.encoder(x)` if frozen
      - Or use upper model's own encoder if unfrozen
      - Apply masking via `boolTransCompToOx` from ml_indexer
      - Compute weighted loss for chemistry + molar outputs
    - Implement physics polish functions (`.polish_negative_px()`, etc.) from `NN.py` as optional post-processing

---

### **Phase 4: Optuna Integration**

11. **Create Optuna objectives** in `src/builder/training/tuning/optuna_objectives.py`:
    - `objective_lower(trial, train_loader, val_loader, ml_indexer, base_config)`:
      - Trial samples: `encoderLayerUp`, `encoderLayerDown`, `low_regularization`, activation_leak
      - Creates LowerTrainer, runs training, returns best val loss
      - Early stopping per trial (prune if not improving)
    - `objective_upper(trial, train_loader, val_loader, lower_model, ml_indexer, base_config)`:
      - Trial samples: `middleLayerUp`, `middleLayerDown`, `high_regularization`, `highWD`, `noise`
      - Creates UpperTrainer with frozen encoder, returns best val loss
      - Report intermediate values for Optuna pruning

12. **Define search space** in `src/builder/training/tuning/search_space.py`:
    - Replace hardcoded grid (lines 419-428 in current tuners.py) with Optuna suggest calls
    - Map training.yaml `optuna.search_space` to trial.suggest_* methods
    - Support categorical, range, and fixed hyperparameters

---

### **Phase 5: Configuration & Entry Point**

13. **Create training config dataclass** in `src/builder/training/config/training_config.py`: 
    - Parse `config/training.yaml` → typed Python objects
    - Validate hyperparameter ranges
    - Provide defaults for missing keys

14. **Delete Chromium logic**:
    - Remove `Cr` parameter from all training functions
    - Delete Cr/NoCr dataset variants in data loading
    - Remove weight weighting logic for Cr (lines 138-156 in current tuners.py)
    - Update `notes/README.md` to document removal

15. **Create CLI entry point** in `src/builder/training/main.py`:
    ```
    python -m src.builder.training.main train --config config/training.yaml --stage lower
    python -m src.builder.training.main train --config config/training.yaml --stage upper
    python -m src.builder.training.main tune --config config/training.yaml --stage lower --n-trials 100
    python -m src.builder.training.main tune --config config/training.yaml --stage upper --n-trials 100
    ```

---

### **Phase 6: Migration & Documentation**

16. **Update `src/builder/training/loadTrainData.py`**:
    - Keep tar.gz unpacking but refactor to return raw numpy arrays
    - New function: `load_arrays_from_bundle(bundle_path)` → dict with features, labels, ml_indexer

17. **Keep backward compatibility**:
    - Preserve old `tuners.py` temporarily as `tuners_legacy.py` in `Legacy/`
    - Document migration path in `ChangeLogs/ChangeLogV1.md`

18. **Create `config/training.yaml` template**:
    ```yaml
    lower_model:
      encoderLayerUp: [1, 2, 3]         # Optuna search range
      encoderLayerDown: [0, 1]
      low_regularization: [none, dropout0.2, batchnorm]
      activation_leak: 0.05
    
    upper_model:
      middleLayerUp: [1, 2, 3]
      middleLayerDown: [0, 1]
      high_regularization: [batchnormdropout0, layernormdropout0]
      highWD: [0, 1e-6, 1e-5, 1e-4]
      noise: [0, 0.01, 0.05]
    
    training:
      batch_size: 1024
      learning_rate: 1e-3
      epochs: 20
      early_stopping_patience: 2
    
    optuna:
      n_trials: 100
      study_name: "nMELTS_lower_optuna"
      sampler: "TPESampler"
      pruner: "MedianPruner"
    
    training_strategy:
      stages:
        - name: "lower"
          freeze_encoder: false
        - name: "upper"
          freeze_encoder: true
        - name: "finetune"
          freeze_encoder: false
    ```

---

## Verification

**Testing approach:**
1. **Unit tests** for each module:
   - `test_dataset.py` - MLDataset shape, indexing
   - `test_model_factory.py` - checkpoint save/load with ml_indexer roundtrip
   - `test_loss_factory.py` - loss computation shapes and masking
   - `test_trainers.py` - single epoch forward pass, gradient flow
   
2. **Integration tests**:
   - Load sample training bundle → create dataloader
   - Create lower model → train 1 epoch → checkpoint roundtrip
   - Load checkpoint → verify ml_indexer is present
   - Create upper model with frozen lower encoder → verify gradients blocked
   - Run single Optuna trial → verify objective function returns scalar

---

## Decisions

- **Optuna wrapper approach chosen**: Minimal refactoring (wrap existing training loops) rather than full rewrite for lower risk
- **ml_indexer serialization**: Full pickle in .pt files maintains backward compatibility and preserves all transformation matrices
- **Unified config**: Single training.yaml simplifies deployment (vs separate lower/upper files)
- **Encoder freezing strategy**: Support both frozen-encoder upper training (fast iteration) and unfrozen fine-tuning (convergence)
- **Chromium removal**: Complete—no Cr variants, weights, or data filtering
- **Module granularity**: 8 modules in `src/builder/training/` for clear separation of concerns (data, loss, optimizer, training logic)
