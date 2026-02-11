import nMELTS.engine.NN as NN

def symmetric_rel_l1(pred, target, eps=1e-6):
    denom = torch.clamp(torch.abs(pred) + torch.abs(target), min=eps)
    return torch.mean(torch.abs(pred - target) / denom)

def symmetric_rel_l2(pred, target, eps=1e-6):
    denom = torch.clamp(torch.abs(pred) + torch.abs(target), min=eps)
    return torch.mean((pred - target)**2 / denom)



def train_Lower_MELTS(Model, criterion = nn.BCEWithLogitsLoss(), Cr = False, Epochs = 20, lr = 1E-3, device = 'cuda'):
        # --- Build (copy) model ---
    model = NN.MidLevelNetwork(**Model.config).to(device)#.to(Model.device) # Copy the old model, so no overwriting. 

    noise = model.noise

    # freeze chem & mole heads like original code
    for p in model.parameters():
        p.requires_grad = True
    for p in model.chem_heads.parameters():
        p.requires_grad = False
    for p in model.mole_head.parameters():
        p.requires_grad = False

    if Cr:
        binary_train_set, binary_test_set = binary_train_set_Cr, binary_test_set_Cr
        print('Loading Cr')
    else:
        binary_train_set, binary_test_set = binary_train_set_NoCr, binary_test_set_NoCr
        print('Loading Non-Cr')

    if 'dropout' in model.config['low_regularization'].lower():
        dropout_rate = pull_number(model.config['low_regularization'].lower())
        print(f"dropout in {model.config['low_regularization']}: {pull_number(model.config['low_regularization'])}")
    else:
        dropout_rate = 0

    # --- Loaders (same for both "Cr" and "NoCr" if you only want one test here) ---
    batch_size = 1024
    train_loader = DataLoader(binary_train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(binary_test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=model.config['lowWD'])
    best_test_loss = np.inf
    train_losses, test_losses = [], []

    last_missed = False
    # --- Train for specified epochs ---
    for epoch in range(Epochs):
        start = time.time()
        model.train()
        running_train_loss = 0.0

        for xb, yb in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            #bulk_zero_mask = (x_batch != 0).to(torch.float) # Bulk zero mask different shape for training and testing because this mask is doubling as a filter for the noise
            if noise != 0:
                xb = xb + (xb * torch.randn_like(xb) * noise)  # noise injection

            optimizer.zero_grad()
            logits = model.forward_binaries(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * xb.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # --- Evaluate ---
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                logits = model.forward_binaries(xb)
                loss = criterion(logits, yb)
                running_test_loss += loss.item() * xb.size(0)

        avg_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1:02d}: Train {avg_train_loss:.5f} | Test {avg_test_loss:.5f} | Δt={time.time()-start:.1f}s")

        # simple early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'Models/temp_binary_train.pt')
            last_missed = False
        elif epoch > 2 and avg_test_loss > best_test_loss * 1.01:
            print("No improvement; stopping early?")
            if last_missed:
                break
            last_missed = True

        #ADAPTIVE DROPOUT
        if avg_test_loss > avg_train_loss * 1.02:
            anyDropout = False
            if min(dropout_rate + 0.05, 0.6) > dropout_rate:
                old_drop = dropout_rate
                dropout_rate = min(dropout_rate + 0.05, 0.6)
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = dropout_rate
                        anyDropout  = True
                if anyDropout:
                    print(f"Overfitting. Increasing Dropout: {old_drop} -> {dropout_rate}")
            else: 
                print(f"Overfitting, but dropout_rate is at the maximum: {dropout_rate}")


        elif avg_test_loss < avg_train_loss:
            anyDropout = False
            if max(dropout_rate - 0.02, 0) < dropout_rate:
                old_drop = dropout_rate
                dropout_rate = max(dropout_rate - 0.02, 0) 
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = dropout_rate
                        anyDropout  = True
                if anyDropout:
                    print(f"Underfitting. Decreasing Dropout: {old_drop}->{dropout_rate}")

            else:
                print(f"Underfitting, but dropout_rate is at the minimum: {dropout_rate}")

        gc.collect()

        

    model.load_state_dict(torch.load('Models/temp_binary_train.pt'))
    return model, best_test_loss


weight_dict_both = {
    'olivine': 3,
    'nepheline': 15,
    'leucite': 10,
    'alloy-solid': 5,
    'muscovite': 3,
    'k-feldspar': 5
}
# This overprints the above weights, does not apply to NoCr model
wweight_dict_Cr = {
    'rhm-oxide': 5
}

binWeightsNoCr = torch.ones(ml_indexer.nphases, dtype = torch.float32).reshape(1,-1)
binWeightsCr = torch.ones(ml_indexer.nphases, dtype = torch.float32).reshape(1,-1)
compWeightsNoCr = torch.ones(ml_indexer.ncompsVaried, dtype = torch.float32).reshape(1,-1)
compWeightsCr = torch.ones(ml_indexer.ncompsVaried, dtype = torch.float32).reshape(1,-1)
"""for phase, W in weight_dict_both.items(): # Weights removed for 110 due to bad overfitting... Take a look at phase abundances??? 
    binWeightsCr[:,mass_phasedict[phase]] = W
    binWeightsNoCr[:,mass_phasedict[phase]] = W
    if phase in compositionally_variable_phases:
        compWeightsCr[:,comp_phasedict[phase]] = W
        compWeightsNoCr[:,comp_phasedict[phase]] = W
for phase, W in weight_dict_Cr.items():
    binWeightsCr[:,mass_phasedict[phase]] = W
    if phase in compositionally_variable_phases:
        compWeightsCr[:,comp_phasedict[phase]] = W"""


def train_Upper_MELTS(Model, criterion = symmetric_rel_l2, criterion_sat = nn.BCEWithLogitsLoss(), Cr = False, Epochs = 20, lr = 1E-3, device = 'cuda'):
        # --- Build (copy) model ---
    model = NN.MidLevelNetwork(**Model.config).to(device)#.to(Model.device) # Copy the old model, so no overwriting. 

    noise = model.noise

    # Copy Lower Parameters! 
    model.sat_head.load_state_dict(Model.sat_head.state_dict())
    model.encoder.load_state_dict(Model.encoder.state_dict())

    # freeze chem & mole heads like original code
    for p in model.parameters():
        p.requires_grad = True
    for p in model.sat_head.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = False
    

    if Cr:
        full_train_set, full_test_set = full_train_set_Cr, full_test_set_Cr
        print('Loading Cr')
        binWeights, compWeights = binWeightsCr, compWeightsCr
    else:
        full_train_set, full_test_set = full_train_set_NoCr, full_test_set_NoCr
        print('Loading Non-Cr')
        binWeights, compWeights = binWeightsNoCr, compWeightsNoCr

    if 'dropout' in model.config['high_regularization'].lower():
        dropout_rate = pull_number(model.config['high_regularization'].lower())
        print(f"dropout in {model.config['high_regularization']}: {pull_number(model.config['high_regularization'])}")
    else:
        dropout_rate = 0

    # --- Loaders (same for both "Cr" and "NoCr" if you only want one test here) ---
    batch_size = 1024
    train_loader = DataLoader(full_train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(full_test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=model.config['highWD'])
    best_test_loss = np.inf
    train_losses, test_losses = [], []

    binWeights = binWeights.cuda()
    compWeights = compWeights.cuda()

    chem_alpha = 1
    mole_alpha = 1
    bulk_alpha = 0

    criterion_chem = criterion 
    criterion_mole = criterion
    criterion_bulk = criterion

    last_missed = False
    # --- Train for specified epochs ---
    for epoch in range(Epochs):
        start = time.time()
        model.train()
        running_train_loss = 0.0
        running_sat_loss = 0
        running_chem_loss = 0
        running_mole_loss = 0
        running_bulk_loss = 0
        for batch_idx, (x_batch, b_batch, y_batch, m_batch) in enumerate(tqdm(train_loader, desc="Training", leave=False)):

            optimizer.zero_grad()

            x_batch, b_batch, y_batch, m_batch = x_batch.to(device, non_blocking=True), b_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True), m_batch.to(device, non_blocking=True)
            
            bulk_zero_mask = (x_batch != 0).to(torch.float) # Bulk zero mask different shape for training and testing because this mask is doubling as a filter for the noise
            #x_batch = x_batch + torch.randn_like(x_batch) * 0.005 * bulk_zero_mask # Add Small Gaussian Noise to avoid overfitting during training Oct 14: Moved 0.002->0.005
            if noise != 0:
                x_batch = x_batch + (x_batch * torch.randn_like(x_batch) * noise)
            logits, chem_preds, chem_zero_mask, mole_preds, bulk_preds = model(x_batch, binaries=b_batch, NN_only = True)

            # Binary saturation loss
            loss_sat = criterion_sat(logits, b_batch) 

            # Chemistry and mass losses: apply masks
            chem_loss_raw = criterion_chem(chem_preds, y_batch)
            mole_loss_raw = criterion_mole(mole_preds, m_batch)
            bulk_loss_raw = criterion_bulk(bulk_preds, x_batch[:,3:])
            #mole_zero_mask = (torch.sigmoid(logits) > 0.5).to(torch.float).detach() # Only use binary preds for masking when those neurons are free
            mole_zero_mask = (b_batch > 0.5).to(torch.float).detach()

            """if batch_idx == 0:
                print(chem_loss_raw.device)
                print(chem_zero_mask.device)
                print(compWeights.device)"""

            chem_loss_masked = (chem_loss_raw * chem_zero_mask * compWeights).sum() / (chem_zero_mask * compWeights).sum().clamp(min=1)
            mole_loss_masked = (mole_loss_raw * mole_zero_mask * binWeights).sum() / (mole_zero_mask * binWeights).sum().clamp(min=1)
            bulk_loss_masked = (bulk_loss_raw * bulk_zero_mask[:,3:]).sum() / (bulk_zero_mask[:,3:]).sum().clamp(min=1)
            
            running_sat_loss += loss_sat.item()
            running_mole_loss += mole_loss_masked.item()
            running_chem_loss += chem_loss_masked.item()
            running_bulk_loss += bulk_loss_masked.item()
            
            
            
            loss = loss_sat + chem_alpha*chem_loss_masked + mole_alpha*mole_loss_masked + bulk_alpha*bulk_loss_masked
            #print(f"Sat loss: {loss_sat.item()}, Chem loss: {chem_loss_masked.item()}")

            if not torch.isfinite(loss):
                print("Non-finite loss detected!")
                print(f"Sat loss: {loss_sat.item()}, Chem loss: {chem_loss_masked.item()}, Mole loss: {mole_loss_masked}, Bulk loss: {bulk_loss_masked}")
                continue

            loss.backward()


            optimizer.step()

            running_train_loss += loss.item() * x_batch.size(0)

            #if batch_idx % 200 == 0:
                #percent_done = 100 * batch_idx / len(train_loader)
                #train_losses.append(loss.item())
                #print(f"[{percent_done:>5.1f}%] Batch {batch_idx:>5d} Loss: {loss.item():.4f}")

            if (batch_idx * batch_size) > 1E6: # (Only 1 million samples per epoch when training)
                break

        avg_train_loss = running_train_loss / (batch_idx*batch_size)

        print(f"[TRAIN] Running Saturation Loss: {running_sat_loss/(batch_idx*batch_size):.3e}\tRunning Chem Loss: {running_chem_loss/(batch_idx*batch_size):.3e}")
        print(f"[TRAIN] Running Molar Loss: {running_mole_loss/(batch_idx*batch_size):.3e}\tRunning Bulk Loss: {running_bulk_loss/(batch_idx*batch_size):.3e}")

        # ---- Evaluation ----
        model.eval()
        running_test_loss = 0.0
        running_sat_loss = 0
        running_chem_loss = 0
        running_mole_loss = 0
        running_bulk_loss = 0
        
        with torch.no_grad():
            for batch_idx, (x_batch, b_batch, y_batch, m_batch) in enumerate(test_loader):
                x_batch, b_batch, y_batch, m_batch = x_batch.to(device, non_blocking=True), b_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True), m_batch.to(device, non_blocking=True)
                logits, chem_preds, chem_zero_mask, mole_preds, bulk_preds = model(x_batch, binaries=b_batch, NN_only = True)
                
                # Binary saturation loss
                loss_sat = criterion_sat(logits, b_batch)
                
                # Chemistry losses: apply masks
                chem_loss_raw = criterion_chem(chem_preds, y_batch)
                mole_loss_raw = criterion_mole(mole_preds, m_batch)
                bulk_loss_raw = criterion_bulk(bulk_preds, x_batch[:,3:])
                bulk_zero_mask = (x_batch[:,3:] != 0).to(torch.float)
                #mole_zero_mask = (torch.sigmoid(logits) > 0.5).to(torch.float).detach()
                mole_zero_mask = (b_batch > 0.5).to(torch.float)
                
                chem_loss_masked = (chem_loss_raw * chem_zero_mask*compWeights).sum() / (chem_zero_mask*compWeights).sum().clamp(min=1)
                mole_loss_masked = (mole_loss_raw * mole_zero_mask*binWeights).sum() / (mole_zero_mask*binWeights).sum().clamp(min=1)
                bulk_loss_masked = (bulk_loss_raw * bulk_zero_mask).sum() / bulk_zero_mask.sum().clamp(min=1)
                
                running_sat_loss += loss_sat.item()
                running_mole_loss += mole_loss_masked.item()
                running_chem_loss += chem_loss_masked.item()
                running_bulk_loss += bulk_loss_masked.item()
                
                # Total loss
                #loss = loss_sat + chem_alpha*chem_loss_masked
                #loss = mole_alpha*mole_loss_masked + bulk_alpha*bulk_loss_masked
                loss = loss_sat + chem_alpha*chem_loss_masked + mole_alpha*mole_loss_masked + bulk_alpha*bulk_loss_masked
                

                running_test_loss += loss.item() * x_batch.size(0)
               


        print(f"[TEST] Running Saturation Loss: {running_sat_loss/len(full_test_set):.3e}\tRunning Chem Loss: {running_chem_loss/len(full_test_set):.3e}")
        print(f"[TEST] Running Molar Loss: {running_mole_loss/len(full_test_set):.3e}\tRunning Bulk Loss: {running_bulk_loss/len(full_test_set):.3e}")

        
        avg_test_loss = running_test_loss / len(full_test_set)
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch+1:02d}: Train {avg_train_loss:.5f} | Test {avg_test_loss:.5f} | Δt={time.time()-start:.1f}s")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'Models/temp_upper_train.pt')
            last_missed = False
        elif epoch > 2 and avg_test_loss > best_test_loss * 1.01:
            print("No improvement; stopping early?")
            if last_missed:
                break
            last_missed = True
        

        #ADAPTIVE DROPOUT
        if avg_test_loss > avg_train_loss * 1.02:
            anyDropout = False
            if min(dropout_rate + 0.05, 0.6) > dropout_rate:
                old_drop = dropout_rate
                dropout_rate = min(dropout_rate + 0.05, 0.6)
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = dropout_rate
                        anyDropout  = True
                if anyDropout:
                    print(f"Overfitting. Increasing Dropout: {old_drop} -> {dropout_rate}")
            else: 
                print(f"Overfitting, but dropout_rate is at the maximum: {dropout_rate}")


        elif avg_test_loss < avg_train_loss:
            anyDropout = False
            if max(dropout_rate - 0.02, 0) < dropout_rate:
                old_drop = dropout_rate
                dropout_rate = max(dropout_rate - 0.02, 0) 
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = dropout_rate
                        anyDropout  = True
                if anyDropout:
                    print(f"Underfitting. Decreasing Dropout: {old_drop}->{dropout_rate}")

            else:
                print(f"Underfitting, but dropout_rate is at the minimum: {dropout_rate}")

        gc.collect()

        

    model.load_state_dict(torch.load('Models/temp_upper_train.pt'))
    return model, best_test_loss


def tune_Upper_MELTS(Model, Param_Dict=None, Cr=False, Epochs=7, best_loss = None):
    """
    Function to 
     lower binary saturation model. Initializes new model if none given.
    Best to generate one and give it a description.
    Returns model with best parameters, with the same weights as before.
    """

    import numpy as np
    from copy import deepcopy

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
        Model, best_loss = train_Upper_MELTS(Model, Cr=Cr, Epochs=Epochs)

    results = [{'model': Model.config, 'loss': best_loss}]
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

                Model, trial_loss = train_Upper_MELTS(Model, Cr=Cr, Epochs=Epochs)
                results.append({'model': Model.config, 'loss': trial_loss})

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

                Model, trial_loss = train_Upper_MELTS(Model, Cr=Cr, Epochs=Epochs)
                results.append({'model': Model.config, 'loss': trial_loss})

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

                Model, trial_loss = train_Upper_MELTS(Model, Cr=Cr, Epochs=Epochs)
                results.append({'model': Model.config, 'loss': trial_loss})

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


# AI REVISED:
def tune_Lower_MELTS(Model=None, Param_Dict=None, Cr=False, Epochs=7):
    """
    Function to 
     lower binary saturation model. Initializes new model if none given.
    Best to generate one and give it a description.
    Returns model with best parameters, with the same weights as before.
    """

    import numpy as np
    from copy import deepcopy

    # === Default model if none given ===
    if Model is None:
        Model = NN.MidLevelNetwork(
            encoderLayerUp=1,
            encoderLayerDown=0,
            low_regularization='layernormdropout0',
            description=f"MELTS {MELTSModel}, {CalcType}, {'Cr' if Cr else 'NoCr'}, {date}",
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
    Model, best_loss = train_Lower_MELTS(Model, Cr=Cr, Epochs=Epochs)
    results = [{'model': Model.config, 'loss': best_loss}]
    best_config = deepcopy(Model.config)
    best_weights = Model.state_dict()

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
                Model, trial_loss = train_Lower_MELTS(Model, Cr=Cr, Epochs=Epochs)
                results.append({'model': Model.config, 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"✅ Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
                    best_config = deepcopy(working_config)
                    best_loss = trial_loss
                    best_weights = Model.state_dict()

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
                Model, trial_loss = train_Lower_MELTS(Model, Cr=Cr, Epochs=Epochs)
                results.append({'model': Model.config, 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"✅ Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
                    best_config = deepcopy(working_config)
                    best_loss = trial_loss
                    best_weights_WD = Model.state_dict() # Save state dict without loading it for the next WD trial for fairness
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
                Model, trial_loss = train_Lower_MELTS(Model, Cr=Cr, Epochs=Epochs)
                results.append({'model': Model.config, 'loss': trial_loss})

                if trial_loss < best_loss:
                    print(f"✅ Improved! Loss {trial_loss:.4e} < {best_loss:.4e}")
                    best_config = deepcopy(working_config)
                    best_loss = trial_loss
                    best_weights = Model.state_dict()

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

    return Model, results




