# PC vs MAC Script Comparison

Covers two file pairs:
- `scripts/MELTedGEOROCsepCr.py` (PC) vs `scripts/MELTedGEOROCsepCrMAC.py` (MAC)
- `src/builder/alphamelts/engine/RandomMelters.py` (PC) vs `src/builder/alphameltsMAC/engineMAC/RandomMelters20.py` (MAC)

---

## 1. Top-level scripts

### Import source
The PC script imports from the Linux-targeting `alphamelts` package; the MAC script imports from the parallel `alphameltsMAC` package.

| | PC | MAC |
|---|---|---|
| Import | `builder.alphamelts.engine.RandomMelters` | `builder.alphameltsMAC.engineMAC.RandomMelters20` |

### Models run
The PC script is a focused single-model run; the MAC script iterates over all three model families.

| | PC | MAC |
|---|---|---|
| `MELTSmodels` | `['120']` | `['102', '120', 'p']` |
| `startTs` | `[1800, 1800]` (2 entries) | `[1800, 1800, 1800]` (3 entries) |
| `input_liquid_fractions` | `[102, 102]` | `[102, 102, 102]` |

### Cr/NoCr tag loop
The PC script generates only Cr-bearing datasets; the MAC script generates both NoCr and Cr datasets in sequence.

| | PC | MAC |
|---|---|---|
| Tag loop | `['Cr']` only | `['NoCr', 'Cr']` |

### `allowed_phases` list
The only difference is handling of `cristobalite` for non-pMELTS models:

| | PC | MAC |
|---|---|---|
| `cristobalite` | Present but commented out (`#'cristobalite'`) | Absent from list entirely |

### Oxygen string
The Oxygen argument passed to `alphaMELTScooling` differs in capitalisation. This matters because the PC version of `RandomMelters.py` uses strict equality (`== 'Closed'`), while the MAC version uses `.lower()`.

| | PC | MAC |
|---|---|---|
| `Oxygen` value | `'Closed'` | `'closed'` |

### Other minor differences

| Item | PC | MAC |
|---|---|---|
| Date label (`input_date`) | `'June17closed'` | `'June10closed'` |
| `alphaMELTSLocation` variable | Defined (path to Linux binary) | Not present |
| Sleep block | Not present | Commented out (`time.sleep(3600 * 40)`) |

---

## 2. RandomMelters engine files

### `_process_compositions` — CO2 handling

The PC version adds a `replace_CO2` parameter (default `True`) and prints a diagnostic message when CO2 is assigned. The MAC version always assigns CO2 and has no guard flag or print.

**PC** (`RandomMelters.py:79`):
```python
def _process_compositions(..., zeroOxides=['MnO', 'NiO'], replace_CO2=True):
    ...
    if (MELTSModel == '120') and ('CO2' not in zeroOxides) and replace_CO2:
        ...
        print(f"Adding CO2 to {allWet} hydrous/soaked compositions ...")
```

**MAC** (`RandomMelters20.py:79`):
```python
def _process_compositions(..., zeroOxides=['MnO', 'NiO']):  # no replace_CO2
    ...
    if (MELTSModel == '120') and ('CO2' not in zeroOxides):  # always runs
        ...
        # no print
```

### `alphaMELTScooling` — Fe³⁺ / fO₂ partitioning algorithm

This is the most significant algorithmic difference. The two versions use different methods to convert a random fO₂ offset into a ferric iron proportion.

**PC** — samples from a discrete exponential distribution (values concentrated near R=0):
```python
R = np.linspace(0, 0.15, 100)
P = np.exp(-20 * R)
P = P / P.sum()
R_chosen = np.random.choice(R, size=length, replace=True, p=P)
ferric = conditions[:, col_dict['FeO']] * R_chosen * (1 / ferric_to_ferrous)
```

**MAC** — derives R from a log-linear fO₂ draw (thermodynamically motivated):
```python
logfo2 = np.random.uniform(-5, 5, size=length)
logR = (logfo2 * 0.2) - 1
R32 = 10 ** logR
R_chosen = R32 / (R32 + 1)   # Fe³⁺ / Fe_total molar ratio
ferric = conditions[:, col_dict['FeO']] * R_chosen * (1 / ferric_to_ferrous)
```

The MAC approach produces a physically coupled Fe³⁺/FeT ratio from fO₂; the PC approach samples R directly from an ad-hoc exponential prior.

### `alphaMELTScooling` — Oxygen string comparison

| | PC | MAC |
|---|---|---|
| Comparison style | Strict equality (`== 'Closed'`, `== 'Open'`) | Case-insensitive (`.lower() == 'closed'`) |

Because the top-level PC script passes `'Closed'` and the PC engine expects exactly `'Closed'`, these are consistent. The MAC script passes `'closed'` and the MAC engine uses `.lower()`, so they are also consistent — but the two pairs are not cross-compatible without adjustment.

### `alphaMELTScooling` — `replace_CO2` propagation

The PC cooling function accepts and forwards `replace_CO2` to `_process_compositions`; the MAC version does not have this parameter.

**PC** (`RandomMelters.py:292`):
```python
def alphaMELTScooling(..., replace_CO2=True):
    ...
    compositions = _process_compositions(..., replace_CO2=replace_CO2)
```

**MAC** (`RandomMelters20.py:296`): `replace_CO2` parameter absent throughout.

### `alphaMELTScompress` — column label bug (MAC)

The third column in the compression run is assigned as an `fO2` offset (`np.random.uniform(-5, 5)`), but the key arrays differ:

| | PC | MAC |
|---|---|---|
| Third key label | `'fO2'` ✓ | `'Fe2O3'` ✗ (mismatch with assigned values) |

This appears to be a labelling bug in the MAC version.

### `alphaMELTSERph` function — PC only

The PC engine contains a third public function, `alphaMELTSERph`, which runs ensemble MELTS calculations driven by external settings and batch files (e.g. for HeFESTo-style workflows). This function does not exist in the MAC engine.

---

## Summary table

| Difference | PC | MAC |
|---|---|---|
| Engine package | `alphamelts` | `alphameltsMAC` |
| Models run by script | `['120']` | `['102', '120', 'p']` |
| Cr/NoCr datasets | Cr only | Both NoCr and Cr |
| `replace_CO2` guard | Yes | No |
| CO2 diagnostic print | Yes | No |
| Fe³⁺ partitioning | Discrete exponential prior on R | Log-linear from fO₂ draw |
| Oxygen string matching | Strict case | Case-insensitive `.lower()` |
| `compress` third-column key | `'fO2'` (correct) | `'Fe2O3'` (mislabelled) |
| `alphaMELTSERph` function | Present | Absent |
| `cristobalite` in phases | Commented out | Removed entirely |
