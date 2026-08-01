# DARWIN Phase 0 — Project README

**Last updated:** pipeline finalized, all four datasets verified,
preprocessing confirmed, caching script ready.

---

## What this project is

DARWIN (Dynamic Adaptive Representation with Weighted INference) is a
calibration-free EEG brain-computer interface framework that handles two
distinct problems separately:

- **Cross-user generalization** (who is this person?) — handled by
  similarity-weighted donor matching (SSRS + FiLM conditioning)
- **Cross-session generalization** (how is this person's signal drifting
  right now?) — handled by a diagonal Kalman filter (DTKF)

Built by a 3-person undergraduate research team. No adversarial training
anywhere in the pipeline. Total model size under 50K parameters.

---

## Project folder structure

```
DARWIN_Phase0/
├── data_loader.py           # MOABB-based loader for all 4 datasets
├── preprocessing.py         # Notch filter, MAD clip, robust normalize
├── test_pipeline.py         # End-to-end check: load + preprocess one subject
├── check_all_datasets.py    # Quick sanity check across all 4 datasets
├── full_pool_preprocess.py  # Full preprocessing + disk caching for all subjects
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── .venv/                   # Virtual environment (not committed to Git)
├── cache/                   # Preprocessed .npz files (not committed to Git)
│   ├── bci_iv_2a/
│   │   └── subject_N.npz
│   ├── bci_iv_2b/
│   ├── openbmi/
│   └── physionet/
└── C-/                      # MOABB raw download cache (not committed to Git)
    └── Users/DELL/mne_data/
```

---

## Environment setup

**Requirements:** Python 3.9–3.14, Windows/Mac/Linux, no GPU needed for
preprocessing (GPU needed later for Phase 0 training).

### Step 1: Create and activate virtual environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the script:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Then retry the activate command. Your prompt should show `(.venv)` when active.

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install dependencies
```powershell
pip install -r requirements.txt
```

Installs: moabb, mne, scipy, numpy (plus their dependencies automatically).

### Step 3: Point VS Code at the environment
`Ctrl+Shift+P` → "Python: Select Interpreter" → pick `.venv\Scripts\python.exe`

---

## Datasets

| Dataset | Subjects | Sessions | Channels | Classes | Notch | Status |
|---|---|---|---|---|---|---|
| BCI-IV 2a | 9 | 2 | 22 | 4 (left, right, feet, tongue) | 50 Hz | Verified |
| BCI-IV 2b | 9 | 5 | 3 (C3/Cz/C4) | 2 (left, right) | 50 Hz | Verified |
| OpenBMI | 54 | 2 | 62 | 2 (left, right) | 60 Hz | Verified (subject 1) |
| PhysioNet | 109 | ~1 | 64 | varies | 60 Hz | Verified |

**Important notes:**
- BCI-IV datasets recorded in Austria → 50 Hz notch
- OpenBMI recorded in South Korea → 60 Hz notch
- PhysioNet recorded in USA → 60 Hz notch
- OpenBMI is capped at 20 subjects in the full-pool run (~24GB download).
  Full 54 subjects = ~66GB which exceeds available disk space.
- MOABB caches raw downloads to `C-\Users\DELL\mne_data\` (a path quirk
  on this machine). First download per subject takes time; subsequent
  loads are instant from cache.

---

## Full preprocessing pipeline (finalized)

```
Raw EEG (C_in × T_native)
tagged with [dataset_id, subject_id, session_id]
  ↓
Resample → 250 Hz (anti-aliased, handled by MOABB at load time,
  saved to disk once — never re-run)
  ↓
CAR — Common Average Reference
  (per-dataset flag, logged; not all datasets need it)
  ↓
Notch filter — 50 or 60 Hz + 2nd harmonic
  (per-dataset based on recording country, bandstop ±2 Hz)
  ↓
Trial/Epoch extraction
  (cue-aligned, fixed length L, handled by MOABB MotorImagery paradigm)
  ↓
MAD artifact clipping
  (per-epoch, robust outlier removal, log % clipped per epoch)
  ↓
Robust amplitude normalization
  (per-epoch median/MAD — NOT whole-recording)
  LOG: pre-normalization median and MAD per epoch as metadata
  (lets you check later whether DTKF drift correlates with
   amplitude changes that were already normalized away here)
  ↓
LSP — Learnable Spatial Projection
  (C_in → C_out, dataset-specific learned matrix,
   Gaussian spatial init, L1 regularized,
   ablate against rigid channel intersection)
  ↓
9-band FIR filter bank
  Band edges (Hz): [1-4, 4-8, 8-13, 13-20, 20-30,
                    30-40, 40-50, 50-65, 65-80]
  All edges capped to minimum dataset Nyquist (80 Hz —
  PhysioNet's native 160 Hz sampling rate sets this ceiling)
  ↓
════════════════════ SPLIT BOUNDARY ════════════════════
  Assign each [subject_id, session_id] → train / val / test
  per protocol (LOSO / LOSSO) BEFORE any windowing.
  Windows must never straddle a train/test boundary.
  Splitting at window level AFTER windowing causes leakage
  with 50% overlap — this is the correct order.
════════════════════════════════════════════════════════
  ↓
Windowing
  (length T, stride T/2 = 50% overlap,
   applied only within each split's already-assigned epochs)
  ↓
Pool window → single 64-dim vector
  (mean pooled across time within each window)
  DTKF updates once per window (NOT per raw sample)
  Q = 0.001 per-window drift assumption
  ↓
Output: (num_windows, 64)
  tagged with [dataset_id, subject_id, session_id,
               split, window_idx]
```

### Why each stage exists

| Stage | Why it's there |
|---|---|
| Resample to 250 Hz | Common rate across all four datasets; PhysioNet's 160 Hz native rate sets the 80 Hz Nyquist ceiling for the filter bank |
| CAR | Removes common-mode noise (powerline, movement artifacts shared across all electrodes) |
| Notch filter | Removes mains electrical interference (50/60 Hz) — a building wiring artifact, nothing to do with brain activity |
| MAD clipping | Robust artifact removal: uses each epoch's own statistics so one wild spike doesn't bias the whole epoch; more robust than fixed-threshold clipping |
| Per-epoch normalization | Strips trivial amplitude-scale drift (electrode impedance, gain) so the model trains on signal shape; DTKF tracks representational drift in feature space later |
| LSP | Replaces rigid channel intersection (which would cap everyone at BCI-IV 2b's 3 channels) with a learned, spatially-initialized projection into a shared 16-dim space |
| 9-band filter bank | Decomposes signal into neurophysiologically meaningful frequency bands (delta, theta, alpha, mu, beta, gamma etc.) before spatial convolution |
| Split boundary here | Prevents data leakage — 50% overlapping windows share raw samples; splitting after windowing leaks test data into training |
| Pool to 64-dim | Produces fixed-size vector per window for DTKF; mean pooling is cheap and effective for slowly-varying drift tracking |

---

## Running the pipeline

### Quick sanity check — one subject, one dataset
```powershell
python test_pipeline.py
```
Loads BCI-IV 2a subject 1, preprocesses, prints diagnostics.
Should complete in under 10 seconds (data already cached).

### Per-dataset check — all four datasets
```powershell
python check_all_datasets.py
```
Loads subject 1 from each dataset and preprocesses. Uses retry logic
for network resilience on large OpenBMI downloads.

### Full preprocessing + caching — all subjects
```powershell
python full_pool_preprocess.py
```
Processes every subject and saves to `cache/<dataset>/subject_N.npz`.
Safe to interrupt and resume — already-cached subjects are skipped.


**Estimated time:** 3-4 hours (dominated by OpenBMI downloads and
PhysioNet's 107 remaining subjects). Keep laptop plugged in and lid open.

---

## What's in each cached .npz file

```python
import numpy as np
data = np.load('cache/bci_iv_2a/subject_1.npz', allow_pickle=True)

data['X']                     # (n_epochs, n_channels, n_times) -- preprocessed
data['y']                     # (n_epochs,) -- class label strings
data['subject']               # (n_epochs,) -- subject ID per epoch
data['session']               # (n_epochs,) -- session label per epoch
data['pct_clipped_per_epoch'] # (n_epochs,) -- artifact diagnostic
```

Note: the cache stores post-notch/clip/normalize epochs, before LSP and
the filter bank. LSP and the filter bank are learned/applied during
training, not during caching — this is intentional so LSP weights can
be updated during training without invalidating the cache.

---

## Verified output shapes

| Dataset | Subject | X shape | Mean clip | Mean | Std |
|---|---|---|---|---|---|
| bci_iv_2a | 1 | (576, 22, 1001) | 0.199% | -0.0046 | 1.0168 |
| bci_iv_2b | 1 | (720, 3, 1126) | 0.288% | 0.0064 | 1.0283 |
| openbmi | 1 | (100, 62, 1000) | 0.737% | -0.0003 | 1.0671 |
| physionet | 1 | (174, 64, 752) | 1.055% | 0.0051 | 1.0910 |

All four datasets show mean near 0 and std near 1 after normalization,
confirming consistent preprocessing behavior across heterogeneous sources.

## .gitignore

```
.venv/
cache/
C-/
__pycache__/
*.pyc
.DS_Store
```

---

## What comes next

### Immediate (before full-pool run)
- [ ] Full-pool preprocessing run across all subjects, all datasets
      (run overnight, ~3-4 hours)

### After full-pool cache is ready — build order

**Stage 1: Learnable Spatial Projection (LSP)**
- PyTorch `nn.Module`, one learned matrix per dataset
- Input: raw epoch `(C_in, T)` per dataset
- Output: `(16, T)` shared channel space
- Gaussian spatial init using real electrode coordinates
- L1 regularization during training
- Ablation: compare against rigid channel intersection

**Stage 2: 9-band FIR filter bank**
- Apply 9 bandpass filters to LSP output
- Band edges: [1-4, 4-8, 8-13, 13-20, 20-30, 30-40, 40-50, 50-65, 65-80] Hz
- All edges capped to 80 Hz (PhysioNet's Nyquist ceiling)
- Output: `(9, 16, T)` — 9 bands × 16 channels × time

**Stage 3: EEGNet-style encoder**
- Temporal convolution per band
- Depthwise spatial convolution across channels
- ~18K parameters total
- Output: feature vector per window

**Stage 4: 64-dim projection head**
- 2-layer MLP → fixed 64-dim output
- This is the unified embedding space — all of SSRS,
  DTKF, and FiLM operate in this same 64-dim space
- Non-negotiable width (changing it breaks DTKF's
  state/measurement consistency by construction)

**Stage 5: Phase 0 — InfoNCE contrastive pretraining**
- Subject-balanced negative sampling
- Train on pooled unlabeled data from all datasets
- Validate with two linear probes before proceeding:
  - Subject-ID probe (expect high accuracy — SSRS needs this)
  - Task/class probe (expect above-chance — if near-chance,
    pretraining learned identity shortcut, fix augmentations)

**Stage 6: Stage 3.5 hypothesis gate**
- Using baseline features, check that cross-subject differences
  look discrete (large, unstructured jumps) while cross-session
  differences look continuous (small, correlated drift)
- Cheap PCA/distance check — if hypothesis doesn't hold,
  know before building SSRS and DTKF around it

**Stage 7: Module A — FiLM conditioning**
- Subject embedding (64-dim) modulates encoder features
- Scale and shift per frequency band
- Backbone frozen from Phase 0; only FiLM + classifier train

**Stage 8: Module B — SSRS**
- Nearest-centroid matching in 64-dim space
- Entropy-gated fallback to population average
- Ablation: with vs without confidence gating

**Stage 9: Module C — DTKF**
- Diagonal Kalman filter, 64 independent scalar trackers
- State and measurement both in same 64-dim space (by construction)
- Updates once per window, not per raw sample
- Innovation residual spike = artifact rejection

**Stage 10: Full ablation study**
- Remove each module individually
- Evaluate under LOSO, LOSSO, cross-dataset protocols
- SSRS bad-init recovery test

---
