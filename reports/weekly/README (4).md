# DARWIN — Phase 0

**D**ynamic **A**daptive **R**epresentation with **W**eighted **IN**ference

A calibration-free EEG brain-computer interface (BCI) framework that separates *who is this person* from *how is their signal drifting right now*, and solves each with a dedicated, lightweight module.

![Status](https://img.shields.io/badge/phase-0%20%2F%20SSL%20pretraining-blue)
![Params](https://img.shields.io/badge/model%20size-%3C50K%20params-success)
![Team](https://img.shields.io/badge/team-3%20undergrads-informational)
![Adversarial training](https://img.shields.io/badge/adversarial%20training-none-lightgrey)

> **Last updated:** Phase 0 self-supervised pretraining pipeline (sampler, augmentations, LSP, encoder, projection head, InfoNCE loss, training loop) is fully built and verified end-to-end on real cached data. LSP spatial init is confirmed for 2 of 4 datasets (`bci_iv_2a`, `bci_iv_2b`); PhysioNet and OpenBMI channel names are pending their downloads finishing. A synthetic-scale smoke test passed on `bci_iv_2b`; a full `bci_iv_2a` training run is in progress.

---

## Table of Contents

- [What This Project Is](#what-this-project-is)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Phase 0 Pipeline](#phase-0-pipeline)
- [Running the Pipeline](#running-the-pipeline)
- [Verified Results](#verified-results)
- [Known Issues & Decisions](#known-issues--decisions)
- [Roadmap](#roadmap)
- [Team](#team)

---

## What This Project Is

DARWIN handles two distinct generalization problems in EEG-based BCIs **separately**, rather than asking one model to solve both at once:

| Problem | Question it answers | Module |
|---|---|---|
| **Cross-user generalization** | Who is this person? | Similarity-weighted donor matching (SSRS) + FiLM conditioning |
| **Cross-session generalization** | How is this person's signal drifting *right now*? | Diagonal Kalman filter (DTKF) |

**Design constraints:** built by a 3-person undergraduate research team · no adversarial training anywhere in the pipeline · total model size under 50K trainable parameters.

This repository covers **Phase 0**: the self-supervised pretraining stage that builds the shared 64-dimensional embedding space every later module (SSRS, DTKF, FiLM) depends on.

## System Architecture

![DARWIN system architecture](assets/architecture_overview.svg)

Phase 0 (this repo) is the foundation layer. Once it's validated by the Gate 1 / Gate 2 probes (see [Roadmap](#roadmap)), Modules A–C are built on top of the frozen embedding space.

## Repository Structure

```
DARWIN_Phase0/
├── data_loader.py            # MOABB-based loader for all 4 datasets
├── preprocessing.py          # Notch filter, MAD clip, robust normalize
├── test_pipeline.py          # End-to-end check: load + preprocess one subject
├── check_all_datasets.py     # Quick sanity check across all 4 datasets
├── full_pool_preprocess.py   # Full preprocessing + disk caching for all subjects
├── get_channel_names.py      # Pulls real channel names/order from MOABB raw data
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── phase0/
│   ├── sampler.py             # Subject-balanced P×K batch sampler
│   ├── augmentations.py       # 7-augmentation pipeline (spec Section 6)
│   ├── lsp.py                 # Per-dataset channel mixer (LSP)
│   ├── encoder.py              # EEGNet 9-band spectral encoder
│   ├── projection_head.py      # 2-layer MLP → 64-dim projection
│   ├── losses.py                # InfoNCE with subject-balanced negatives
│   └── train_phase0.py           # Full training loop, wires everything together
│
├── .venv/                     # Virtual environment (not committed)
├── cache/                     # Preprocessed .npz files (not committed)
│   ├── bci_iv_2a/subject_N.npz
│   ├── bci_iv_2b/
│   ├── openbmi/
│   └── physionet/
├── phase0_checkpoint_*.pt      # Trained LSP + encoder + proj head weights (per dataset)
└── C-/Users/DELL/mne_data/     # MOABB raw download cache (not committed)
```

## Environment Setup

**Requirements:** Python 3.9–3.14 · Windows / Mac / Linux · GPU not required for preprocessing, strongly recommended for Phase 0 training (local reference setup: RTX 4060, 8GB VRAM — sufficient for single-dataset runs at current batch sizes).

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks the script:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Your prompt should show `(.venv)` once active.

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
This installs `moabb`, `mne`, `scipy`, `numpy`, and their dependencies.

For Phase 0 training specifically (not yet in `requirements.txt` — add before sharing this repo):
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Use the `cu121` build for a local NVIDIA GPU. The CPU-only build (`--index-url https://download.pytorch.org/whl/cpu`) works everywhere but is far slower beyond smoke tests.

### 3. Point your IDE at the environment

VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv\Scripts\python.exe`

## Datasets

![Dataset comparison](assets/dataset_comparison.png)

| Dataset | MOABB Class | Subjects | Sessions | EEG Channels | Classes | Notch |
|---|---|---|---|---|---|---|
| BCI-IV 2a | `BNCI2014_001` | 9 | 2 | 22 | 4 (left, right, feet, tongue) | 50 Hz |
| BCI-IV 2b | `BNCI2014_004` | 9 | 5 | 3 (C3/Cz/C4) | 2 (left, right) | 50 Hz |
| OpenBMI | `Lee2019_MI` | 54 (capped at 20) | 2 | 62 | 2 (left, right) | 60 Hz |
| PhysioNet | `PhysionetMI` | 109 | ~1 | 64 | varies | 60 Hz |

**Notes:**
- BCI-IV datasets recorded in Austria → 50 Hz notch; OpenBMI (South Korea) and PhysioNet (USA) → 60 Hz notch.
- OpenBMI is capped at 20 subjects in the full-pool run (~24 GB download). The full 54 subjects (~66 GB) exceeds available disk space.
- MOABB caches raw downloads to `C-\Users\DELL\mne_data\` (a path quirk on the development machine). First download per subject takes time; subsequent loads are instant from cache.
- Raw MOABB channel lists include EOG/STI channels (e.g. `bci_iv_2a` is 26 raw channels, 22 after EEG-only filtering). MOABB's `MotorImagery` paradigm filters these automatically before caching, so cached `X` arrays already reflect the EEG-only counts above.

## Phase 0 Pipeline

The full pipeline runs in two stages: a **preprocessing stage** (writes to disk once, cached) and a **Phase 0 representation-learning stage** (runs at train time, on top of the cache).

![Phase 0 pipeline flow](assets/pipeline_flow.svg)

### Why each stage exists

| Stage | Why it's there |
|---|---|
| Resample to 250 Hz | Common rate across all four datasets; PhysioNet's 160 Hz native rate sets the 80 Hz Nyquist ceiling for the filter bank |
| CAR | Removes common-mode noise (powerline, movement artifacts shared across all electrodes) |
| Notch filter | Removes mains electrical interference (50/60 Hz) — a wiring artifact, unrelated to brain activity |
| MAD clipping | Robust artifact removal using each epoch's own statistics, so one wild spike doesn't bias the whole epoch |
| Per-epoch normalization | Strips trivial amplitude-scale drift so the model trains on signal shape |
| Subject-balanced sampling | Guarantees InfoNCE always has valid intra-subject positive pairs and cross-subject negatives, by construction |
| Augmentations (7, fixed order) | Defines what variation the encoder should learn to ignore, while still recognizing the underlying trial |
| LSP | Replaces rigid channel intersection (which would cap everyone at BCI-IV 2b's 3 channels) with a learned, spatially-initialized projection into a shared 16-dim space |
| 9-band spectral decomposition | Decomposes signal into physiological frequency bands before spatial/temporal convolution; matches what `freq_mask` perturbs, by construction |
| Projection head (64-dim) | Produces the single unified embedding space every later module (SSRS, DTKF, FiLM) depends on — width is spec-locked |
| InfoNCE, subject-balanced negatives | Teaches subject identity to be linearly separable (Gate 1) without letting it become the sole discriminative signal, which would destroy task structure (Gate 2) |

## Running the Pipeline

**Quick sanity check — one subject, one dataset**
```bash
python test_pipeline.py
```

**Per-dataset check — all four datasets**
```bash
python check_all_datasets.py
```

**Full preprocessing + caching — all subjects**
```bash
python full_pool_preprocess.py
```
Safe to interrupt and resume — already-cached subjects are skipped. Verified in practice: survived a mid-run power loss with zero data loss.

**Pull real channel names for LSP spatial init**
```bash
python get_channel_names.py
```
Prints exact channel names, in cache order, for each dataset — required by `lsp.py`'s Gaussian spatial initialization. Already run and confirmed for `bci_iv_2a` and `bci_iv_2b`; re-run once PhysioNet/OpenBMI finish downloading and paste the output into `lsp.py`'s `DATASET_CHANNEL_NAMES`.

**Individual component smoke tests** — each file in `phase0/` is independently runnable and self-verifying:
```bash
python phase0\sampler.py              # subject-balanced batch composition check
python phase0\augmentations.py        # shape/NaN checks on real cached epochs
python phase0\lsp.py                  # LSP param count + shape checks (bci_iv_2a, bci_iv_2b)
python phase0\encoder.py              # EEGNet param count (~17.3K) + shape checks
python phase0\projection_head.py      # projection head param count + determinism check
python phase0\losses.py               # InfoNCE unit tests (identical vs random pairs)
```

**Full training run**
```bash
python phase0\train_phase0.py --dataset bci_iv_2a --batch_size 16 --max_epochs 50
```
- `--dataset`: currently `bci_iv_2a` or `bci_iv_2b` only (the only datasets with confirmed real channel names for LSP's spatial init so far)
- `--batch_size`: 16 is the practical ceiling for a single BCI-IV dataset alone (9 subjects × `k_per_subject=2` → max 8 subjects/batch fits comfortably; the spec's ≥64 target needs ≥32 subjects, which requires pooling in PhysioNet/OpenBMI once available)
- Trains until InfoNCE loss plateaus (10 epochs without ≥1e-4 improvement) or `--max_epochs` is hit, whichever comes first
- Saves `phase0_checkpoint_<dataset>.pt` containing LSP + encoder + projection head weights

### Cache contents (`cache/<dataset>/subject_N.npz`)

```python
import numpy as np
data = np.load('cache/bci_iv_2a/subject_1.npz', allow_pickle=True)

data['X']                       # (n_epochs, n_channels, n_times) -- preprocessed
data['y']                       # (n_epochs,) -- class label strings
data['subject']                 # (n_epochs,) -- subject ID per epoch
data['session']                 # (n_epochs,) -- session label per epoch
data['pct_clipped_per_epoch']   # (n_epochs,) -- artifact diagnostic
```
Cache stores post-notch/clip/normalize epochs, **before** LSP and the filter bank — both are applied at train time in `phase0/`, so LSP weights can update without invalidating cached data.

### Checkpoint contents (`phase0_checkpoint_*.pt`)

```python
import torch
ckpt = torch.load('phase0_checkpoint_bci_iv_2a.pt', weights_only=False)

ckpt['lsp_state_dict']               # trained LSP weights for this dataset
ckpt['encoder_state_dict']           # trained EEGNet encoder weights
ckpt['projection_head_state_dict']   # trained projection head weights
ckpt['dataset_key']                  # which dataset this checkpoint was trained on
ckpt['channel_names']                # channel order used for this dataset's LSP
ckpt['loss_history']                 # InfoNCE loss per epoch
ckpt['final_loss']                   # last epoch's mean loss
```

## Verified Results

### Preprocessing output (unchanged from prior verification)

| Dataset | Subject | X shape | Mean clip | Mean | Std |
|---|---|---|---|---|---|
| bci_iv_2a | 1 | (576, 22, 1001) | 0.199% | -0.0046 | 1.0168 |
| bci_iv_2b | 1 | (720, 3, 1126) | 0.288% | 0.0064 | 1.0283 |
| openbmi | 1 | (100, 62, 1000) | 0.737% | -0.0003 | 1.0671 |
| physionet | 1 | (174, 64, 752) | 1.055% | 0.0051 | 1.0910 |

### Phase 0 component parameter counts

![Parameter breakdown](assets/param_breakdown.png)

| Component | Trainable params | Verified shape |
|---|---|---|
| LSP (bci_iv_2a) | 352 (16×22) | (22, 1001) → (16, 1001) |
| LSP (bci_iv_2b) | 48 (16×3) | (3, 1001) → (16, 1001) |
| LSP total (2/4 datasets) | 400 | — |
| EEGNet 9-band encoder | 17,302 | (8, 16, 1001) → (8, 64) |
| Projection head | 16,832 | (8, 64) → (8, 64) |
| **Full trainable stack (per dataset)** | **~34.5K** | — |

### Build status

![Phase 0 progress](assets/progress_status.png)

## Known Issues & Decisions

| Issue | Decision |
|---|---|
| MOABB caches to nested path `C-\...` inside project folder | Cosmetic quirk, doesn't affect correctness. Added to `.gitignore`. |
| OpenBMI full dataset (~66GB) exceeds available disk space | Capped at 20 subjects (~24GB). Increase cap later if space is freed. |
| OpenBMI downloads throttled on campus WiFi | Use a phone hotspot for OpenBMI downloads specifically. |
| CAR and notch applied at epoch level, not raw signal level | Acceptable simplification. Raw-level CAR/notch is a future improvement. |
| `W_dataset` shape ambiguity — spec Section 3 says `(C_dataset × 16)`, Section 4's matmul only works as `(16 × C_dataset)` | Went with `(16, C_dataset)` — consistent with the matmul equation and Gaussian-init description. Needs a sync with whoever owns spec Section 3's notation. |
| LSP param count mismatch — spec estimates ~1K total, actual dense-matrix count across all 4 datasets ≈ 2,432 | Proceeding with full dense per-dataset matrices (matches spec Section 4's exact pseudocode). Possibly the spec assumed fewer datasets or a low-rank factorization — worth confirming before treating as final. |
| Augmentation order bug, found and fixed — an earlier version reasoned the order from pipeline-design intuition, not re-checked against spec Section 6's explicit table | Fixed to: `time_shift → amplitude_scale → gaussian_noise → channel_dropout → time_mask → temporal_crop → freq_mask`. Re-verified on real data — shapes/NaN checks unaffected, since only sequencing changed. |
| 9-band edges not given exactly by spec — only `max_bands_to_mask=2` is specified, not the Hz boundaries | Used standard EEG sub-band convention (delta/theta/alpha/beta/gamma split finer to reach 9 bands, 0.5–45 Hz). `freq_mask` and `encoder.py`'s spectral decomposition import the same band list (`EEG_9_BANDS_HZ`) so they can't silently drift apart — but exact edges are a design choice, not spec-verified. |
| Channel name order for LSP assumes MOABB's `MotorImagery` paradigm filters EOG/STI while preserving original electrode order | Channel counts match cache exactly (22/3) for bci_iv_2a/2b under this assumption; not independently re-verified against `preprocessing.py`'s internals beyond confirming no channel-selection logic lives there. |
| InfoNCE temperature = 0.1 — not specified anywhere in spec v3 | Used as the standard SimCLR/contrastive-learning default. Treat as tunable once real Gate 1/Gate 2 results are available — no evidence yet it's right for EEG specifically. |
| Same-subject-different-clip pairs in InfoNCE — spec defines intra-subject positives and cross-subject negatives, but doesn't address same-subject-different-trial pairs that `sampler.py`'s P×K batching also produces | Excluded from the loss entirely (neither positive nor negative) — treating them as negatives would fight Gate 1 (subject-ID separability); treating them as positives would conflate subject-level and trial-level structure. Reasonable interpretation, not spec-explicit — worth a second opinion. |
| Multi-dataset pooled training not yet built — spec wants training pooled across all datasets in one run; `sampler.py`/`train_phase0.py` currently support one dataset per batch/run only | Real extension needed: route each dataset's clips through its own LSP layer before merging into a shared batch. Not built yet — training is currently per-dataset, with checkpoints saved separately. Needs to be built regardless of when PhysioNet/OpenBMI finish downloading. |
| `--batch_size` currently capped below spec's ≥64 for single-dataset runs | `bci_iv_2a`/`2b` only have 9 subjects each; with `k_per_subject=2`, max batch is 16 (8 subjects). True ≥64 batches need ≥32 subjects, achievable once multi-dataset pooling is built and PhysioNet/OpenBMI are available. |
| Python 3.14 is very new | All packages installed with `cp314` wheels cleanly. Flag if compatibility issues arise later. |

## `.gitignore`

```
.venv/
cache/
C-/
__pycache__/
*.pyc
.DS_Store
phase0_checkpoint_*.pt
```

## Roadmap

### Done
- [x] Full-pool preprocessing pipeline, verified across all four datasets
- [x] Subject-balanced batch sampler (`sampler.py`)
- [x] Full augmentation pipeline, spec-order-corrected (`augmentations.py`)
- [x] LSP per-dataset channel mixer, spatial Gaussian init — 2/4 datasets confirmed (`lsp.py`)
- [x] EEGNet 9-band spectral encoder, ~17.3K params (`encoder.py`)
- [x] Projection head, 2-layer MLP → 64-dim (`projection_head.py`)
- [x] InfoNCE loss with subject-balanced negative handling (`losses.py`)
- [x] Full training loop, wired end-to-end and smoke-tested on real data (`train_phase0.py`)

### In Progress
- [ ] PhysioNet + OpenBMI downloads/caching (background)
- [ ] Real (non-smoke-test) training run on `bci_iv_2a`
- [ ] LSP spatial init for PhysioNet + OpenBMI, once channel names are pulled

### Immediate Next
- [ ] Gate 1 probe: subject-ID linear probe on frozen Phase 0 embeddings (expect high accuracy)
- [ ] Gate 2 probe: task/class linear probe on frozen Phase 0 embeddings (expect above-chance; near-chance means pretraining learned an identity shortcut — fix augmentations)
- [ ] Multi-dataset batch/LSP-routing extension, so training can pool across all four datasets in one run per the spec's actual design
- [ ] Resolve open spec discrepancies (`W_dataset` shape notation, LSP param budget, 9-band edge definition) with the team

### After Phase 0 + Gates Pass — Build Order (unchanged from original plan)

| Stage | Description |
|---|---|
| **Stage 6** — Stage 3.5 hypothesis gate | Using baseline features, check that cross-subject differences look discrete (large, unstructured jumps) while cross-session differences look continuous (small, correlated drift). Cheap PCA/distance check — if the hypothesis doesn't hold, know before building SSRS and DTKF around it. |
| **Stage 7** — Module A: FiLM conditioning | Subject embedding (64-dim) modulates encoder features; scale and shift per frequency band. Backbone frozen from Phase 0; only FiLM + classifier train. |
| **Stage 8** — Module B: SSRS | Nearest-centroid matching in 64-dim space. Entropy-gated fallback to population average. Ablation: with vs. without confidence gating. |
| **Stage 9** — Module C: DTKF | Diagonal Kalman filter, 64 independent scalar trackers. State and measurement both in the same 64-dim space (by construction). Updates once per window, not per raw sample. Innovation residual spike = artifact rejection. |
| **Stage 10** — Full ablation study | Remove each module individually. Evaluate under LOSO, LOSSO, cross-dataset protocols. SSRS bad-init recovery test. |

## Team

| Person | Responsibility |
|---|---|
| Person A | Preprocessing, LSP, data pipeline, Phase 0 build |
| Person B | SSL backbone, FiLM conditioning |
| Person C | SSRS, DTKF, evaluation and ablations |

---

*This README is updated weekly to track pipeline progress. See [Known Issues & Decisions](#known-issues--decisions) for open questions that need team sync, and [Roadmap](#roadmap) for what's next.*
