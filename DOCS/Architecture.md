# Architecture

Source: Internal Architecture Specification Rev. 3.5 (July 2026).

## Pipeline overview

```
Raw EEG (C channels × T samples)
        │
        ▼
CAR + MAD  (common average reference, outlier clip)
        │
        ▼
Learnable Spatial Projection → C' channels (data-adaptive mixer)
        │
        ▼
EEGNet Encoder (9-band spectral decomposition, ~18K params)
        │
        ▼
Projection Head → 64-dim unified embedding z
        │
        ▼
Phase-0 SSL (InfoNCE, subject-balanced negatives)
        │
        ▼
Subject Embedding e_u (64-dim) ──► SSRS (donor matching) ──┐
        │                                                   │
        ▼                                                   │
      FiLM (scale/shift per band) ◄────────────────────────┘
        │
        ▼
   Classifier (2-layer MLP)
        │
        ▼
   Task Prediction
        │
        ▼
  DTKF (diagonal Kalman)
  innovation = z_t − e_u(t−1)
  updates e_u every sample
```

**Key invariant:** the 64-dim projection-head output is the *only* embedding space. Phase-0 SSL, SSRS centroids, DTKF measurements, and FiLM conditioning all operate in this same space by construction — no separate measurement model, no bridge module, no dimension mismatch.

---

## 4.1 Learnable Spatial Projection (LSP)

**Motivation.** Prior versions used rigid channel intersection (keep only channels common to all 4 datasets), discarding task-relevant information. LSP replaces this with a lightweight, data-adaptive channel mixer.

**Input:** raw EEG with per-dataset channel sets (e.g. 64, 32, 22, 16 channels).

**Architecture:**
- Per-dataset learnable matrix `W_dataset ∈ R^(C_dataset × C_shared)`, where `C_shared = 16` (fixed across all datasets).
- `W` is initialized as a soft channel selector: each row peaks at one physical channel and decays with a Gaussian kernel over spatial neighbours.
- During Phase 0, all dataset-specific `W` matrices are trained jointly. At test time, only the matrix for the incoming dataset is active.

**Parameters:** ~1K total (summed over all datasets).

**Why this works:** the initialization encodes spatial locality (neighbouring electrodes mix), but the network learns to sharpen or blur this mixing per dataset. Unlike rigid intersection, information from non-intersecting channels is preserved through learned projection rather than discarded.

---

## 4.2 Phase-0: Self-Supervised Pretraining

**Input:** pooled unlabeled EEG from all datasets, after LSP.

**Encoder:** EEGNet-style 9-band backbone (~18K params), no FiLM, no subject embedding.

**Projection head:** 2-layer MLP, output fixed at 64 dimensions. This width is non-negotiable — it must match the subject embedding dimension.

**Loss:** InfoNCE with subject-balanced negative sampling — every batch contains ≥2 clips from the same subject as the anchor, alongside cross-subject negatives. Prevents the identity shortcut.

**Diagnostics (go/no-go gate before Phase 1):**
- Subject-ID linear probe on projected embeddings — expect high accuracy (SSRS needs this).
- Trial/class linear probe on the same embeddings — expect above-chance accuracy (task content must be present, not just identity).
- If the trial probe is near-chance while the subject probe is high, the identity shortcut has won — revisit augmentation/sampling before Phase 1.

---

## 4.3 Module B: Source Subject Relevance Scorer (SSRS)

**Training.** For each source subject `i`, compute centroid `c_i = mean(z)` over all their clips in the 64-dim projected space. Learn subject embedding `e_i ∈ R^64`.

**Inference (new user, zero calibration):**
1. Collect ~10s unlabeled EEG from the new user.
2. Pass through the frozen Phase-0 encoder + projection head → `z_raw(t)`.
3. Compute new-user centroid: `c_u = mean(z_raw)` over T seconds.
4. Similarities: `s_i = softmax(−||c_u − c_i||² / τ)`.
5. Entropy check: `H(s) = −Σ s_i log s_i`.
   - **Low H** → composite `ê_u = Σ s_i · e_i` (personalized).
   - **High H** → fallback to `ê_u = mean(e_i)` (population average).

**Key property:** the similarity computation uses the same 64-dim projected space as Phase 0 and DTKF — no "pre-FiLM" ambiguity.

**Failure mode:** if entropy is high for many test subjects, source pool diversity is insufficient. Log fallback frequency as a diagnostic (see [RISKS.md](RISKS.md), F3).

---

## 4.4 Module A: Subject-Conditional Encoder (FiLM)

Warm-started from the Phase-0 checkpoint. Subject embedding `e_u` (64-dim) modulates each of the 9 frequency bands via learned scale `γ` and shift `β`:

```
z'_b = z_b ⊙ γ_b(e_u) + β_b(e_u),   for b = 1..9
```

Trainable in Phase 1: FiLM MLP, subject embedding table, classifier head only. Backbone frozen. Total ~15K parameters.

---

## 4.5 Module C: Drift-Tracking Kalman Filter (DTKF)

**State:** `e_u(t) ∈ R^64`, with per-dimension variance `σ²_k(t)`.

**Measurement:** `z_t` is the projection-head output on the incoming test sample. Since `z_t` and `e_u` live in the identical 64-dim space:

```
innovation_k = z_t,k − e_u(t−1)_k     [no H matrix needed]
```

**Prediction:** `e_u(t|t−1) = e_u(t−1|t−1) + small drift`; `σ²(t|t−1) = σ²(t−1|t−1) + Q`

**Update:** standard Kalman gain `K = σ²_pred / (σ²_pred + R)`; `e_u(t|t) = e_u(t|t−1) + K · innovation`

**Outlier rejection:** if `|innovation_k| > 3√S_k` for any `k`, reject the update (artifact detected).

**Projection clip:** post-update, clip each dimension to `[−3, 3]` to prevent runaway drift.

**Contingency:** if Stage 3.5 finds subject/session entanglement, relax the hard-freeze on the "subject" component of `e_u` — allow DTKF to track combined dynamics.

> Note the preprocessing pipeline updates DTKF **once per window** (not per raw sample) with `Q = 0.001` per-window drift assumption — see [PREPROCESSING.md](PREPROCESSING.md). Confirm which cadence (per-sample vs. per-window) the current implementation targets before Phase 1, and keep this doc and the pipeline doc in sync.

---

## 4.6 Classifier

2-layer MLP: `9 → 32 → num_classes`, ~500 parameters. Trained in Phase 1; frozen at test time. All adaptation happens upstream — SSRS + DTKF update `e_u`, which modulates features via FiLM.

---

## Training losses (Phase 1)

```
L_total = L_task + λ_sparse · L_sparse + λ_smooth · L_smooth
```

- `L_task = CrossEntropy(classifier(FiLM(z, e_u)), y)`
- `L_sparse = −Σ s_i log s_i` (peaked SSRS weights)
- `L_smooth = ||e_u(t) − e_u(t−1)||²` (temporal stability)

### Fixed hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| λ_sparse (sparsity) | 0.01 | Gentle push toward peaked donor weights |
| λ_smooth (smoothness) | 0.001 | Prevents embedding jitter, allows drift |
| τ (SSRS temperature) | 0.5 | Mid-range: not overconfident, not uniform |
| Q (process noise) | 0.001/dim | Slow drift assumption; DTKF corrects if wrong |
| R (measurement noise) | 0.1/dim | Trust prediction over noisy single-sample |
| Clip range | [−3, 3] | Prevents runaway; covers ~99.7% of training distribution |

Set by reasoning, not grid search — see spec §5.2, Table 1.

---

## Test-time inference (deployment)

1. **Initialization:** run SSRS on ~10s of unlabeled data → `ê_u(0)`.
2. **Per-sample loop:**
   1. Forward `x_t` through the frozen encoder + projection head → `z_t`.
   2. DTKF: `innovation = z_t − e_u(t−1)`; update `e_u(t)`.
   3. FiLM: modulate features with `e_u(t)`.
   4. Frozen classifier → prediction.

No labels, no backprop, no calibration.

