# DARWIN — Dual Adaptation framework for Robust cross-session and cross-subject WIN-decoding

A calibration-free framework for cross-session and cross-subject EEG decoding. DARWIN combines **subject-adaptive conditioning** via similarity-weighted donor selection (SSRS) with **continuous session-drift tracking** via a lightweight diagonal Kalman filter (DTKF), operating in a single unified 64-dimensional embedding space.

> Status: Final Review (Spec Rev. 3.5) — implementation in progress. This repo tracks weekly development against the [8-week plan](docs/ROADMAP.md).

---

## Why DARWIN

EEG decoders trained on one subject or session degrade sharply on new users or new recordings, because two different kinds of variation get conflated:

- **Cross-subject variation** — discrete, fixed at session start, unstructured (skull thickness, cortical folding, electrode placement).
- **Cross-session variation** — continuous, evolving, structured (fatigue, electrode impedance, attention drift).

Prior approaches (adversarial domain adaptation, full disentanglement, meta-learning) either fight instability at small scale, over-parameterize an unvalidated assumption, or require test-time calibration. DARWIN's core hypothesis is that these two variation sources are **separable**, and that the right tool for each is different: **selection/matching** for subjects, **tracking/filtering** for drift. This is validated empirically (not assumed) at Stage 3.5 before any DARWIN-specific module is built.

Read the full rationale in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## How it works, in one paragraph

Raw EEG passes through a per-dataset **Learnable Spatial Projection** into a shared channel space, then an EEGNet-style 9-band encoder and a fixed-width projection head produce a **64-dim embedding** — the single space in which everything downstream lives. This embedding is pretrained self-supervised (InfoNCE, subject-balanced negatives) so it carries task content without collapsing onto subject identity. At test time, ~10s of unlabeled data gives an initial subject embedding via **SSRS** (donor similarity, with confidence-gated fallback to the population average), and a **diagonal Kalman filter** then updates that embedding continuously — sample by sample — as the session progresses, with no undefined measurement model because state and measurement share the same 64-dim space by construction. A **FiLM** layer conditions a frozen classifier on the current subject+session state. No labels, no backprop, and no calibration step at test time.

## Repo structure

```
.
├── README.md                  ← you are here
├── CHANGELOG.md                ← weekly development log (append-only)
├── docs/
│   ├── ARCHITECTURE.md         ← module-by-module spec (LSP, SSRS, FiLM, DTKF, classifier)
│   ├── PREPROCESSING.md        ← the raw-EEG → windowed-embedding data pipeline
│   ├── ROADMAP.md              ← 8-week milestone plan, tracked as a checklist
│   ├── EVALUATION.md           ← protocols, baselines, ablations
│   └── RISKS.md                ← assumptions, failure criteria, contingencies
├── src/                        ← (to be populated — see ROADMAP Week 1)
├── data/                       ← (gitignored — pointers/configs only, no raw EEG committed)
└── notebooks/                  ← exploratory + diagnostic notebooks (probes, t-SNE, entropy plots)
```

> `src/`, `data/`, and `notebooks/` are placeholders for this documentation drop — create them as Week 1 deliverables land.

## Key design decisions (quick reference)

| Decision | Why |
|---|---|
| Unified 64-dim space for SSL, SSRS, DTKF, and FiLM | Eliminates the need for a learned/undefined Kalman measurement model — `innovation = z_t − e_u(t−1)` directly, no `H` matrix. |
| Learnable Spatial Projection instead of rigid channel intersection | Preserves information from non-overlapping channels across 4 heterogeneous montages (64/32/22/16 ch) instead of discarding it. |
| Confidence-gated SSRS fallback | Converts a single point of failure (bad donor match) into graceful degradation to population-average conditioning. |
| Subject-balanced negative sampling in Phase-0 SSL | Stops the encoder from using subject identity as a shortcut instead of learning task-relevant structure. |
| Stage 3.5 empirical gate | Validates the core "subjects vs. sessions are separable" hypothesis on baseline features *before* building any DARWIN-specific module. |
| Fixed hyperparameters (no grid search) | λ_sparse=0.01, λ_smooth=0.001, τ=0.5, Q=0.001/dim, R=0.1/dim, clip=[−3,3] — set by reasoning; see [Table 1](docs/ARCHITECTURE.md#fixed-hyperparameters). |

## Evaluation at a glance

- **LOSO** (Leave-One-Subject-Out), **LOSSO** (Leave-One-Session-Out), **Cross-Dataset**, and a **Calibration Curve** (accuracy vs. 0–60s of unlabeled adaptation data).
- Baselines: EEGNet, FBCNet, EEG Conformer, GAT, CORAL-DG, SCALE-Net, TS²-DER.
- Full ablation table (remove LSP / SSRS / DTKF / FiLM / subject-balanced sampling) in [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Data pipeline

Raw EEG is resampled, referenced, notch-filtered, epoched, artifact-clipped, normalized, spatially projected (LSP), and band-filtered *before* the train/val/test split boundary; windowing and pooling happen *after* the split so no window ever straddles two splits. Full detail, including the exact filter-bank edges and per-window DTKF update cadence, is in [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md).

## Model budget

Total model size is designed to stay **under 50K parameters**, with **no test-time backpropagation** — LSP (~1K), EEGNet-style encoder (~18K), FiLM MLP + subject embedding table + classifier (~15.5K, Phase 1 only).

## Team & scope

Scoped for a **2–3 person team**, no fixed deadline. Every planned addition is a diagnostic, a reordering, or a small parameter change — no new model components beyond the fixed-width projection head and the lightweight spatial mixer.

## Weekly development log

Progress against the roadmap is tracked in [`CHANGELOG.md`](CHANGELOG.md). Each week's entry links to the relevant diagnostics (probe accuracies, entropy distributions, innovation residuals, ablation deltas) as they're produced.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch conventions, how to log a diagnostic result, and how to propose a deviation from the fixed hyperparameters in Table 1.

---

*This README summarizes Internal Architecture Specification Rev. 3.5 (July 2026, Status: Final Review). For the full spec, see the original document.*
