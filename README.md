# DARWIN
### Dual Adaptation Framework for Robust Cross-Session and Cross-Subject EEG Decoding

> A calibration-free deep learning framework for robust EEG decoding that jointly models inter-subject variability and intra-session drift through subject-adaptive conditioning and continuous latent-state tracking.

---

## Overview

Brain-Computer Interface (BCI) systems based on electroencephalography (EEG) often exhibit a significant drop in performance when models trained on one subject or recording session are deployed on unseen users or future sessions. These distribution shifts arise from two fundamentally different sources of variability:

- **Cross-subject variability** caused by anatomical and physiological differences between individuals.
- **Cross-session variability** caused by temporal changes such as fatigue, electrode impedance, cognitive state, and attention drift.

Most existing approaches attempt to remove these variations through adversarial learning, domain adaptation, or meta-learning. While effective under specific settings, these methods frequently require calibration, introduce optimization instability, or increase computational complexity.

DARWIN approaches this problem from a different perspective.

Instead of forcing invariance, DARWIN explicitly models both sources of variability using specialized adaptation mechanisms operating within a unified latent representation. Subject-specific characteristics are estimated through similarity-based donor selection, while session-specific changes are continuously tracked using a lightweight Kalman filtering strategy. The resulting framework enables calibration-free inference without requiring gradient updates during deployment.

---

# Key Contributions

- **Unified Latent Representation**
  - A shared 64-dimensional embedding space is used consistently across representation learning, subject adaptation, drift estimation, and classification.

- **Learnable Spatial Projection**
  - Dataset-specific EEG channel layouts are harmonized using trainable spatial projections rather than rigid channel intersection, preserving more physiological information.

- **Subject Similarity-Based Adaptation (SSRS)**
  - New users are initialized through similarity-weighted donor embeddings with confidence-aware fallback mechanisms.

- **Continuous Drift Tracking (DTKF)**
  - A lightweight diagonal Kalman Filter continuously updates subject representations throughout inference without requiring labeled data.

- **Feature-wise Conditional Modulation**
  - FiLM conditioning dynamically adapts learned EEG representations according to the estimated subject state.

- **Calibration-Free Deployment**
  - No backpropagation, parameter updates, or supervised calibration are required during inference.

---

# Framework Overview

The DARWIN pipeline consists of six primary stages:

```

Raw EEG
│
├── Signal Preprocessing
│
├── Learnable Spatial Projection
│
├── EEG Feature Encoder
│
├── Self-Supervised Representation Learning
│
├── Subject & Session Adaptation
│ │
│ ├── SSRS
│ └── DTKF
│
└── FiLM Conditioned Classifier
↓
Task Prediction

```

The framework maintains a single latent representation throughout the entire inference pipeline, eliminating representation mismatch between adaptation modules and ensuring computational efficiency.

---

# Core Components

## Learnable Spatial Projection (LSP)

A lightweight dataset-aware projection layer aligns heterogeneous EEG montages into a common spatial representation while preserving information from non-overlapping electrodes.

---

## Self-Supervised Representation Learning

DARWIN first learns task-aware EEG embeddings using contrastive learning with subject-balanced negative sampling. This stage produces representations that capture meaningful neural activity without relying on labeled calibration data.

---

## Source Subject Relevance Scorer (SSRS)

Given a small amount of unlabeled EEG from a previously unseen user, SSRS computes similarity scores against source subjects and estimates an initial personalized latent representation.

When similarity confidence is low, the model gracefully falls back to a population-level embedding instead of relying on unreliable donor selection.

---

## Drift Tracking Kalman Filter (DTKF)

Rather than assuming the subject representation remains constant throughout an experiment, DARWIN continuously updates the latent subject embedding using a lightweight diagonal Kalman Filter.

This enables the framework to adapt naturally to gradual physiological changes while maintaining computational efficiency suitable for real-time BCI systems.

---

## FiLM-Based Subject Conditioning

Feature-wise Linear Modulation (FiLM) conditions the classifier using the current latent subject state, allowing predictions to remain personalized without modifying the backbone encoder.

---

# Methodology

DARWIN follows a two-stage learning pipeline.

## Phase I — Self-Supervised Representation Learning

- Learn dataset-independent EEG representations
- Contrastive learning objective
- Subject-balanced negative sampling
- Frozen encoder after convergence

---

## Phase II — Supervised Fine-Tuning

- Initialize subject representations
- Train FiLM conditioning layers
- Train lightweight classifier
- Keep encoder weights fixed

---

## Test-Time Adaptation

Inference proceeds without gradient updates.

1. Estimate initial subject embedding via SSRS.
2. Continuously refine the embedding using DTKF.
3. Condition extracted EEG features through FiLM.
4. Produce task predictions.

No labels, calibration, or optimization steps are required during deployment.

---

# Repository Structure

```

DARWIN/

├── README.md
├── LICENSE
├── CHANGELOG.md
├── requirements.txt
│
├── configs/
│
├── docs/
│ ├── architecture.md
│ ├── methodology.md
│ ├── preprocessing.md
│ ├── training.md
│ ├── evaluation.md
│ ├── datasets.md
│ ├── usage.md
│ └── references.md
│
├── assets/
│ ├── architecture.png
│ ├── pipeline.png
│ └── figures/
│
├── src/
│ ├── data/
│ ├── models/
│ ├── losses/
│ ├── trainers/
│ ├── evaluation/
│ └── utils/
│
├── notebooks/
│
└── experiments/

```

---

# Datasets

DARWIN is designed to support heterogeneous EEG datasets with varying channel layouts and recording protocols.

The preprocessing pipeline accommodates multiple electrode configurations through learnable spatial harmonization rather than fixed channel intersection.

Detailed dataset information is available in:

```

docs/datasets.md

```

---

# Experimental Evaluation

The framework is evaluated under multiple generalization settings, including:

- Leave-One-Subject-Out (LOSO)
- Leave-One-Session-Out (LOSSO)
- Cross-Dataset Evaluation
- Calibration-Free Adaptation
- Component Ablation Studies

Performance is compared against both classical EEG decoders and modern domain adaptation methods.

---

# Design Principles

DARWIN is built around four guiding principles:

- Explicit modeling instead of enforced invariance
- Lightweight adaptation suitable for real-time deployment
- Calibration-free inference
- Unified latent representations across all adaptation modules

---

# Documentation

Comprehensive technical documentation is provided in the `docs/` directory.

| Document | Description |
|----------|-------------|
| architecture.md | Complete architectural design |
| methodology.md | Training methodology |
| preprocessing.md | EEG preprocessing pipeline |
| datasets.md | Dataset preparation |
| training.md | Training procedures |
| evaluation.md | Experimental protocols |
| usage.md | Running experiments |
| references.md | Bibliography |

---

# Future Extensions

Potential future directions include:

- Graph-based subject similarity propagation
- Full covariance Kalman filtering
- Cross-paradigm EEG adaptation
- Multi-modal physiological integration
- Online continual learning

---

# Citation

If you use DARWIN in your research, please cite the accompanying publication (to be released).

```bibtex
@article{darwin2026,
  title={DARWIN: Dual Adaptation Framework for Robust Cross-Session and Cross-Subject EEG Decoding},
  author={Authors},
  year={2026}
}
```

---

# License

This project is released under the MIT License.

See the `LICENSE` file for details.

---

## Acknowledgements

DARWIN is inspired by recent advances in self-supervised representation learning, domain generalization, adaptive filtering, and calibration-free Brain-Computer Interface research.
