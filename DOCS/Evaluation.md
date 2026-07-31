# Evaluation Plan & Ablations

## Protocols

- **LOSO (Leave-One-Subject-Out):** train on N−1 subjects, test on the held-out subject. SSRS initializes from the source pool; DTKF tracks drift across the held-out session.
- **LOSSO (Leave-One-Session-Out):** train on all subjects' session 1, test on session 2 of the same subjects. Measures pure session-drift recovery.
- **Cross-Dataset:** train on 3 datasets, test on the 4th. Tests generalization to unseen electrode montages and task paradigms.
- **Calibration Curve:** accuracy vs. amount of unlabeled adaptation data (0s, 5s, 10s, 30s, 60s). Shows where SSRS+DTKF catches up to the supervised upper bound.

## Baselines

- EEGNet, FBCNet, EEG Conformer — supervised, no adaptation
- GAT (graph attention)
- CORAL-DG (domain generalization)
- SCALE-Net, TS²-DER — state-of-the-art cross-subject methods

## Diagnostic analyses

- Subject-ID and trial/class linear probes on Phase-0 embeddings (Stage 4 gate)
- t-SNE/UMAP of the 64-dim space: check source-subject centroid separability (Stage 6a)
- SSRS entropy distribution at test time: report fraction of users triggering fallback
- DTKF innovation residuals: identify artifact spikes vs. genuine drift

## Ablation study

Each component removed in isolation, measured against the full system:

| Ablation | What it tests |
|---|---|
| Remove LSP | Rigid channel intersection vs. learnable mixer |
| Remove SSRS | Population-mean `e_u` vs. similarity-weighted donor selection |
| Remove DTKF | Static subject embedding vs. continuous drift tracking |
| Remove FiLM | Subject-agnostic features vs. conditional modulation |
| Remove subject-balanced sampling | Identity shortcut — does the encoder collapse to subject-only? |
| Full system | Upper bound of the proposed architecture |

## Where results should live

As each protocol/ablation is run, log:
- results table (accuracy ± std per protocol per dataset)
- the diagnostic artifacts above (probe accuracies, entropy histograms, residual plots)

into `notebooks/` or a `results/` directory, and link them from the corresponding week's entry in [`CHANGELOG.md`](../CHANGELOG.md).
