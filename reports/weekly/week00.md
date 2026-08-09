# PROJECT OVERVIEW

**DARWIN (Dynamic Adaptive Representation with Weighted INference)** is a calibration-free EEG decoding framework that treats cross-subject and cross-session variability as two distinct problems. Cross-subject generalization is achieved through **similarity-weighted donor matching (SSRS + FiLM conditioning)**, while cross-session variability is handled using a lightweight **Diagonal Temporal Kalman Filter (DTKF)** that continuously tracks session drift in the latent space.

By modeling subject differences as a matching problem and session drift as a tracking problem within a shared 64-dimensional embedding space, DARWIN enables robust EEG decoding without labels, calibration, or backpropagation during inference, making it suitable for practical real-time Brain-Computer Interface applications.

# RESEARCH  OBJECTIVES

The primary objective of DARWIN is to develop a **calibration-free EEG decoding framework** that maintains robust performance across both unseen subjects and recording sessions. The project aims to:

- Improve cross-subject generalization without requiring subject-specific calibration.
- Handle continuous cross-session drift through adaptive latent state tracking.
- Separate subject variability and session variability into two independently modeled components.
- Enable real-time inference with low computational overhead.
- Provide an interpretable and modular framework that can be extended to different EEG decoding tasks.

# PROPOSED SOLUTION

DARWIN (Dynamic Adaptive Representation with Weighted INference) addresses the two major sources of EEG variability using specialized adaptation mechanisms. **Cross-subject variability** is treated as a similarity matching problem, where the **Subject Similarity Retrieval System (SSRS)** identifies the most relevant source subjects and **FiLM conditioning** dynamically adapts the model to the target user. **Cross-session variability** is modeled as a temporal tracking problem using a **Diagonal Temporal Kalman Filter (DTKF)**, which continuously updates the latent representation to compensate for gradual session drift.

By combining these components within a shared **64-dimensional latent embedding space**, DARWIN performs adaptive EEG decoding without labels, calibration, or backpropagation during inference, making it suitable for practical real-time Brain-Computer Interface applications.

 **Summary of Work Completed**

The following components have been designed, implemented, and verified:

 1. Project repository structure finalized, with a clear separation between code, cache, and raw-download directories.

 2. Reproducible environment setup process documented and tested (virtual environment creation, dependency installation, IDE interpreter configuration).

 3. A MOABB-based data loader implemented and integrated with all four target datasets.

 4. The full preprocessing pipeline, from raw EEG through per-epoch normalization, implemented and verified end to end.

 5.  Correct train/validation/test split ordering established at the epoch level, prior to windowing, to prevent leakage from overlapping windows.

1. A resumable, interruption-safe full-pool preprocessing and caching script implemented and ready to execute.
2.  Preprocessing correctness verified on all four datasets, with consistent, expected output statistics.

 **Environment & Repository Setup**

A standard, reproducible Python environment has been established:

- Python 3.9–3.14, cross-platform (Windows); no GPU required at this stage (GPU will be required later, for Phase 0 model training).
- Virtual environment (.venv) created and activated; dependencies installed via requirements.txt (moabb, mne, scipy, numpy, and transitive dependencies).
- IDE (VS Code) configured to use the project virtual environment as its Python interpreter.
- Version control hygiene established via .gitignore, excluding the virtual environment, preprocessed cache, and raw MOABB download directories from the repository.

 **Repository Deliverables**

The following scripts constitute the completed data pipeline codebase:

| **File** | **Purpose** |
| --- | --- |
| data_loader.py | MOABB-based loader for all four datasets |
| preprocessing.py | Notch filtering, MAD clipping, robust normalization |
| test_pipeline.py | End-to-end sanity check: load and preprocess one subject |
| check_all_datasets.py | Quick sanity check across all four datasets |
| full_pool_preprocess.py | Full preprocessing and disk caching for all subjects |
| requirements.txt | Python dependency list |

 **Datasets Integrated and Verified**

Four public motor-imagery EEG datasets have been integrated through the data loader and preprocessing pipeline. Each dataset's preprocessing has been verified on at least one subject:

| **Dataset** | **Subjects** | **Sessions** | **Channels** | **Classes** | **Notch** | **Status** |
| --- | --- | --- | --- | --- | --- | --- |
| BCI-IV 2a | 9 | 2 | 22 | 4 (left, right, feet, tongue) | 50 Hz | Verified |
| BCI-IV 2b | 9 | 5 | 3 (C3/Cz/C4) | 2 (left, right) | 50 Hz | Verified |
| OpenBMI | 54 | 2 | 62 | 2 (left, right) | 60 Hz | Verified (subject 1) |
| PhysioNet | 109 | ~1 | 64 | varies | 60 Hz | Verified |

 **Preprocessing Pipeline (Completed Stage)**

The following stage of the processing pipeline has been fully implemented and is what is executed and cached by full_pool_preprocess.py:

- Resampling to a common 250 Hz sampling rate across all datasets (anti-aliased, performed once at load time).
- Common Average Reference (CAR), applied per dataset via a logged flag, to remove common-mode noise.
- Notch filtering at the dataset-appropriate mains frequency (50 Hz or 60 Hz) plus its second harmonic.
- Cue-aligned trial / epoch extraction of fixed length, via the MOABB MotorImagery paradigm.
- Median Absolute Deviation (MAD)-based artifact clipping, computed per epoch, with the percentage of samples clipped logged as a diagnostic.
- Robust amplitude normalization, computed per epoch (median / MAD) rather than over the whole recording, with pre-normalization statistics logged as metadata for later drift analysis.

Design decisions for the remaining pipeline stages — Learnable Spatial Projection (LSP), the 9-band FIR filter bank, windowing, and pooling to a 64-dimensional vector — have been specified and documented, but are intentionally applied at training time rather than at caching time, since LSP is a learned component. These stages are not yet implemented in code and are described in Section 8 as planned work.

 **Verification Results**

Each dataset was validated end to end (load → preprocess) on at least one subject. Output tensors were checked for the correct shape and for expected post-normalization statistics (mean near 0, standard deviation near 1):

| **Dataset** | **Subject** | **X Shape** | **Mean Clip** | **Mean** | **Std** |
| --- | --- | --- | --- | --- | --- |
| bci_iv_2a | 1 | (576, 22, 1001) | 0.199% | -0.0046 | 1.0168 |
| bci_iv_2b | 1 | (720, 3, 1126) | 0.288% | 0.0064 | 1.0283 |
| openbmi | 1 | (100, 62, 1000) | 0.737% | -0.0003 | 1.0671 |
| physionet | 1 | (174, 64, 752) | 1.055% | 0.0051 | 1.0910 |

*All four datasets show a mean close to 0 and a standard deviation close to 1 after normalization, confirming consistent and correct preprocessing behavior across heterogeneous data sources with different channel counts, epoch lengths, and recording conditions.*

 **Immediate Next Step**

With preprocessing verified on all four datasets, the immediate next step is to execute the full-pool preprocessing run: processing and caching every subject across all four datasets using full_pool_preprocess.py. This step is estimated to take three to four hours, dominated by OpenBMI downloads and the remaining PhysioNet subjects, and is designed to be safely interruptible and resumable.

 **Planned Work (Not Yet Started)**

The following model-development stages are designed but not yet implemented, and will follow completion of the full-pool caching run:

- Stage 1 — Learnable Spatial Projection (LSP): a per-dataset learned matrix mapping raw channels to a shared 16-dimensional channel space, with Gaussian spatial initialization and L1 regularization.
- Stage 2 — 9-band FIR filter bank applied to LSP output, decomposing the signal into physiologically meaningful frequency bands.
- Stage 3 — EEGNet-style encoder (temporal and depthwise spatial convolutions), approximately 18K parameters.
- Stage 4 — A 2-layer MLP projection head producing the unified 64-dimensional embedding space shared by all downstream modules.
- Stage 5 — Phase 0 InfoNCE contrastive pretraining, with subject-ID and task/class linear probes for validation.
- Stage 6 — A hypothesis-gate check confirming that cross-subject and cross-session differences behave as expected before building SSRS and DTKF.
- Stage 7 — Module A: FiLM conditioning for cross-user adaptation.
- Stage 8 — Module B: SSRS (similarity-weighted donor matching).
- Stage 9 — Module C: DTKF (diagonal Kalman filter for cross-session drift tracking).
- Stage 10 — Full ablation study across LOSO, LOSSO, and cross-dataset protocols.

 **Conclusion**

The data ingestion, preprocessing, and caching layer of the DARWIN Phase 0 pipeline is complete and verified across all four target datasets. This forms a stable, reproducible foundation for the model-development stages that follow. No model components (LSP, filter bank, encoder, projection head, contrastive pretraining, FiLM, SSRS, or DTKF) have been implemented yet; these remain as planned work, beginning with the full-pool preprocessing run.
