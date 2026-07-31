# Preprocessing Pipeline

This describes the data pipeline from raw EEG to the windowed, pooled embeddings consumed downstream. Every stage runs once per dataset and is logged for reproducibility.

## Stage-by-stage

1. **Raw EEG** — `C_in × T_native`, tagged with `[dataset_id, subject_id, session_id]`.
2. **Resample → 250 Hz** — anti-aliased; saved to disk once (not recomputed per run).
3. **CAR (Common Average Reference)** — per-dataset flag, logged.
4. **Notch filter** — 50/60 Hz + 2nd harmonic, per-dataset, bandstop ±2 Hz.
5. **Trial/epoch extraction** — cue-aligned, fixed length `L`.
6. **MAD artifact clipping** — per-epoch, robust; logs % clipped per epoch.
7. **Robust amplitude normalization** — per-epoch median/MAD.
   - **Logged as metadata:** pre-normalization median and MAD per epoch.
8. **Learnable Spatial Projection (LSP)** — `C_in → C_out`, Gaussian init, L1 regularization; ablated against a rigid-intersection baseline (see [ARCHITECTURE.md §4.1](ARCHITECTURE.md#41-learnable-spatial-projection-lsp)).
9. **9-band FIR filter bank** — edges (Hz): `[1–4, 4–8, 8–13, 13–20, 20–30, 30–40, 40–50, 50–65, 65–80]`, capped to the minimum dataset Nyquist frequency.

---

### ═══ Split boundary ═══

`[subject_id, session_id]` is assigned to train/val/test **per protocol, before windowing**. **Windows never straddle splits.**

This ordering matters: doing the split *before* windowing (rather than after) is what guarantees no data leakage between splits via overlapping windows.

---

10. **Windowing** — length `T`, stride `T/2`, per-sample output kept for pooling.
11. **Pool window → single 64-dim vector** — mean across time.
    - **DTKF updates once per window** (not per raw sample).
    - **Q = 0.001** per-window drift assumption.
12. **Output:** `(num_windows, 64)`, tagged with `[dataset_id, subject_id, session_id, split, window_idx]`.

## Diagram

```
Raw EEG (C_in × T_native) [dataset_id, subject_id, session_id]
        │
        ▼
Resample → 250 Hz (anti-aliased, saved once)
        │
        ▼
CAR (per-dataset flag, logged)
        │
        ▼
Notch 50/60 Hz + 2nd harmonic (bandstop ±2 Hz)
        │
        ▼
Trial/epoch extraction (cue-aligned, fixed length L)
        │
        ▼
MAD artifact clipping (per-epoch, logs % clipped)
        │
        ▼
Robust amplitude norm (per-epoch median/MAD)
    → logs pre-normalization median, MAD per epoch
        │
        ▼
LSP (C_in → C_out, Gaussian init, L1 reg; ablated vs. rigid)
        │
        ▼
9-band FIR filter bank (1–80 Hz, 9 bands, capped to min Nyquist)
        │
════════════════ SPLIT BOUNDARY ════════════════
   Assign [subject_id, session_id] → train/val/test
   BEFORE windowing. Windows never straddle splits.
═════════════════════════════════════════════════
        │
        ▼
Windowing (length T, stride T/2, per-sample output)
        │
        ▼
Pool window → single 64-dim vector (mean across time)
    → DTKF updates once per window, Q = 0.001
        │
        ▼
(num_windows, 64) [dataset_id, subject_id, session_id, split, window_idx]
```

## Notes for implementers

- **Windowing cadence vs. DTKF spec:** the architecture spec (§4.5) describes DTKF updating "every sample," while this pipeline updates DTKF **once per pooled window**. Confirm with the team which cadence the current implementation targets, and reconcile the wording between this file and [`ARCHITECTURE.md`](ARCHITECTURE.md) once decided — flag any discrepancy in the weekly changelog rather than silently picking one.
- **Reproducibility:** every stage that involves a per-dataset flag, threshold, or clipping percentage should be logged to run metadata, not just applied silently — this is required for the diagnostics in [`RISKS.md`](RISKS.md) and [`EVALUATION.md`](EVALUATION.md).
- **Filter bank edges are capped per dataset** to that dataset's Nyquist frequency (i.e. datasets with lower native sampling rates may have fewer than 9 usable bands) — this should be logged per dataset, not silently truncated.
