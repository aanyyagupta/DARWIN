"""
Quick per-dataset sanity check: load ONE subject from each of the four
datasets, run it through preprocessing, and report shape + diagnostics.

This is deliberately cheap (one subject each, not the full pool) -- the
goal is to catch per-dataset bugs (wrong channel count assumptions, wrong
notch frequency, a moabb quirk specific to one dataset) fast, before
committing to the much longer full-pool preprocessing run.

Each dataset is wrapped in its own try/except so one failure (e.g. a slow
or broken download for one dataset) doesn't block checking the others --
you want to know about all problems in one run, not one at a time.
"""

from data_loader import load_dataset_with_retry as load_dataset, TARGET_SFREQ
from preprocessing import preprocess
import numpy as np

# Mains frequency depends on where each dataset was recorded, not on
# anything about the dataset's content -- this is a fact about electrical
# grids, not about EEG. Get this wrong and the notch filter targets the
# wrong frequency entirely, silently leaving real interference untouched.
NOTCH_FREQ_BY_DATASET = {
    "bci_iv_2a": 50.0,   # recorded in Austria
    "bci_iv_2b": 50.0,   # recorded in Austria
    "openbmi": 60.0,     # recorded in South Korea
    "physionet": 60.0,   # recorded in the United States
}


def check_dataset(dataset_key):
    print(f"\n{'='*60}")
    print(f"Checking: {dataset_key}")
    print('='*60)

    try:
        print("  Loading subject 1 ...")
        X, y, meta = load_dataset(dataset_key, subjects=[1])
        print(f"  Loaded: {X.shape[0]} epochs, {X.shape[1]} channels, "
              f"{X.shape[2]} samples/epoch")
        print(f"  Classes: {np.unique(y)}")
        print(f"  Sessions seen: {meta['session'].unique().tolist()}")

    except Exception as e:
        print(f"  LOADING FAILED: {type(e).__name__}: {e}")
        return False

    try:
        notch_freq = NOTCH_FREQ_BY_DATASET[dataset_key]
        print(f"  Preprocessing (notch_base_freq={notch_freq}) ...")
        X_processed, diagnostics = preprocess(
            X, sfreq=TARGET_SFREQ, notch_base_freq=notch_freq
        )

        shapes_match = X_processed.shape == X.shape
        pct_clipped = diagnostics["pct_clipped_per_epoch"]

        print(f"  Output shape: {X_processed.shape} "
              f"(matches input: {shapes_match})")
        print(f"  Clipping: mean {pct_clipped.mean():.3f}%, "
              f"max {pct_clipped.max():.3f}%")
        print(f"  Post-normalization mean/std: "
              f"{X_processed.mean():.4f} / {X_processed.std():.4f} "
              f"(expect near 0 / near 1)")

        if not shapes_match:
            print("  WARNING: shape mismatch after preprocessing -- investigate")
        if pct_clipped.max() > 5.0:
            print("  NOTE: worst epoch has >5% clipped -- worth a look later")

        print(f"  {dataset_key}: OK")
        return True

    except Exception as e:
        print(f"  PREPROCESSING FAILED: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    results = {}
    for key in ["bci_iv_2a", "bci_iv_2b", "openbmi", "physionet"]:
        results[key] = check_dataset(key)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for key, ok in results.items():
        status = "OK" if ok else "FAILED -- see error above"
        print(f"  {key}: {status}")

    if all(results.values()):
        print("\nAll four datasets load and preprocess correctly.")
        print("Safe to proceed to full-pool preprocessing + caching.")
    else:
        failed = [k for k, ok in results.items() if not ok]
        print(f"\n{len(failed)} dataset(s) failed: {failed}")
        print("Fix these before moving to full-pool preprocessing.")