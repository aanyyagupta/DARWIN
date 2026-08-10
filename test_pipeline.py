"""
End-to-end check: load one dataset via MOABB, run it through DARWIN's
preprocessing pipeline, and print diagnostics at each stage.

This is the first point where you're looking at the *combined* effect of
loading + preprocessing on real data, not just confirming each piece works
in isolation.
"""

from data_loader import load_dataset, TARGET_SFREQ
from preprocessing import preprocess
import numpy as np

if __name__ == "__main__":
    print("Loading BCI-IV 2a, subject 1 ...")
    X, y, meta = load_dataset("bci_iv_2a", subjects=[1])
    print(f"  Loaded: {X.shape[0]} epochs, {X.shape[1]} channels, "
          f"{X.shape[2]} samples/epoch")
    print(f"  Classes: {np.unique(y)}")

    print("\nRunning preprocessing pipeline "
          "(notch filter -> MAD clip -> robust normalize) ...")
    # notch_base_freq=50.0 because BCI-IV 2a was recorded in Austria (European mains)
    X_processed, diagnostics = preprocess(
        X, sfreq=TARGET_SFREQ, notch_base_freq=50.0
    )

    print(f"  Output shape: {X_processed.shape} "
          f"(should match input shape -- preprocessing doesn't change dimensions)")

    pct_clipped = diagnostics["pct_clipped_per_epoch"]
    print(f"\n  Artifact clipping: mean {pct_clipped.mean():.3f}% of samples "
          f"clipped per epoch, max {pct_clipped.max():.3f}% in worst epoch")
    if pct_clipped.max() > 5.0:
        print("  NOTE: some epoch has >5% clipped -- worth a visual check "
              "later, may indicate a genuinely noisy trial")

    print(f"\n  Sanity check on normalization -- processed data should now "
          f"be roughly centered near 0 with unit-ish scale:")
    print(f"    mean: {X_processed.mean():.4f}  (expect close to 0)")
    print(f"    std:  {X_processed.std():.4f}   (expect roughly near 1)")

    print("\nPipeline ran end to end successfully.")
