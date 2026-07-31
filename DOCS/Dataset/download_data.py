"""
download_data.py
-----------------
Fetches the public EEG datasets used in the "Robust Cross-Session &
Cross-User EEG Decoding" project. Run this after cloning the repo instead
of committing raw data files to git (they're far above GitHub's 25MB
upload limit / 100MB hard git limit).

Datasets covered:
  - bci_iv_2a   : BCI Competition IV, dataset 2a (9 subjects, 2 sessions each)
  - bci_iv_2b   : BCI Competition IV, dataset 2b (9 subjects, 5 sessions each)
  - openbmi     : Lee et al. 2019 OpenBMI / GIST dataset (54 subjects, 2 sessions)
  - physionet   : PhysioNet EEG Motor Movement/Imagery DB (109 subjects)

Usage:
    pip install -r requirements.txt

    python download_data.py --all
    python download_data.py --datasets bci_iv_2a bci_iv_2b
    python download_data.py --datasets physionet --subjects 1 2 3 4 5
    python download_data.py --datasets openbmi          # prints manual steps

Data lands in ./raw/<dataset_name>/ (gitignored). Nothing here is committed
to version control -- only this script and the README describing the data.
"""

import argparse
import sys
from pathlib import Path

DATA_ROOT = Path(__file__).parent / "raw"

# moabb has renamed dataset classes across versions (e.g. BNCI2014001 ->
# BNCI2014_001). We try a few known aliases so the script keeps working
# regardless of which moabb version is installed.
MOABB_ALIASES = {
    "bci_iv_2a": ["BNCI2014_001", "BNCI2014001"],
    "bci_iv_2b": ["BNCI2014_004", "BNCI2014004"],
    "openbmi":   ["Lee2019_MI", "Lee2019MI"],
}


def _get_moabb_class(name):
    from moabb import datasets as moabb_datasets

    for cls_name in MOABB_ALIASES[name]:
        if hasattr(moabb_datasets, cls_name):
            return getattr(moabb_datasets, cls_name), cls_name

    raise AttributeError(
        f"Could not find a moabb dataset class for '{name}' under any of "
        f"{MOABB_ALIASES[name]}. Run `python -c \"import moabb.datasets as d; "
        f"print([x for x in dir(d) if 'BNCI' in x or 'Lee' in x])\"` to see "
        f"what your installed moabb version actually exposes, then update "
        f"MOABB_ALIASES above."
    )


def _download_via_moabb(name, root, subjects=None):
    try:
        import mne
    except ImportError:
        print("Missing dependency. Run: pip install mne moabb", file=sys.stderr)
        sys.exit(1)

    cls, resolved_name = _get_moabb_class(name)
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # moabb/mne cache datasets under the MNE_DATA config path
    mne.set_config("MNE_DATA", str(out_dir))

    ds = cls()
    subj_list = subjects if subjects else ds.subject_list
    print(f"[{name}] Using moabb class '{resolved_name}', "
          f"downloading subjects: {subj_list}")
    ds.get_data(subjects=subj_list)
    print(f"[{name}] Done. Files cached under: {out_dir}")


def download_bci_iv_2a(root, subjects=None):
    _download_via_moabb("bci_iv_2a", root, subjects)


def download_bci_iv_2b(root, subjects=None):
    _download_via_moabb("bci_iv_2b", root, subjects)


def download_physionet(root, subjects=None):
    """PhysioNet EEG Motor Movement/Imagery Database via MNE's built-in
    fetcher. 109 subjects total; each subject is small (~40-50MB across
    14 runs), so pulling all of them is ~3-4GB. Use --subjects to grab a
    handful for quick testing."""
    try:
        from mne.datasets import eegbci
    except ImportError:
        print("Missing dependency. Run: pip install mne", file=sys.stderr)
        sys.exit(1)

    out_dir = root / "physionet"
    out_dir.mkdir(parents=True, exist_ok=True)

    subj_list = subjects if subjects else list(range(1, 110))
    runs = list(range(1, 15))  # 14 runs per subject

    print(f"[physionet] Downloading subjects: {subj_list}")
    for subj in subj_list:
        eegbci.load_data(subj, runs, path=str(out_dir), update_path=False)
    print(f"[physionet] Done. Files cached under: {out_dir}")


def download_openbmi(root, subjects=None):
    """Tries moabb's Lee2019_MI wrapper first (recommended, handles the
    download automatically). Falls back to manual instructions since
    GigaDB doesn't offer a single stable programmatic endpoint."""
    try:
        _download_via_moabb("openbmi", root, subjects)
        return
    except Exception as e:
        print(f"[openbmi] Automatic download via moabb failed: {e}\n")

    print(
        "[openbmi] MANUAL DOWNLOAD REQUIRED\n"
        "  1. Go to: http://gigadb.org/dataset/100542\n"
        "  2. Download the .mat files for the sessions/subjects you need\n"
        "     (54 subjects x 2 sessions, split across MI / ERP / SSVEP paradigms\n"
        "     -- we only need the MI (motor imagery) files).\n"
        f"  3. Place them under: {root / 'openbmi'}\n"
        "  4. Keep the original GigaDB folder/file naming so preprocessing\n"
        "     scripts elsewhere in the repo can find them.\n"
    )


DATASET_FUNCS = {
    "bci_iv_2a": download_bci_iv_2a,
    "bci_iv_2b": download_bci_iv_2b,
    "physionet": download_physionet,
    "openbmi": download_openbmi,
}


def main():
    parser = argparse.ArgumentParser(description="Download EEG datasets for BCI PS1 project.")
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASET_FUNCS.keys()),
        help="Which dataset(s) to download.",
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets.")
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=None,
        help="Restrict to specific subject numbers (e.g. --subjects 1 2 3). "
             "Useful for a quick smoke test before downloading everything.",
    )
    parser.add_argument(
        "--out", type=str, default=str(DATA_ROOT),
        help=f"Output root directory (default: {DATA_ROOT})",
    )
    args = parser.parse_args()

    if not args.all and not args.datasets:
        parser.error("Specify --all or --datasets <name ...>")

    targets = list(DATASET_FUNCS.keys()) if args.all else args.datasets
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    for name in targets:
        print(f"\n=== {name} ===")
        DATASET_FUNCS[name](root, args.subjects)

    print("\nAll requested downloads finished (see notes above for any manual steps).")


if __name__ == "__main__":
    main()
