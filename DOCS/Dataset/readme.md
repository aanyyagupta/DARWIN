# DARWIN DATASETS

Raw EEG data is **not** committed to this repo (GitHub blocks web uploads over
25MB, and these datasets run into hundreds of MB to several GB). Instead, run
`download_data.py` to fetch everything locally into `raw/`, which is
gitignored.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Download everything
python download_data.py --all

# Download specific datasets
python download_data.py --datasets bci_iv_2a bci_iv_2b

# Quick smoke test with just a few subjects (fast, for pipeline debugging)
python download_data.py --datasets physionet --subjects 1 2 3
```

## Datasets

| Name | Subjects | Sessions | Task | Source link |
|---|---|---|---|---|
| `bci_iv_2a` | 9 | 2 | 4-class motor imagery | [bbci.de/competition/iv/#dataset2a](https://www.bbci.de/competition/iv/#dataset2a) |
| `bci_iv_2b` | 9 | 5 | 2-class motor imagery | [bbci.de/competition/iv/#dataset2b](https://www.bbci.de/competition/iv/#dataset2b) |
| `openbmi` | 54 | 2 | MI / ERP / SSVEP | [gigadb.org/dataset/100542](http://gigadb.org/dataset/100542) |
| `physionet` | 109 | ~1 (14 runs) | Motor execution + imagery | [physionet.org/content/eegmmidb](https://physionet.org/content/eegmmidb/1.0.0/) |

The `download_data.py` script fetches these automatically via `moabb`/`mne`
(which pull from the underlying BNCI-Horizon / GigaDB / PhysioNet servers
under the hood) — you shouldn't need to click through the links above unless
the script's automatic fetch fails and you have to grab OpenBMI manually.

### Citations

- Brunner, C. et al., "BCI Competition 2008 – Graz data set A," 2008. [bbci.de/competition/iv/#dataset2a](https://www.bbci.de/competition/iv/#dataset2a)
- Leeb, R. et al., "BCI Competition 2008 – Graz data set B," 2008. [bbci.de/competition/iv/#dataset2b](https://www.bbci.de/competition/iv/#dataset2b)
- Lee, M.-H. et al., "EEG dataset and OpenBMI toolbox for three BCI paradigms," *GigaScience*, 2019. [gigadb.org/dataset/100542](http://gigadb.org/dataset/100542)
- Schalk, G. et al., "BCI2000: A General-Purpose Brain-Computer Interface (BCI) System," *IEEE TBME*, 2004. [physionet.org/content/eegmmidb](https://physionet.org/content/eegmmidb/1.0.0/)

## Folder layout after download

```
BCI_PS1_datasets/
├── README.md
├── download_data.py
├── requirements.txt
└── raw/                  <- gitignored, created by the script
    ├── bci_iv_2a/
    ├── bci_iv_2b/
    ├── openbmi/
    └── physionet/
```

## Note on moabb version differences

`moabb` has renamed some dataset classes across versions (e.g.
`BNCI2014001` → `BNCI2014_001`). The script tries known aliases
automatically; if it still fails, run:

```bash
python -c "import moabb.datasets as d; print([x for x in dir(d) if 'BNCI' in x or 'Lee' in x])"
```

and update the `MOABB_ALIASES` dict at the top of `download_data.py` with
whatever class name your installed version actually exposes.
