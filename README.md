# DermWorkup-Offline — analysis code and audit results

Code and aggregate audit outputs for the reconstruction and ranking audit of sequential
pigmented-lesion records built from the public longitudinal 3D total-body photography
release of Ghahari et al.

The paper is **not** published yet; it is under review. The repository is deposited so
that the reconstruction and every reported number can be checked independently.

---

## Source data

Ghahari, N., Caffery, L., Betz-Stablein, B., Mothershaw, A., Jayasinghe, D., Primiero, C.,
Chandra, S. S., Torrano, J., Soyer, H. P., Janda, M. "A longitudinal dataset of tile and
corresponding dermoscopic images with metadata for identifying skin cancers."
*Scientific Data* **12**, 1602 (2025). doi: [10.1038/s41597-025-05880-2](https://doi.org/10.1038/s41597-025-05880-2)

Data record: UQ eSpace, doi: [10.48610/a13deaf](https://doi.org/10.48610/a13deaf)

**No pixel images are needed.** The analysis reads only the tile metadata tables. The
OPEN package is large because it contains ~62 GiB of images; the tile and dermoscopy
tables can be taken out of it without downloading the images.

### Licensing — please read before redistributing anything derived from this dataset

The dataset is **not** released under a standard Creative Commons licence. Its DataCite
rights statement is:

> UQ General Usage Terms and Conditions for data publishing in eSpace

and a second rights entry points at a UQ "licence for reuse of data" agreement. Both
reference `guides.library.uq.edu.au` pages that **currently return HTTP 404**, so the
terms under which a derived record table may be redistributed could not be verified.

The *article* is CC BY-NC-ND 4.0. That licence covers the paper, **not** the dataset.

Consequently this repository deliberately contains **no patient-level or record-level
data**. See "What is not in this repository" below.

---

## What is in this repository

| Path | Contents |
|---|---|
| `rda/` | The analysis package: chain reconstruction, event labelling, feature construction, lookup and tabular baselines, metrics, uncertainty and split-stability audits |
| `configs/` | Protocol and experiment configuration: split fractions, seeds, model and training hyperparameters |
| `results/` | Aggregate audit outputs — counts, event rates, average precision, AUROC, and bootstrap intervals. Every number quoted in the paper is reproduced in one of these files |
| `tests/` | Unit tests for trajectory construction and the training step |
| `REPRODUCE.md` | Exact commands, environment, and expected outputs |

## What is not in this repository, and why

- **No raw or pixel images.** Never downloaded; the analysis does not use them.
- **No `tiles.csv`** (the concatenated tile table) and **no per-record JSONL**. These
  contain participant identifiers, lesion identifiers and dermoscopic filenames taken
  from the source dataset, so their redistribution is governed by UQ's terms, which we
  could not verify (see above).
- **No reconstructed lesion–visit record table**, for the same reason.

The record table does not have to be deposited for the work to be checkable. The
reconstruction is deterministic — the patient split is a SHA-256 hash of the split seed
and the participant string, and the model seed is fixed — so `REPRODUCE.md` regenerates
it exactly from the public source. That is a stronger guarantee than a deposited dump,
because it also pins the code that produced it.

If you need the record table, generate it with the commands below, or request permission
from the dataset depositors.

---

## Reproducing the audit

See `REPRODUCE.md` for the full procedure. In short:

```bash
python3 -m pip install -r requirements.txt

python3 -m rda audit \
  --tiles data/csv/tiles.csv \
  --out results/trajectory_audit.json \
  --trajectories results/transitions.jsonl

python3 -m rda audit-ranking \
  --train results/train.jsonl --val results/val.jsonl --test results/test.jsonl \
  --out results/audit_closeout.json

python3 -m rda.revision_audit
python3 -m rda.round2_review_audit
python3 -m rda.r3_audit
```

The `audit` step also writes `train.jsonl`, `val.jsonl` and `test.jsonl` from the patient
split. The ranking and audit steps refuse to run unless `RDA_REMOTE=1` is set, which is a
guard against mistaking a smoke run for the real analysis.

### Verified reproduction

The reconstruction step was re-run from `tiles.csv` on a clean checkout and produced a
`trajectory_audit.json` **identical in every field** to the deposited one, including

| Field | Value |
|---|---|
| Lesion chains | 66,152 components of length ≥ 2 (958,494 components total) |
| Lesion–visit records | 174,160 |
| Participants | 352 |
| Pathology-linked endpoint proxies | 169 (49 visit-aligned, 120 lesion-level) |
| Patient split | 248 / 35 / 69 (train / validation / test), gate: pass |

## The main finding this code supports

The AcquireDerm label — the adjacent-record onset of a dermoscopic identifier — is
dominated by records where the lesion already carried an identifier when it entered the
table. Restricting the positive set to one label source at a time shows that all ranking
skill sits in that stratum: on the 933 chain-first positives the tabular model reaches
AUROC 0.83, but on the 68 identifiers first acquired during follow-up it reaches 0.46 and on
the 31 re-onsets 0.46, i.e. no better than chance. See `results/r3_audit.json`.

---

## Licence

The code in `rda/`, `configs/` and `tests/` is released under the MIT licence
(see `LICENSE`). The aggregate results in `results/` are derived from a dataset that is
**not** under an open licence, so they are provided for verification of the accompanying
paper and are not offered as a redistributable dataset.

## Citation

See `CITATION.cff`. If you use this code, please cite the paper above and the dataset.

## Contact

Corresponding author for the paper: Xianyu Zeng.
