# Reproduce the DermWorkup-Records audit

The public pixels are not required. The OPEN tile tables and the `rda` package are.

## Data

1. Download the UQ eSpace OPEN package (doi: `10.48610/a13deaf`).
   **This step needs a human.** The eSpace site sits behind an AWS WAF bot challenge:
   `curl` and other scripted clients get HTTP 403 on every path, and the record page
   does not finish loading in an automated browser. Open
   `https://espace.library.uq.edu.au/view/UQ:a13deaf` in an ordinary browser, pass the
   check, and download from there. Nothing else in this repository needs interactive
   access.
2. The OPEN zip is roughly 62 GiB, almost all of it pixel images. You do not need it.
   Take the tile and dermoscopy tables out of the zip without downloading the images,
   or fetch those members directly, and concatenate the general-population and
   high-risk tile sheets to `data/csv/tiles.csv`.
3. Do not expect descriptor Table 4 in the zip; HOPS `Study_arm` exclusion cannot be
   applied from the download.

## Environment

- Python 3.11+
- `pandas>=2.0`
- `torch>=2.0` (tabular scorer; CPU is enough)
- Seed `0` for the patient split and the tabular scorer

```bash
python3 -m pip install -r requirements.txt
```

Remote reference run used PyTorch 2.5.1.

## Commands

From the directory that contains the `rda` package:

```bash
python3 -m unittest discover -s tests -q

python3 -m rda audit \
  --tiles data/csv/tiles.csv \
  --out results/trajectory_audit.json \
  --trajectories results/transitions.jsonl

python3 -m rda audit-ranking \
  --train results/train.jsonl \
  --val results/val.jsonl \
  --test results/test.jsonl \
  --out results/audit_closeout.json

python3 -m rda.revision_audit
python3 -m rda.round2_review_audit
```

Figures:

```bash
python3 manuscript/figures/make_paper_figures.py
```

## Outputs used in the paper

| File | Role |
|---|---|
| `05_验证记录/llm_01/audit_closeout.json` | Main ranking, label sources, visit groups |
| `05_验证记录/llm_01/revision_audit.json` | Patient-level bootstrap CIs, endpoint overwrite rows |
| `05_验证记录/llm_01/round2_review_audit.json` | Chain-length histogram, correspondence fields, subset AP |

Patient-level csv/jsonl stay on the analysis machine and are not copied into git.
