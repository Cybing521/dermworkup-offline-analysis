"""Revision analyses for reviewer comments. Run on llm_01 with real jsonl."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .audit_ranking import (
    _apply_lookup,
    _fit_lookup,
    _mask_not_first_record,
    _mask_prev0,
    attach_history,
)
from .benchmark import onset_heuristics, onset_scores, train_onset_scorer
from .dataset import load_jsonl
from .features import LOCATIONS, _loc_index
from .metrics import auroc, average_precision_grouped, ranking_metrics


def _ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)

    def pct(p):
        i = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return xs[i]

    return {"mean": sum(xs) / n, "lo": pct(2.5), "hi": pct(97.5), "n": n}


def _y_current(s):
    return int(s["action"] == 1)


def _y_adjacent(s):
    return int(bool(s.get("adjacent_onset")))


def _y_cancer_endpoint(s):
    if s["action"] == 2 and int(s.get("cancer") or 0):
        return 0
    return int(bool(s.get("adjacent_onset")))


def _block(steps, scores, mask, y_fn):
    y = [y_fn(s) for s, keep in zip(steps, mask) if keep]
    sc = [scores[i] for i, keep in enumerate(mask) if keep]
    return ranking_metrics(y, sc)


def _patient_bootstrap(steps, score_map, mask, y_fn, n_boot=500, seed=0):
    idx = [i for i, keep in enumerate(mask) if keep]
    by_p = defaultdict(list)
    for i in idx:
        by_p[steps[i]["participant"]].append(i)
    patients = list(by_p)
    rng = random.Random(seed)
    aps, aucs = [], []
    for _ in range(n_boot):
        draw = [rng.choice(patients) for _ in patients]
        y, sc = [], []
        for p in draw:
            for i in by_p[p]:
                y.append(y_fn(steps[i]))
                sc.append(score_map[i])
        if sum(y) == 0 or sum(y) == len(y):
            continue
        aps.append(average_precision_grouped(y, sc))
        aucs.append(auroc(y, sc))
    return {"ap": _ci(aps), "auroc": _ci(aucs)}


def run(root: Path, n_boot: int = 500, onset_steps: int = 4000):
    train = load_jsonl(root / "results" / "train.jsonl")
    test = load_jsonl(root / "results" / "test.jsonl")
    attach_history(train)
    attach_history(test)
    prev0 = _mask_prev0(test)
    not_first = _mask_not_first_record(test)
    prev0_nf = [a and b for a, b in zip(prev0, not_first)]
    train_prev0 = _mask_prev0(train)
    heur = onset_heuristics(test)
    visit_sc = heur["neg_visit"]
    fitted = _fit_lookup(train, train_prev0)
    look_all, src_all = _apply_lookup(test, fitted)
    look_prev0, src_prev0 = _apply_lookup([s for s, k in zip(test, prev0) if k], fitted)

    chain_len = {}
    for s in test + train:
        chain_len[s["lesion_id"]] = int(s.get("chain_len") or 0)
    len_hist = Counter(chain_len.values())

    sites = Counter()
    for s, keep in zip(test, prev0):
        if keep:
            sites[LOCATIONS[_loc_index(s.get("location"))]] += 1

    overwrite = []
    for s in test:
        if s.get("adjacent_onset") and s["action"] != 1:
            overwrite.append(
                {
                    "cancer": int(s.get("cancer") or 0),
                    "melanoma": int(s.get("melanoma") or 0),
                    "alignment": s.get("alignment"),
                    "diagnosis": str(s.get("diagnosis") or ""),
                    "action": s.get("action_name"),
                    "visit": s.get("visit"),
                    "cohort": s.get("cohort"),
                }
            )

    ep = [s for s in test if s["action"] == 2]
    ep_dx = Counter(str(s.get("diagnosis") or "").strip().lower() or "(empty)" for s in ep)
    ep_cancer = sum(int(s.get("cancer") or 0) for s in ep)

    next_id = sum(int(bool(str(s.get("next_tile_id") or "").strip())) for s in test)
    within = {}
    by_vis = defaultdict(list)
    for i, s in enumerate(test):
        if prev0[i]:
            by_vis[int(s.get("visit") or 0)].append(i)
    for vis, idxs in sorted(by_vis.items()):
        n = len(idxs)
        pos_idx = [k for k, i in enumerate(idxs) if _y_current(test[i])]
        if not pos_idx:
            continue
        within[str(vis)] = {
            "n": n,
            "n_pos": len(pos_idx),
            "mean_pos_frac": sum(k / max(n - 1, 1) for k in pos_idx) / len(pos_idx),
            "frac_pos_in_first_half": sum(int(k < n / 2) for k in pos_idx) / len(pos_idx),
        }

    labels = {
        "current_action": _y_current,
        "adjacent_no_endpoint": _y_adjacent,
        "cancer_coded_endpoint_only": _y_cancer_endpoint,
    }
    label_metrics = {}
    for name, fn in labels.items():
        label_metrics[name] = {
            "visit": _block(test, visit_sc, prev0, fn),
            "lookup": _block(test, look_all, prev0, fn),
        }

    print("training tabular model...", flush=True)
    net = train_onset_scorer(train, hidden=128, steps=onset_steps, seed=0)
    mlp = onset_scores(net, test)
    label_metrics["current_action"]["tabular"] = _block(test, mlp, prev0, _y_current)
    label_metrics["adjacent_no_endpoint"]["tabular"] = _block(test, mlp, prev0, _y_adjacent)
    label_metrics["cancer_coded_endpoint_only"]["tabular"] = _block(
        test, mlp, prev0, _y_cancer_endpoint
    )

    print(f"bootstrap n={n_boot}...", flush=True)
    boot = {
        "visit": _patient_bootstrap(test, visit_sc, prev0, _y_current, n_boot=n_boot, seed=0),
        "lookup": _patient_bootstrap(test, look_all, prev0, _y_current, n_boot=n_boot, seed=0),
        "tabular": _patient_bootstrap(test, mlp, prev0, _y_current, n_boot=n_boot, seed=0),
    }

    out = {
        "n_train": len(train),
        "n_test": len(test),
        "lookup_score_source_all_test": src_all,
        "lookup_score_source_prev0": dict(Counter(src_prev0)),
        "lookup_score_source_prev0_n": len(look_prev0),
        "chain_length_hist": {str(k): int(v) for k, v in sorted(len_hist.items())},
        "n_chains": len(chain_len),
        "test_prev0_sites": dict(sites),
        "n_anatomical_site_bins": len(LOCATIONS),
        "endpoint_overwrite_test": {
            "n": len(overwrite),
            "n_cancer": sum(r["cancer"] for r in overwrite),
            "n_melanoma": sum(r["melanoma"] for r in overwrite),
            "alignment": dict(Counter(r["alignment"] for r in overwrite)),
            "diagnosis": dict(Counter(r["diagnosis"].lower() or "(empty)" for r in overwrite)),
            "rows": overwrite,
        },
        "test_endpoint_records": {
            "n": len(ep),
            "n_cancer_flag": ep_cancer,
            "diagnosis": dict(ep_dx),
        },
        "test_next_tile_id_present": {"n": next_id, "n_test": len(test), "rate": next_id / max(len(test), 1)},
        "file_order_within_visit_prev0": within,
        "label_sensitivity_prev0": label_metrics,
        "patient_bootstrap_prev0": boot,
        "n_boot": n_boot,
        "exclude_chain_first_n": sum(int(v) for v in prev0_nf),
        "tabular_train": {
            "optimizer": "Adam",
            "lr": 0.001,
            "loss": "BCEWithLogits + pos_weight",
            "hidden": 128,
            "steps": onset_steps,
            "batch_size": 256,
            "seed": 0,
        },
    }
    dest = root / "results" / "revision_audit.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", dest)
    return out


def main():
    run(Path("/root/autodl-tmp/reinforced_derm_agent"))


if __name__ == "__main__":
    main()
