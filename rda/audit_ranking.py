"""Audit ranking metrics, label provenance, and visit/cohort structure."""

from collections import Counter, defaultdict

from .benchmark import attach_prev_derm, history_vector, onset_heuristics, onset_scores, train_onset_scorer
from .metrics import average_precision_grouped, n_unique_scores, ranking_metrics, scores_constant_within


def attach_history(steps):
    by = defaultdict(list)
    for i, step in enumerate(steps):
        by[step["lesion_id"]].append(i)
    for idxs in by.values():
        idxs.sort(key=lambda i: (steps[i].get("visit") or 0, str(steps[i].get("tile_id") or "")))
        prev = 0
        ever = 0
        for k, i in enumerate(idxs):
            step = steps[i]
            derm = int(step.get("derm_visible") or 0)
            step["chain_index"] = k
            step["chain_len"] = len(idxs)
            step["prev_derm_visible"] = prev
            step["ever_derm_before"] = ever
            step["adjacent_onset"] = int(bool(derm) and not prev)
            step["first_observed_onset"] = int(bool(derm) and not ever)
            step["repeat_presence"] = int(bool(derm) and ever)
            prev = derm
            ever = int(ever or derm)
    return steps


def _subset(steps, scores, pred=None, mask=None):
    if mask is None:
        idx = list(range(len(steps)))
    else:
        idx = [i for i, keep in enumerate(mask) if keep]
    y = [int(pred(steps[i])) for i in idx] if pred else [int(steps[i]["action"] == 1) for i in idx]
    sc = [scores[i] for i in idx]
    groups = [steps[i].get("visit") for i in idx]
    return idx, y, sc, groups


def _visit_rates(steps, y_fn, mask=None):
    by = {}
    for i, step in enumerate(steps):
        if mask is not None and not mask[i]:
            continue
        vis = int(step.get("visit") or 0)
        rec = by.setdefault(vis, {"n": 0, "pos": 0})
        rec["n"] += 1
        rec["pos"] += int(y_fn(step))
    out = {}
    for vis in sorted(by):
        n = by[vis]["n"]
        pos = by[vis]["pos"]
        out[str(vis)] = {"n": n, "pos": pos, "rate": pos / max(n, 1)}
    return out


def label_provenance(steps):
    attach_history(steps)
    n = len(steps)
    lesions = defaultdict(list)
    for step in steps:
        lesions[step["lesion_id"]].append(step)
    n_lesion = len(lesions)
    n_adjacent = sum(int(s["adjacent_onset"]) for s in steps)
    n_first = sum(int(s["first_observed_onset"]) for s in steps)
    n_repeat = sum(int(s["repeat_presence"]) for s in steps)
    n_action_acq = sum(int(s["action"] == 1) for s in steps)
    n_adj_not_first = sum(int(s["adjacent_onset"] and not s["first_observed_onset"]) for s in steps)
    n_first_not_action = sum(int(s["first_observed_onset"] and s["action"] != 1) for s in steps)
    first_rec_derm = sum(int(s["chain_index"] == 0 and s.get("derm_visible")) for s in steps)
    first_rec_acq = sum(int(s["chain_index"] == 0 and s["action"] == 1) for s in steps)
    hist_always_false_excise = 0
    hist_none_excise = 0
    alignments = Counter()
    for chain in lesions.values():
        al = chain[0].get("alignment") or "none"
        alignments[al] += 1
        if any(s["action"] == 2 for s in chain) and al == "none":
            hist_none_excise += 1
    n_records_minus_lesions = n - n_lesion
    return {
        "n_lesion_visit_records": n,
        "n_lesion_chains": n_lesion,
        "n_adjacent_pairs_if_linear": n_records_minus_lesions,
        "n_participants": len({s["participant"] for s in steps}),
        "adjacent_onset": n_adjacent,
        "first_observed_onset": n_first,
        "repeat_presence": n_repeat,
        "action_AcquireDerm": n_action_acq,
        "adjacent_equals_action": int(n_adjacent == n_action_acq),
        "adjacent_not_first_observed": n_adj_not_first,
        "first_observed_not_AcquireDerm": n_first_not_action,
        "chain_first_record_with_dermoscopy": first_rec_derm,
        "chain_first_record_AcquireDerm": first_rec_acq,
        "alignment_lesions": dict(alignments),
        "excise_on_alignment_none_lesions": hist_none_excise,
        "excise_on_always_false_without_evidence": hist_always_false_excise,
    }


def _mask_prev0(steps):
    return [not s.get("prev_derm_visible") for s in steps]


def _mask_never_derm(steps):
    return [not s.get("ever_derm_before") for s in steps]


def _mask_not_first_record(steps):
    return [int(s.get("chain_index") or 0) > 0 for s in steps]


def score_block(name, steps, scores, mask, y_fn=None):
    pred = y_fn or (lambda s: int(s["action"] == 1))
    _, y, sc, groups = _subset(steps, scores, pred=pred, mask=mask)
    rates = _visit_rates(steps, pred, mask=mask)
    max_rate = max((v["rate"] for v in rates.values()), default=0.0)
    metrics = ranking_metrics(y, sc)
    metrics.update(
        {
            "name": name,
            "n_unique_scores": n_unique_scores(sc),
            "visit_constant_scores": scores_constant_within(groups, sc),
            "visit_rates": rates,
            "max_visit_rate": max_rate,
            "grouped_ap_le_max_visit_rate": (
                metrics["ap_grouped"] is None or metrics["ap_grouped"] <= max_rate + 1e-12
            ),
        }
    )
    return metrics


def cohort_scores(steps):
    # Higher score to the cohort with typically higher onset base rate in reconstruction.
    return [1.0 if s.get("cohort") == "general" else 0.0 for s in steps]


def cohort_visit_scores(steps):
    return [
        (1.0 if s.get("cohort") == "general" else 0.0) * 10.0 - float(s.get("visit") or 0)
        for s in steps
    ]


def visit_rates_by_cohort(steps, mask):
    out = {}
    for cohort in ("general", "highrisk"):
        cmask = [(mask[i] if mask else True) and s.get("cohort") == cohort for i, s in enumerate(steps)]
        out[cohort] = _visit_rates(steps, lambda s: int(s["action"] == 1), mask=cmask)
        n = sum(int(v) for v in cmask)
        pos = sum(int(steps[i]["action"] == 1) for i, v in enumerate(cmask) if v)
        out[cohort]["overall"] = {"n": n, "pos": pos, "rate": pos / max(n, 1)}
    return out


def audit_split(name, steps, mlp_scores=None):
    attach_history(steps)
    heur = onset_heuristics(steps)
    visit_sc = heur["neg_visit"]
    prev0 = _mask_prev0(steps)
    never = _mask_never_derm(steps)
    not_first = _mask_not_first_record(steps)
    prev0_not_first = [a and b for a, b in zip(prev0, not_first)]
    never_not_first = [a and b for a, b in zip(never, not_first)]
    blocks = {
        "prev0_visit": score_block("prev0_visit", steps, visit_sc, prev0),
        "never_derm_visit": score_block("never_derm_visit", steps, visit_sc, never),
        "prev0_exclude_chain_first_visit": score_block(
            "prev0_exclude_chain_first_visit", steps, visit_sc, prev0_not_first
        ),
        "never_derm_exclude_chain_first_visit": score_block(
            "never_derm_exclude_chain_first_visit", steps, visit_sc, never_not_first
        ),
        "prev0_cohort": score_block("prev0_cohort", steps, cohort_scores(steps), prev0),
        "prev0_cohort_visit": score_block("prev0_cohort_visit", steps, cohort_visit_scores(steps), prev0),
        "majority_prev0": score_block("majority_prev0", steps, [0.0] * len(steps), prev0),
    }
    if mlp_scores is not None:
        blocks["prev0_history"] = score_block("prev0_history", steps, mlp_scores, prev0)
        blocks["never_derm_history"] = score_block("never_derm_history", steps, mlp_scores, never)
        blocks["prev0_exclude_chain_first_history"] = score_block(
            "prev0_exclude_chain_first_history", steps, mlp_scores, prev0_not_first
        )
    y_adj = lambda s: int(s.get("adjacent_onset") or 0)
    y_first = lambda s: int(s.get("first_observed_onset") or 0)
    blocks["prev0_visit_adjacent_label"] = score_block(
        "prev0_visit_adjacent_label", steps, visit_sc, prev0, y_fn=y_adj
    )
    blocks["never_derm_visit_first_onset_label"] = score_block(
        "never_derm_visit_first_onset_label", steps, visit_sc, never, y_fn=y_first
    )
    return {
        "split": name,
        "provenance": label_provenance(steps),
        "cohort_prev0": visit_rates_by_cohort(steps, prev0),
        "ranking": blocks,
    }


def run_audit(splits, train_steps=None, mlp=True, onset_steps=4000, seed=0, hidden=128):
    attach_prev_derm(train_steps or [])
    mlp_scores = {}
    if mlp and train_steps:
        net = train_onset_scorer(train_steps, hidden=hidden, steps=onset_steps, seed=seed)
        for name, rows in splits.items():
            mlp_scores[name] = onset_scores(net, rows)
    report = {
        "eval_kind": "dermworkup_ranking_audit",
        "publication_evidence": False,
        "metric_definitions": {
            "ap_rank": "mean precision at each positive; ties keep input order",
            "ap_grouped": "sklearn-style AP; one PR point per unique score",
            "pr_auc_trapz": "trapezoidal area under grouped PR curve including (0,1)",
            "AcquireDerm": "adjacent-record dermoscopy-presence onset; not a logged clinician action",
            "prev0": "previous chained record has no dermoscopy id; not lifetime-null",
            "never_derm_before": "no dermoscopy id on any earlier chained record",
        },
        "splits": {},
    }
    for name, rows in splits.items():
        report["splits"][name] = audit_split(name, rows, mlp_scores.get(name))
    if train_steps is not None and "test" in splits:
        report["actual_features"] = {
            "history_vector": [
                "prev_derm_visible (constant 0 on prev0)",
                "visit/10",
                "cohort high-risk indicator",
                "naevus count / freckling / prior excision / family history from participant meta",
                "anatomical site one-hot",
            ],
            "note": "Empty participant meta zeroes the numeric patient fields.",
        }
        report["closeout"] = run_closeout(
            splits,
            train_steps,
            mlp_scores,
            onset_steps=onset_steps,
            seed=seed,
            hidden=hidden,
            mlp=mlp,
        )
    return report


def _cell_key(step):
    return (step.get("cohort") or "unknown", int(step.get("visit") or 0))


def _fit_lookup(train_steps, mask):
    n_cell, pos_cell = {}, {}
    n_coh, pos_coh = {}, {}
    n_all = pos_all = 0
    for i, step in enumerate(train_steps):
        if mask is not None and not mask[i]:
            continue
        y = int(step["action"] == 1)
        key = _cell_key(step)
        cohort = step.get("cohort") or "unknown"
        n_cell[key] = n_cell.get(key, 0) + 1
        pos_cell[key] = pos_cell.get(key, 0) + y
        n_coh[cohort] = n_coh.get(cohort, 0) + 1
        pos_coh[cohort] = pos_coh.get(cohort, 0) + y
        n_all += 1
        pos_all += y
    rates = {k: pos_cell[k] / n_cell[k] for k in n_cell}
    cohort_rates = {c: pos_coh[c] / n_coh[c] for c in n_coh}
    glob = pos_all / max(n_all, 1)
    return {
        "cells": {
            f"{k[0]}|{k[1]}": {"n": n_cell[k], "pos": pos_cell[k], "rate": rates[k]}
            for k in sorted(n_cell)
        },
        "cohort_rates": cohort_rates,
        "global_rate": glob,
        "rates": rates,
        "fit_split": "train",
        "empty_cell_backoff": "cohort then global train prevalence",
    }


def _apply_lookup(steps, fitted):
    rates = fitted["rates"]
    cohort_rates = fitted["cohort_rates"]
    glob = fitted["global_rate"]
    scores = []
    src = Counter()
    for step in steps:
        key = _cell_key(step)
        cohort = step.get("cohort") or "unknown"
        if key in rates:
            scores.append(rates[key])
            src["cell"] += 1
        elif cohort in cohort_rates:
            scores.append(cohort_rates[cohort])
            src["cohort"] += 1
        else:
            scores.append(glob)
            src["global"] += 1
    return scores, dict(src)


def _visit_score_table(steps, scores, mask, y_fn):
    by = {}
    for i, step in enumerate(steps):
        if mask is not None and not mask[i]:
            continue
        vis = int(step.get("visit") or 0)
        rec = by.setdefault(vis, {"n": 0, "pos": 0, "scores": []})
        rec["n"] += 1
        rec["pos"] += int(y_fn(step))
        rec["scores"].append(float(scores[i]))
    out = {}
    for vis in sorted(by):
        sc = by[vis]["scores"]
        n = by[vis]["n"]
        pos = by[vis]["pos"]
        out[str(vis)] = {
            "n": n,
            "pos": pos,
            "rate": pos / max(n, 1),
            "score_min": min(sc),
            "score_max": max(sc),
            "n_unique_scores": len(set(sc)),
        }
    return out


def _permute_ap(y, scores, n_perm=30, seed=0):
    import random

    from .metrics import average_precision_rank

    rng = random.Random(seed)
    grouped0 = average_precision_grouped(y, scores)
    rank0 = average_precision_rank(y, scores)
    idx = list(range(len(y)))
    ranks = []
    groups = []
    for _ in range(n_perm):
        rng.shuffle(idx)
        y2 = [y[i] for i in idx]
        s2 = [scores[i] for i in idx]
        ranks.append(average_precision_rank(y2, s2))
        groups.append(average_precision_grouped(y2, s2))
    return {
        "n_perm": n_perm,
        "ap_rank_original": rank0,
        "ap_standard_original": grouped0,
        "ap_rank_perm_min": min(ranks) if ranks else None,
        "ap_rank_perm_max": max(ranks) if ranks else None,
        "ap_rank_perm_mean": sum(ranks) / len(ranks) if ranks else None,
        "ap_standard_invariant": all(abs(g - grouped0) < 1e-12 for g in groups),
    }


def acquire_sources(steps):
    attach_history(steps)
    counts = Counter()
    for step in steps:
        if step["action"] != 1:
            continue
        if int(step.get("chain_index") or 0) == 0:
            counts["chain_first_present"] += 1
        elif step.get("first_observed_onset"):
            counts["later_first_observed"] += 1
        else:
            counts["re_onset"] += 1
    counts["total_AcquireDerm"] = sum(counts.values())
    return dict(counts)


def chain_integrity(steps):
    by = defaultdict(list)
    for step in steps:
        by[step["lesion_id"]].append(step)
    n_multi_patient = 0
    n_dup_visit = 0
    n_unsorted_in_file = 0
    for recs in by.values():
        patients = {s["participant"] for s in recs}
        if len(patients) > 1:
            n_multi_patient += 1
        visits = [int(s.get("visit") or 0) for s in recs]
        if len(visits) != len(set(visits)):
            n_dup_visit += 1
        if visits != sorted(visits):
            n_unsorted_in_file += 1
    return {
        "n_chains": len(by),
        "n_multi_patient_chains": n_multi_patient,
        "n_chains_with_duplicate_visit": n_dup_visit,
        "n_chains_unsorted_in_file_order": n_unsorted_in_file,
        "ordered_by": "visit then tile_id",
    }


def _filter(steps, mask):
    return [s for s, keep in zip(steps, mask) if keep]


def run_closeout(splits, train_steps, mlp_scores, onset_steps=4000, seed=0, hidden=128, mlp=True):
    import random

    for rows in [train_steps, *splits.values()]:
        attach_history(rows)
    test = splits["test"]
    y_fn = lambda s: int(s["action"] == 1)
    heur = onset_heuristics(test)
    visit_sc = heur["neg_visit"]
    prev0 = _mask_prev0(test)
    not_first = _mask_not_first_record(test)
    prev0_not_first = [a and b for a, b in zip(prev0, not_first)]
    drop_singleton = [keep and int(s.get("visit") or 0) != 1 for s, keep in zip(test, prev0_not_first)]

    train_prev0 = _mask_prev0(train_steps)
    train_not_first = _mask_not_first_record(train_steps)
    train_prev0_not_first = [a and b for a, b in zip(train_prev0, train_not_first)]
    lookup_all = _fit_lookup(train_steps, train_prev0)
    lookup_nofirst = _fit_lookup(train_steps, train_prev0_not_first)
    look_all, src_all = _apply_lookup(test, lookup_all)
    look_nf, src_nf = _apply_lookup(test, lookup_nofirst)

    _, y_prev0, sc_visit, _ = _subset(test, visit_sc, pred=y_fn, mask=prev0)

    meta_empty = sum(int(not (s.get("meta") or {})) for s in test)
    out = {
        "actual_features_on_test": {
            "n": len(test),
            "meta_empty": meta_empty,
            "usable_on_prev0": ["visit", "cohort", "anatomical site"],
            "constant_on_prev0": ["prev_derm_visible"],
        },
        "acquire_sources": {name: acquire_sources(rows) for name, rows in splits.items()},
        "chain_integrity": {name: chain_integrity(rows) for name, rows in splits.items()},
        "exclude_chain_first_visit_groups": _visit_score_table(test, visit_sc, prev0_not_first, y_fn),
        "exclude_chain_first_visit_drop_visit1_singleton": score_block(
            "drop_visit1_n1", test, visit_sc, drop_singleton
        ),
        "permute_visit_prev0": _permute_ap(y_prev0, sc_visit, n_perm=30, seed=seed),
        "train_fitted_cohort_visit_lookup": {
            "fit": {
                k: lookup_all[k]
                for k in ("cells", "cohort_rates", "global_rate", "fit_split", "empty_cell_backoff")
            },
            "test_prev0": score_block("lookup_prev0", test, look_all, prev0),
            "test_prev0_score_source": src_all,
        },
        "exclude_chain_first_protocol": {
            "rescore": "model trained on all train records; score test chain_index>0",
            "retrain": "train on train chain_index>0 and prev0; score the same test subset",
        },
        "exclude_chain_first_rescore": {
            "visit": score_block("rescore_visit", test, visit_sc, prev0_not_first),
            "lookup": score_block("rescore_lookup", test, look_nf, prev0_not_first),
        },
    }
    if mlp_scores and mlp_scores.get("test") is not None:
        mlp_test = mlp_scores["test"]
        _, _, sc_mlp, _ = _subset(test, mlp_test, pred=y_fn, mask=prev0)
        out["permute_history_prev0"] = _permute_ap(y_prev0, sc_mlp, n_perm=30, seed=seed)
        out["exclude_chain_first_rescore"]["history"] = score_block(
            "rescore_history", test, mlp_test, prev0_not_first
        )
    if mlp:
        train_nf = _filter(train_steps, train_prev0_not_first)
        random.seed(seed)
        net = train_onset_scorer(train_nf, hidden=hidden, steps=onset_steps, seed=seed)
        mlp_nf = onset_scores(net, test)
        out["exclude_chain_first_retrain"] = {
            "n_train": len(train_nf),
            "visit": score_block("retrain_visit", test, visit_sc, prev0_not_first),
            "lookup": score_block("retrain_lookup", test, look_nf, prev0_not_first),
            "history": score_block("retrain_history", test, mlp_nf, prev0_not_first),
        }
    return out

