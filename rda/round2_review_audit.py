"""Round-2 reviewer analyses. Run on llm_01 with real tiles.csv and jsonl."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .audit_ranking import (
    _apply_lookup,
    _fit_lookup,
    _mask_not_first_record,
    _mask_prev0,
    attach_history,
)
from .benchmark import onset_heuristics, onset_scores, train_onset_scorer
from .dataset import load_jsonl
from .fields import parse_participant_visit, present
from .io import rename_columns
from .metrics import ranking_metrics
from .trajectories import NEXT_KEYS, _row_id
from .unionfind import UnionFind


def _compact(metrics):
    return {
        k: metrics[k]
        for k in ("n", "n_pos", "prevalence", "auroc", "ap_grouped", "ap_rank")
        if k in metrics
    }


def _y_action(s):
    return int(s["action"] == 1)


def _y_adjacent(s):
    return int(bool(s.get("adjacent_onset")))


def _y_visit_endpoint_only(s):
    if s["action"] == 2 and (s.get("alignment") or "") == "visit":
        return 0
    return int(bool(s.get("adjacent_onset")))


def _y_lesion_endpoint_only(s):
    if s["action"] == 2 and (s.get("alignment") or "") == "lesion_level":
        return 0
    return int(bool(s.get("adjacent_onset")))


def _y_cancer_endpoint(s):
    if s["action"] == 2 and int(s.get("cancer") or 0):
        return 0
    return int(bool(s.get("adjacent_onset")))


def _block(steps, scores, mask, y_fn):
    y = [y_fn(s) for s, keep in zip(steps, mask) if keep]
    sc = [scores[i] for i, keep in enumerate(mask) if keep]
    return _compact(ranking_metrics(y, sc))


def _chain_table(steps):
    by = defaultdict(list)
    for s in steps:
        by[s["lesion_id"]].append(s)
    lengths = Counter()
    n_multi = 0
    n_dup_visit = 0
    n_uid_dup = 0
    n_len1 = 0
    dup_visit_ids = []
    for lid, recs in by.items():
        recs = sorted(recs, key=lambda r: (int(r.get("visit") or 0), str(r.get("tile_id") or "")))
        lengths[len(recs)] += 1
        if len({r["participant"] for r in recs}) > 1:
            n_multi += 1
        visits = [int(r.get("visit") or 0) for r in recs]
        if len(visits) != len(set(visits)):
            n_dup_visit += 1
            dup_visit_ids.append(lid)
        uids = [str(r.get("tile_id") or "") for r in recs]
        if len(uids) != len(set(uids)):
            n_uid_dup += 1
        if len(recs) == 1:
            n_len1 += 1
    return {
        "n_chains": len(by),
        "length_hist": {str(k): int(v) for k, v in sorted(lengths.items())},
        "n_multi_patient": n_multi,
        "n_duplicate_visit": n_dup_visit,
        "n_duplicate_uid": n_uid_dup,
        "n_length_1": n_len1,
        "dup_visit_ids": dup_visit_ids,
    }


def _correspondence_stats(tile_csv: Path):
    df = pd.read_csv(tile_csv, dtype=str, keep_default_na=False)
    df, _ = rename_columns(df)
    uids = []
    uid_to_part = {}
    n_blank_uid = 0
    for row in df.to_dict("records"):
        uid = _row_id(row)
        if not present(uid):
            n_blank_uid += 1
            continue
        part, _ = parse_participant_visit(row.get("participant_visit"), row.get("participant_number"))
        uids.append(uid)
        uid_to_part.setdefault(uid, set()).add(part)
    uid_set = set(uids)
    n_rows = len(df)
    n_unique_uid = len(uid_set)
    n_uid_collision = n_rows - n_blank_uid - n_unique_uid

    n_any_next = 0
    n_next_values = 0
    n_next_in_uid = 0
    n_next_missing = 0
    n_next_same_uid = 0
    n_next_cross_patient = 0
    n_next_same_patient = 0
    uf = UnionFind()
    for row in df.to_dict("records"):
        uid = _row_id(row)
        if not present(uid):
            continue
        uf.add(uid)
        part, _ = parse_participant_visit(row.get("participant_visit"), row.get("participant_number"))
        has_next = False
        for key in NEXT_KEYS:
            nxt = row.get(key)
            if not present(nxt):
                continue
            nxt = str(nxt).strip()
            has_next = True
            n_next_values += 1
            uf.union(uid, nxt)
            if nxt == uid:
                n_next_same_uid += 1
            if nxt in uid_set:
                n_next_in_uid += 1
                other_parts = uid_to_part.get(nxt) or set()
                if other_parts and part not in other_parts:
                    n_next_cross_patient += 1
                elif part in other_parts:
                    n_next_same_patient += 1
            else:
                n_next_missing += 1
        if has_next:
            n_any_next += 1

    # Cross-patient connected components among real rows.
    groups = defaultdict(set)
    part_of = {}
    for row in df.to_dict("records"):
        uid = _row_id(row)
        if not present(uid):
            continue
        part, _ = parse_participant_visit(row.get("participant_visit"), row.get("participant_number"))
        part_of[uid] = part
        groups[uf.find(uid)].add(uid)
    n_comp = len(groups)
    n_comp_multi_patient = 0
    n_comp_len_ge2 = 0
    for members in groups.values():
        parts = {part_of[u] for u in members if u in part_of}
        if len(parts) > 1:
            n_comp_multi_patient += 1
        if len(members) >= 2:
            n_comp_len_ge2 += 1

    return {
        "n_tile_rows": n_rows,
        "n_blank_uid": n_blank_uid,
        "n_unique_uid": n_unique_uid,
        "n_uid_row_collisions_extra": n_uid_collision,
        "n_rows_with_any_next_field": n_any_next,
        "rate_rows_with_any_next_field": n_any_next / max(n_rows, 1),
        "n_next_field_values": n_next_values,
        "n_next_values_found_as_uid": n_next_in_uid,
        "n_next_values_missing_as_uid": n_next_missing,
        "rate_next_values_found_as_uid": n_next_in_uid / max(n_next_values, 1),
        "n_next_self_loop": n_next_same_uid,
        "n_next_cross_patient_uid": n_next_cross_patient,
        "n_next_same_patient_uid": n_next_same_patient,
        "n_union_components": n_comp,
        "n_union_components_len_ge2": n_comp_len_ge2,
        "n_union_components_multi_patient": n_comp_multi_patient,
    }


def run(root: Path):
    train = load_jsonl(root / "results" / "train.jsonl")
    val = load_jsonl(root / "results" / "val.jsonl")
    test = load_jsonl(root / "results" / "test.jsonl")
    for rows in (train, val, test):
        attach_history(rows)

    prev0 = _mask_prev0(test)
    not_first = _mask_not_first_record(test)
    heur = onset_heuristics(test)
    visit_sc = heur["neg_visit"]
    fitted = _fit_lookup(train, _mask_prev0(train))
    look_all, src_all = _apply_lookup(test, fitted)
    look_prev0, src_prev0 = _apply_lookup([s for s, k in zip(test, prev0) if k], fitted)

    chain_test = _chain_table(test)
    chain_train = _chain_table(train)
    chain_val = _chain_table(val)
    dup_set = set(chain_test["dup_visit_ids"])
    drop_dup = [s["lesion_id"] not in dup_set for s in test]
    prev0_drop_dup = [a and b for a, b in zip(prev0, drop_dup)]

    print("training tabular model...", flush=True)
    net = train_onset_scorer(train, hidden=128, steps=4000, seed=0)
    mlp = onset_scores(net, test)

    general = [s.get("cohort") == "general" for s in test]
    highrisk = [s.get("cohort") == "highrisk" for s in test]
    prev0_gen = [a and b for a, b in zip(prev0, general)]
    prev0_hr = [a and b for a, b in zip(prev0, highrisk)]

    label_rules = {
        "recorded_onset_default": _y_action,
        "adjacent_no_endpoint": _y_adjacent,
        "visit_aligned_endpoint_only": _y_visit_endpoint_only,
        "lesion_level_endpoint_only": _y_lesion_endpoint_only,
        "cancer_coded_endpoint_only": _y_cancer_endpoint,
    }
    label_metrics = {}
    for name, fn in label_rules.items():
        label_metrics[name] = {
            "visit": _block(test, visit_sc, prev0, fn),
            "lookup": _block(test, look_all, prev0, fn),
            "tabular": _block(test, mlp, prev0, fn),
        }

    subsets = {
        "prev0": prev0,
        "prev0_drop_duplicate_visit_chains": prev0_drop_dup,
        "prev0_general_only": prev0_gen,
        "prev0_highrisk_only": prev0_hr,
        "prev0_not_chain_first": [a and b for a, b in zip(prev0, not_first)],
    }
    subset_metrics = {}
    for name, mask in subsets.items():
        subset_metrics[name] = {
            "visit": _block(test, visit_sc, mask, _y_action),
            "lookup": _block(test, look_all, mask, _y_action),
            "tabular": _block(test, mlp, mask, _y_action),
        }

    n_pos = sum(int(s["action"] == 1) for s in test)
    n_pos_first = sum(int(s["action"] == 1 and int(s.get("chain_index") or 0) == 0) for s in test)
    n_pos_first_drop = sum(
        int(s["action"] == 1 and int(s.get("chain_index") or 0) == 0 and s["lesion_id"] not in dup_set)
        for s in test
    )
    n_pos_drop = sum(int(s["action"] == 1 and s["lesion_id"] not in dup_set) for s in test)

    print("correspondence fields on tiles.csv...", flush=True)
    corr = _correspondence_stats(root / "data" / "csv" / "tiles.csv")

    out = {
        "test_chain_table": {k: v for k, v in chain_test.items() if k != "dup_visit_ids"},
        "train_chain_table": {k: v for k, v in chain_train.items() if k != "dup_visit_ids"},
        "val_chain_table": {k: v for k, v in chain_val.items() if k != "dup_visit_ids"},
        "test_onset": {
            "n_recorded_onset": n_pos,
            "n_chain_first": n_pos_first,
            "frac_chain_first": n_pos_first / max(n_pos, 1),
            "n_recorded_onset_drop_dup_visit_chains": n_pos_drop,
            "n_chain_first_drop_dup_visit_chains": n_pos_first_drop,
            "frac_chain_first_drop_dup_visit_chains": n_pos_first_drop / max(n_pos_drop, 1),
        },
        "lookup_score_source_all_test": src_all,
        "lookup_score_source_prev0": dict(Counter(src_prev0)),
        "label_rule_sensitivity_prev0": label_metrics,
        "subset_sensitivity": subset_metrics,
        "correspondence": corr,
        "test_cells_prev0": _fit_lookup(test, prev0)["cells"],
        "train_cells_prev0": fitted["cells"],
    }
    dest = root / "results" / "round2_review_audit.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", dest)
    return out


def main():
    run(Path("/root/autodl-tmp/reinforced_derm_agent"))


if __name__ == "__main__":
    main()
