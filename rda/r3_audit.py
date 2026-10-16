"""Round-3 reviewer analyses for Reviewer 1 (Major Revision, 08 Oct 2026).

Three requests are not covered by revision_audit.json / round2_review_audit.json:

  R3-A  Ranking performance decomposed by AcquireDerm label source
        (chain-first presence / later first observation / re-onset after a gap),
        to test whether the 90.4% chain-first share drives the reported skill.
  R3-B  Repeated patient-level splits. The split is a deterministic
        sha256("seed|participant") hash, so seeds are reproducible; only the
        split seed varies, the tabular model seed stays at 0 so the comparison
        isolates the split.
  R3-C  Patient-level bootstrap CIs on the chain-first-excluded evaluation
        subset, including the variant that also drops the singleton visit-1 group.

Run on llm_01 with the real jsonl.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .audit_ranking import (
    _apply_lookup,
    _fit_lookup,
    _mask_not_first_record,
    _mask_prev0,
    attach_history,
    acquire_sources,
)
from .benchmark import onset_heuristics, onset_scores, train_onset_scorer
from .dataset import load_jsonl
from .metrics import ranking_metrics
from .revision_audit import _ci, _patient_bootstrap, _y_current
from .split import patient_split

STRATA = ("chain_first_present", "later_first_observed", "re_onset")


def _stratum(s):
    """Same rule as audit_ranking.acquire_sources, per record."""
    if int(s.get("action") or 0) != 1:
        return None
    if int(s.get("chain_index") or 0) == 0:
        return "chain_first_present"
    if s.get("first_observed_onset"):
        return "later_first_observed"
    return "re_onset"


def _mask_only_stratum(steps, stratum):
    """Negatives are all kept; a positive is kept only if it is in `stratum`."""
    return [_stratum(s) is None or _stratum(s) == stratum for s in steps]


def _keep_only_visit(steps, keep_visits):
    return [int(s.get("visit") or 0) in keep_visits for s in steps]


def _metrics(y, sc):
    return ranking_metrics(y, sc)


def _subset_block(steps, scores, mask):
    idx = [i for i, k in enumerate(mask) if k]
    y = [_y_current(steps[i]) for i in idx]
    sc = [scores[i] for i in idx]
    return _metrics(y, sc)


def _one_split(all_steps, proto_seed, onset_steps=4000, n_boot=500, do_boot=True):
    proto = {"split": {"unit": "participant", "train": 0.7, "val": 0.1,
                       "test": 0.2, "seed": proto_seed}}
    grouped, n_patients = patient_split(all_steps, proto)
    tr, te = grouped["train"], grouped["test"]
    attach_history(tr)
    attach_history(te)

    tr_prev0 = _mask_prev0(tr)
    te_prev0 = _mask_prev0(te)
    te_notfirst = _mask_not_first_record(te)
    te_prev0_nf = [a and b for a, b in zip(te_prev0, te_notfirst)]

    fitted = _fit_lookup(tr, tr_prev0)
    look, _look_src = _apply_lookup(te, fitted)
    visit_sc = onset_heuristics(te)["neg_visit"]

    net = train_onset_scorer(tr, hidden=128, steps=onset_steps, seed=0)
    mlp = onset_scores(net, te)

    scores = {"visit": visit_sc, "lookup": look, "tabular": mlp}

    out = {
        "seed": proto_seed,
        "n_train_records": len(tr),
        "n_test_records": len(te),
        "n_train_participants": n_patients["train"],
        "n_test_participants": n_patients["test"],
        "prev0": {k: _subset_block(te, v, te_prev0) for k, v in scores.items()},
        "prev0_not_chain_first": {k: _subset_block(te, v, te_prev0_nf) for k, v in scores.items()},
    }

    # singleton-visit-1 sensitivity: also drop the one-record visit-1 group
    keep = [v for v in sorted({int(s.get("visit") or 0) for s in te})
            if not (_keep_only_visit(te, {v}) and
                    sum(1 for i, k in enumerate(te_prev0_nf) if k and int(te[i].get("visit") or 0) == v) < 50)]
    te_prev0_nf_nov1 = [a and b for a, b in
                        zip(te_prev0_nf, _keep_only_visit(te, set(keep)))]
    out["prev0_not_chain_first_drop_visit1"] = {
        k: _subset_block(te, v, te_prev0_nf_nov1) for k, v in scores.items()
    }
    out["prev0_not_chain_first_drop_visit1"]["kept_visits"] = keep

    if do_boot:
        out["bootstrap_prev0_not_chain_first"] = {
            k: _patient_bootstrap(te, v, te_prev0_nf, _y_current, n_boot=n_boot, seed=0)
            for k, v in scores.items()
        }
        out["bootstrap_prev0_not_chain_first_drop_visit1"] = {
            k: _patient_bootstrap(te, v, te_prev0_nf_nov1, _y_current, n_boot=n_boot, seed=0)
            for k, v in scores.items()
        }

    return out


def run(root: Path, seeds=(0, 1, 2, 3, 4), n_boot: int = 500, onset_steps: int = 4000):
    root = Path(root)
    results = root / "results"

    train0 = load_jsonl(results / "train.jsonl")
    val0 = load_jsonl(results / "val.jsonl")
    test0 = load_jsonl(results / "test.jsonl")
    all_steps = train0 + val0 + test0
    attach_history(all_steps)
    n_participants = len({s["participant"] for s in all_steps})

    print(f"pool: {len(all_steps)} records / {n_participants} participants", flush=True)

    # ---------------- R3-A: stratified performance on the published split ----
    attach_history(train0)
    attach_history(val0)
    attach_history(test0)
    prev0 = _mask_prev0(test0)
    fitted0 = _fit_lookup(train0, _mask_prev0(train0))
    look0, _look_src0 = _apply_lookup(test0, fitted0)
    visit0 = onset_heuristics(test0)["neg_visit"]
    net0 = train_onset_scorer(train0, hidden=128, steps=onset_steps, seed=0)
    mlp0 = onset_scores(net0, test0)
    scores0 = {"visit": visit0, "lookup": look0, "tabular": mlp0}

    strat = {}
    for s in STRATA:
        m = [a and b for a, b in zip(prev0, _mask_only_stratum(test0, s))]
        strat[s] = {k: _subset_block(test0, v, m) for k, v in scores0.items()}
        strat[s]["n_source_in_test_prev0"] = int(sum(
            1 for i, k in enumerate(prev0) if k and _stratum(test0[i]) == s))
    print("R3-A stratified done", flush=True)

    # sensitivity of the chain-first share to the 18 same-visit duplicate chains
    dup_chains = set()
    by_les = {}
    for i, s in enumerate(test0):
        by_les.setdefault(s["lesion_id"], []).append(i)
    for lid, idxs in by_les.items():
        vs = [int(test0[i].get("visit") or 0) for i in idxs]
        if len(vs) != len(set(vs)):
            dup_chains.add(lid)
    src_all = acquire_sources(test0)
    kept = [i for i, s in enumerate(test0) if s["lesion_id"] not in dup_chains]
    src_drop = acquire_sources([test0[i] for i in kept])

    # ---------------- R3-B / R3-C: repeated patient-level splits ---------------
    splits = []
    for sd in seeds:
        print(f"R3-B/C split seed={sd} ...", flush=True)
        splits.append(_one_split(all_steps, sd, onset_steps=onset_steps,
                                 n_boot=n_boot, do_boot=(sd == 0)))

    out = {
        "eval_kind": "dermworkup_r3_reviewer_audit",
        "publication_evidence": None,   # author decision: not set by this script
        "n_pool_records": len(all_steps),
        "n_pool_participants": n_participants,
        "stratum_rule": ("action==1; chain_index==0 -> chain_first_present; "
                         "elif first_observed_onset -> later_first_observed; "
                         "else re_onset. Non-positives are kept as negatives."),
        "stratified_prev0_seed0": strat,
        "acquire_sources_test": src_all,
        "acquire_sources_test_drop_dup_visit_chains": src_drop,
        "n_test_dup_visit_chains": len(dup_chains),
        "repeated_splits": splits,
        "seeds": list(seeds),
        "n_boot": n_boot,
        "tabular_train": {"optimizer": "Adam", "lr": 0.001, "hidden": 128,
                          "steps": onset_steps, "batch_size": 256,
                          "model_seed": 0, "split_seed_varies": True},
    }
    dest = results / "r3_audit.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", dest, flush=True)
    return out


def main():
    run(Path("/root/autodl-tmp/reinforced_derm_agent"))


if __name__ == "__main__":
    main()
