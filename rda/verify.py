from collections import Counter

from .benchmark import attach_prev_derm, onset_heuristics, onset_labels, onset_scores, score_binary, train_onset_scorer
from .fields import present


def split_integrity(splits):
    patients = {name: {s["participant"] for s in rows} for name, rows in splits.items()}
    lesions = {name: {s["lesion_id"] for s in rows} for name, rows in splits.items()}
    names = list(splits)
    patient_leak = []
    lesion_leak = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            patient_leak.extend(sorted(patients[a] & patients[b]))
            lesion_leak.extend(sorted(lesions[a] & lesions[b]))
    return {
        "n_patients": {k: len(v) for k, v in patients.items()},
        "n_lesions": {k: len(v) for k, v in lesions.items()},
        "patient_leak": patient_leak[:10],
        "lesion_leak": lesion_leak[:10],
        "patient_leak_n": len(patient_leak),
        "lesion_leak_n": len(lesion_leak),
        "ok": not patient_leak and not lesion_leak,
    }


def id_patterns(steps):
    return {
        "tile": _id_kind_counts(s.get("tile_id") for s in steps),
        "derm": _id_kind_counts(s.get("derm_id") for s in steps),
    }


def _id_kind_counts(values):
    kinds = Counter()
    nonempty = 0
    for raw in values:
        if not present(raw):
            kinds["empty"] += 1
            continue
        nonempty += 1
        text = str(raw).strip().lower()
        if "isic" in text:
            kinds["contains_isic"] += 1
        elif text[:4].isdigit() or text.startswith("isic"):
            kinds["numeric_or_isic"] += 1
        else:
            kinds["other"] += 1
        head = text.split("_")[0][:8]
        if head:
            kinds[f"head:{head}"] += 1
    heads = {k.replace("head:", ""): v for k, v in kinds.items() if str(k).startswith("head:") and v >= 50}
    top = sorted(heads.items(), key=lambda kv: -kv[1])[:8]
    return {
        "n": nonempty + kinds.get("empty", 0),
        "nonempty": nonempty,
        "contains_isic": kinds.get("contains_isic", 0),
        "empty": kinds.get("empty", 0),
        "other": kinds.get("other", 0),
        "top_heads_n_ge_50": top,
    }


def visit_acquire(steps):
    attach_prev_derm(steps)
    by = {}
    for step in steps:
        if step.get("prev_derm_visible"):
            continue
        vis = int(step.get("visit") or 0)
        rec = by.setdefault(vis, {"n": 0, "acquire": 0})
        rec["n"] += 1
        rec["acquire"] += int(step["action"] == 1)
    out = {}
    for vis in sorted(by):
        n = by[vis]["n"]
        out[str(vis)] = {"n": n, "acquire": by[vis]["acquire"], "rate": by[vis]["acquire"] / max(n, 1)}
    return out


def prev0_scores(steps, scores):
    y = onset_labels(steps)
    idx = [i for i, s in enumerate(steps) if not s.get("prev_derm_visible")]
    y0 = [y[i] for i in idx]
    s0 = [scores[i] for i in idx]
    return score_binary(y0, s0, sum(y0) / max(len(y0), 1))


def cohort_prev0(steps, scores):
    y = onset_labels(steps)
    out = {}
    for cohort in ("general", "highrisk"):
        idx = [i for i, s in enumerate(steps) if (not s.get("prev_derm_visible")) and s.get("cohort") == cohort]
        if not idx:
            continue
        y0 = [y[i] for i in idx]
        out[cohort] = score_binary(y0, [scores[i] for i in idx], sum(y0) / max(len(y0), 1))
        out[cohort]["n"] = len(idx)
    return out


def onset_block(net, steps, ablate=None):
    y = onset_labels(steps)
    scores = onset_scores(net, steps, ablate=ablate)
    heur = onset_heuristics(steps)
    prev = sum(y) / max(len(y), 1)
    return {
        "all": score_binary(y, scores, prev),
        "prev0": prev0_scores(steps, scores),
        "prev0_neg_visit": prev0_scores(steps, heur["neg_visit"]),
        "cohort_prev0": cohort_prev0(steps, scores),
    }
