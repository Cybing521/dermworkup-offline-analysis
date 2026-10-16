from collections import Counter, defaultdict

from .fields import cohort_of, has_any, parse_participant_visit, present, truthy
from .unionfind import UnionFind

NEXT_KEYS = [f"corresponding_lesionid_for_next_{k}_visit" for k in range(1, 7)]
WATCH, ACQUIRE, EXCISE = 0, 1, 2
ACTION_NAME = {0: "WatchTBP", 1: "AcquireDerm", 2: "ReferExcise"}


def _row_id(row):
    return str(row.get("uniqie_id") or row.get("unique_id") or "").strip()


def _derm_id(row):
    for key in (
        "corresponding_dermoscopic_image_id",
        "dermoscopic_image_id",
        "corresponding_dermoscopic",
    ):
        val = row.get(key)
        if present(val):
            return val
    for key, val in row.items():
        if str(key).startswith("corresponding_dermoscopic") and present(val):
            return val
    return None


def build_lesion_chains(tile_df):
    uf = UnionFind()
    records = []
    for row in tile_df.to_dict("records"):
        uid = _row_id(row)
        if not present(uid):
            continue
        uf.add(uid)
        for key in NEXT_KEYS:
            nxt = row.get(key)
            if present(nxt):
                uf.union(uid, str(nxt).strip())
        participant, visit = parse_participant_visit(
            row.get("participant_visit"), row.get("participant_number")
        )
        records.append({**row, "_uid": uid, "_participant": participant, "_visit": visit})
    groups = defaultdict(list)
    for rec in records:
        groups[uf.find(rec["_uid"])].append(rec)
    chains = []
    for lesion_id, items in groups.items():
        items.sort(key=lambda r: (r["_visit"], r["_uid"]))
        seen = set()
        ordered = []
        for rec in items:
            if rec["_uid"] in seen:
                continue
            seen.add(rec["_uid"])
            ordered.append(rec)
        if ordered:
            chains.append({"lesion_id": lesion_id, "visits": ordered})
    return chains


def _cancer_flags(visits, proto):
    diagnoses = [v.get("diagnosis") for v in visits]
    cancer = any(has_any(d, proto["cancer_substrings"]) for d in diagnoses)
    melanoma = any(has_any(d, proto["melanoma_substrings"]) for d in diagnoses)
    hist = [truthy(v.get("histopathologic_result")) for v in visits]
    if any(hist) and not all(hist):
        return cancer, melanoma, "visit", next(i for i, h in enumerate(hist) if h)
    if any(hist) or cancer:
        return cancer, melanoma, "lesion_level", len(visits) - 1
    return False, False, "none", None


def _cost_miss(cancer, action, terminal, alignment, proto):
    if not cancer or action == EXCISE:
        return 0.0
    rule = (proto.get("alignment") or {}).get("cost_miss_lesion_level", "terminal_only")
    if alignment == "lesion_level" and rule == "terminal_only" and not terminal:
        return 0.0
    return proto["costs"]["miss_non_excise_on_cancer"]


def outcome_under_action(step, action, proto):
    action = int(action)
    cancer = bool(step.get("cancer"))
    alignment = step.get("alignment") or "none"
    terminal = bool(step.get("done")) or action == EXCISE
    rw, costs = proto["rewards"], proto["costs"]
    reward = 0.0
    if action == ACQUIRE:
        reward += rw["derm_resource"]
    if terminal and action == EXCISE and cancer:
        reward += rw["catch_cancer"]
    if terminal and action != EXCISE and not cancer:
        reward += rw["watch_benign_terminal"]
    return {
        "reward": reward,
        "cost_miss": _cost_miss(cancer, action, terminal, alignment, proto),
        "cost_over": costs["over_excise_on_benign"] if ((not cancer) and action == EXCISE) else 0.0,
    }


def dataset_gate(stats, proto):
    n = stats.get("n_transitions") or 0
    if not n:
        return "fail_no_transitions"
    reasons = stats.get("reasons") or {}
    for name in proto["actions"]:
        if not reasons.get(f"action_{name}", 0):
            return "fail_missing_action"
    return "pass"


def _actions(visits, excise_at):
    actions = []
    prev_derm = False
    for i, vis in enumerate(visits):
        has_derm = present(_derm_id(vis))
        action = WATCH
        if has_derm and not prev_derm:
            action = ACQUIRE
        if excise_at is not None and i == excise_at:
            action = EXCISE
        actions.append(action)
        prev_derm = has_derm
    return actions


def transitions_from_chain(chain, participant_row, proto):
    visits = chain["visits"]
    if len(visits) < proto["min_visits"]:
        return [], {"reason": "short", "n": len(visits)}
    participant = visits[0]["_participant"]
    arm = str((participant_row or {}).get("study_arm") or "").strip().lower()
    if proto["exclude"].get("hops_control") and arm in {"control", "ctrl"}:
        return [], {"reason": "hops_control"}
    cancer, melanoma, alignment, excise_at = _cancer_flags(visits, proto)
    actions = _actions(visits, excise_at)
    rw, costs = proto["rewards"], proto["costs"]
    steps = []
    for i, vis in enumerate(visits):
        action = actions[i]
        terminal = action == EXCISE or i == len(visits) - 1
        nxt = visits[i + 1] if i + 1 < len(visits) and action != EXCISE else None
        reward = 0.0
        if action == ACQUIRE:
            reward += rw["derm_resource"]
        if terminal and action == EXCISE and cancer:
            reward += rw["catch_cancer"]
        if terminal and action != EXCISE and not cancer:
            reward += rw["watch_benign_terminal"]
        cost_miss = _cost_miss(cancer, action, terminal, alignment, proto)
        cost_over = costs["over_excise_on_benign"] if ((not cancer) and action == EXCISE) else 0.0
        steps.append(
            {
                "lesion_id": chain["lesion_id"],
                "participant": participant,
                "cohort": cohort_of(participant),
                "visit": vis["_visit"],
                "tile_id": vis["_uid"],
                "derm_id": str(_derm_id(vis) or ""),
                "derm_visible": int(present(_derm_id(vis))),
                "next_derm_visible": int(present(_derm_id(nxt))) if nxt else 0,
                "location": vis.get("lesion_simple_location") or "",
                "diagnosis": vis.get("diagnosis") or "",
                "action": action,
                "action_name": ACTION_NAME[action],
                "reward": reward,
                "cost_miss": cost_miss,
                "cost_over": cost_over,
                "done": int(terminal),
                "cancer": int(cancer),
                "melanoma": int(melanoma),
                "alignment": alignment,
                "next_tile_id": nxt["_uid"] if nxt else "",
                "meta": participant_row or {},
            }
        )
        if action == EXCISE:
            break
    return steps, {"reason": "ok", "alignment": alignment, "n": len(steps), "cancer": cancer}


def _pid_key(pid):
    return "".join(ch.lower() for ch in str(pid) if ch.isalnum())


def build_transitions(tile_df, participant_df, proto):
    participants = {}
    if participant_df is not None:
        for row in participant_df.to_dict("records"):
            pid = str(row.get("participant_number") or "").strip()
            if pid:
                participants[_pid_key(pid)] = row
    chains = build_lesion_chains(tile_df)
    transitions = []
    stats = Counter()
    alignments = Counter()
    for chain in chains:
        pid = chain["visits"][0]["_participant"]
        steps, info = transitions_from_chain(chain, participants.get(_pid_key(pid)), proto)
        stats[info["reason"]] += 1
        if info["reason"] != "ok":
            continue
        alignments[info.get("alignment", "none")] += 1
        stats["transitions"] += len(steps)
        stats["cancer_lesions"] += int(info.get("cancer", False))
        for a in steps:
            stats[f"action_{a['action_name']}"] += 1
        transitions.extend(steps)
    return transitions, {
        "n_chains": len(chains),
        "reasons": dict(stats),
        "alignment": dict(alignments),
        "n_transitions": len(transitions),
        "n_participants": len({t["participant"] for t in transitions}),
    }
