from collections import Counter, defaultdict

from .features import feature_vector
from .metrics import ACTION_NAMES, auprc, auroc, n_unique_scores, per_class, ranking_metrics
from .trajectories import ACQUIRE


def attach_prev_derm(steps):
    by = defaultdict(list)
    for i, step in enumerate(steps):
        by[step["lesion_id"]].append(i)
    for idxs in by.values():
        idxs.sort(key=lambda i: (steps[i].get("visit") or 0, str(steps[i].get("tile_id") or "")))
        prev = 0
        for i in idxs:
            steps[i]["prev_derm_visible"] = prev
            prev = int(steps[i].get("derm_visible") or 0)
    return steps


def history_vector(step, ablate=None):
    vec = feature_vector(step)
    vec[0] = float(step.get("prev_derm_visible") or 0)
    if ablate == "no_prev":
        vec[0] = 0.0
    elif ablate == "no_visit":
        vec[1] = 0.0
    elif ablate == "visit_only":
        out = [0.0] * len(vec)
        out[1] = vec[1]
        return out
    elif ablate == "no_location":
        for i in range(7, len(vec)):
            vec[i] = 0.0
    return vec


def protocol_consistency(steps):
    attach_prev_derm(steps)
    n = len(steps)
    acquire_mismatch = 0
    for step in steps:
        rule = int(not step.get("prev_derm_visible") and step.get("derm_visible") and step["action"] != 2)
        acquire_mismatch += int(rule != int(step["action"] == ACQUIRE))
    lesions = {}
    for step in steps:
        rec = lesions.setdefault(
            step["lesion_id"],
            {"cancer": 0, "excise": 0, "alignment": step.get("alignment") or "none"},
        )
        rec["cancer"] = max(rec["cancer"], int(step.get("cancer") or 0))
        rec["excise"] += int(step["action"] == 2)
    n_lesion = len(lesions)
    n_cancer = sum(v["cancer"] for v in lesions.values())
    n_excise = sum(1 for v in lesions.values() if v["excise"])
    n_align = sum(1 for v in lesions.values() if v["alignment"] in {"visit", "lesion_level"})
    return {
        "n_transitions": n,
        "n_lesions": n_lesion,
        "cancer_lesions": n_cancer,
        "excise_lesions": n_excise,
        "aligned_lesions": n_align,
        "acquire_rule_mismatch": acquire_mismatch,
        "acquire_rule_match_rate": 1.0 - acquire_mismatch / max(n, 1),
        "actions": dict(Counter(ACTION_NAMES[s["action"]] for s in steps)),
    }


def onset_labels(steps):
    attach_prev_derm(steps)
    y = [int(s["action"] == ACQUIRE) for s in steps]
    return y


def onset_heuristics(steps):
    attach_prev_derm(steps)
    loc = [str(s.get("location") or "").lower() for s in steps]
    visit = [-float(s.get("visit") or 0) for s in steps]
    head = [1.0 if ("head" in t or "neck" in t) else 0.0 for t in loc]
    early_head = [v + 0.5 * h for v, h in zip(visit, head)]
    return {
        "neg_visit": visit,
        "head_neck": head,
        "neg_visit_plus_head": early_head,
    }


def score_binary(y, scores, prevalence):
    out = ranking_metrics(y, scores, prevalence=prevalence)
    return {
        "auroc": out["auroc"],
        "auprc": out["ap_rank"],
        "ap_rank": out["ap_rank"],
        "ap_grouped": out["ap_grouped"],
        "pr_auc_trapz": out["pr_auc_trapz"],
        "prevalence": prevalence,
        "n": out["n"],
        "n_pos": out["n_pos"],
        "n_unique_scores": n_unique_scores(scores),
    }


def majority_pred(n, action=0):
    return [action] * n


def train_onset_scorer(train_steps, hidden=128, steps=4000, batch_size=256, lr=0.001, seed=0, ablate=None):
    import random

    import torch
    from torch import nn

    from .features import feature_dim

    attach_prev_derm(train_steps)
    y = [int(s["action"] == ACQUIRE) for s in train_steps]
    n_pos = max(sum(y), 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight = torch.tensor([(len(y) - n_pos) / n_pos], device=device)
    net = nn.Sequential(
        nn.Linear(feature_dim(), hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    random.seed(seed)
    torch.manual_seed(seed)
    xs = torch.tensor([history_vector(s, ablate=ablate) for s in train_steps], dtype=torch.float32, device=device)
    ys = torch.tensor([[v] for v in y], dtype=torch.float32, device=device)
    n = xs.size(0)
    net.train()
    for _ in range(steps):
        pick = torch.randint(0, n, (min(batch_size, n),), device=device)
        loss = nn.functional.binary_cross_entropy_with_logits(net(xs[pick]), ys[pick], pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net


def onset_scores(net, steps, ablate=None):
    import torch

    attach_prev_derm(steps)
    device = next(net.parameters()).device
    with torch.no_grad():
        x = torch.tensor([history_vector(s, ablate=ablate) for s in steps], dtype=torch.float32, device=device)
        return torch.sigmoid(net(x)).squeeze(1).cpu().tolist()
