ACTION_NAMES = ("WatchTBP", "AcquireDerm", "ReferExcise")


def auroc(y, scores):
    n_pos = sum(int(v) for v in y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(scores, y), key=lambda p: p[0])
    rank_sum = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            if pairs[k][1]:
                rank_sum += avg_rank
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision_rank(y, scores):
    """Precision at each positive, averaged. Ties follow input order (stable sort)."""
    n_pos = sum(int(v) for v in y)
    if n_pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
    tp = 0
    ap = 0.0
    for k, i in enumerate(order, 1):
        if y[i]:
            tp += 1
            ap += tp / k
    return ap / n_pos


def _pr_points(y, scores):
    n_pos = sum(int(v) for v in y)
    if n_pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    rec = [0.0]
    prec = [1.0]
    tp = 0
    seen = 0
    i = 0
    n = len(order)
    while i < n:
        j = i
        score = scores[order[i]]
        while j < n and scores[order[j]] == score:
            j += 1
        for k in range(i, j):
            seen += 1
            if y[order[k]]:
                tp += 1
        rec.append(tp / n_pos)
        prec.append(tp / seen)
        i = j
    return rec, prec


def average_precision_grouped(y, scores):
    """sklearn-style AP: sum (R_n - R_{n-1}) * P_n after each unique score."""
    points = _pr_points(y, scores)
    if points is None:
        return None
    rec, prec = points
    ap = 0.0
    for i in range(1, len(rec)):
        ap += (rec[i] - rec[i - 1]) * prec[i]
    return ap


def trapezoidal_pr_auc(y, scores):
    """Trapezoidal area under the grouped PR curve, including (0, 1)."""
    points = _pr_points(y, scores)
    if points is None:
        return None
    rec, prec = points
    area = 0.0
    for i in range(1, len(rec)):
        area += (rec[i] - rec[i - 1]) * (prec[i] + prec[i - 1]) / 2.0
    return area


def auprc(y, scores):
    """Backward-compatible alias of order-dependent rank AP. Prefer grouped AP in new reports."""
    return average_precision_rank(y, scores)


def ranking_metrics(y, scores, prevalence=None):
    n = len(y)
    n_pos = sum(int(v) for v in y)
    prev = prevalence if prevalence is not None else (n_pos / n if n else 0.0)
    return {
        "n": n,
        "n_pos": n_pos,
        "prevalence": prev,
        "auroc": auroc(y, scores),
        "ap_rank": average_precision_rank(y, scores),
        "ap_grouped": average_precision_grouped(y, scores),
        "pr_auc_trapz": trapezoidal_pr_auc(y, scores),
        "auprc": average_precision_rank(y, scores),
    }


def n_unique_scores(scores):
    return len({float(s) for s in scores})


def scores_constant_within(groups, scores):
    by = {}
    for g, s in zip(groups, scores):
        by.setdefault(g, set()).add(float(s))
    return {
        "n_groups": len(by),
        "n_groups_with_multiple_scores": sum(int(len(v) > 1) for v in by.values()),
        "ok": all(len(v) == 1 for v in by.values()),
    }


def per_class(y, pred, names=ACTION_NAMES):
    out = {}
    for c, name in enumerate(names):
        tp = sum(int(p == c and t == c) for p, t in zip(pred, y))
        fp = sum(int(p == c and t != c) for p, t in zip(pred, y))
        fn = sum(int(p != c and t == c) for p, t in zip(pred, y))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": tp + fn,
            "predicted": tp + fp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out
