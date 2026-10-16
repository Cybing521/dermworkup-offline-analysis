import hashlib
from collections import defaultdict


def patient_split(transitions, proto):
    spec = proto["split"]
    patients = sorted({t["participant"] for t in transitions})
    buckets = {"train": [], "val": [], "test": []}
    for pid in patients:
        digest = hashlib.sha256(f"{spec['seed']}|{pid}".encode()).hexdigest()
        u = int(digest[:8], 16) / 0xFFFFFFFF
        if u < spec["train"]:
            buckets["train"].append(pid)
        elif u < spec["train"] + spec["val"]:
            buckets["val"].append(pid)
        else:
            buckets["test"].append(pid)
    assign = {pid: split for split, ids in buckets.items() for pid in ids}
    grouped = defaultdict(list)
    for step in transitions:
        grouped[assign[step["participant"]]].append(step)
    overlap = (
        set(buckets["train"]) & set(buckets["val"])
        or set(buckets["train"]) & set(buckets["test"])
        or set(buckets["val"]) & set(buckets["test"])
    )
    if overlap:
        raise RuntimeError(f"patient leak: {sorted(overlap)[:5]}")
    return dict(grouped), {k: len(v) for k, v in buckets.items()}
