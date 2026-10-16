import json
from pathlib import Path

from .features import feature_vector, next_features


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_batch(steps, proto, always_visible=False):
    local = dict(proto)
    channel = dict(local["channel"])
    if always_visible:
        channel["mask_missing_dermoscopy"] = False
    local["channel"] = channel
    states, actions, rewards = [], [], []
    next_states, dones = [], []
    cmiss, cover = [], []
    for step in steps:
        feat = feature_vector(step)
        if always_visible:
            feat[0] = 1.0
        states.append(feat)
        actions.append(int(step["action"]))
        rewards.append(float(step["reward"]))
        next_states.append(next_features(step, local) if not always_visible else _visible_next(step, local))
        dones.append(float(step["done"]))
        cmiss.append(float(step["cost_miss"]))
        cover.append(float(step["cost_over"]))
    return states, actions, rewards, next_states, dones, cmiss, cover


def _visible_next(step, proto):
    nxt = next_features(step, proto)
    nxt[0] = 1.0
    return nxt
