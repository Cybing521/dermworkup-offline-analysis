import argparse
import os
import random
from pathlib import Path

import pandas as pd

from .benchmark import onset_heuristics, onset_labels, onset_scores, protocol_consistency, score_binary, train_onset_scorer
from .dataset import encode_batch, load_jsonl, save_jsonl
from .features import feature_dim
from .io import dump_json, experiment as load_exp, protocol as load_proto, rename_columns
from .metrics import per_class
from .models import Agent, act_greedy, behavior_rates, greedy_rates, train_step
from .split import patient_split
from .trajectories import build_transitions, dataset_gate
from .audit_ranking import run_audit
from .verify import id_patterns, onset_block, split_integrity, visit_acquire

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def read_csv(path):
    df = pd.read_csv(path)
    df, mapping = rename_columns(df)
    return df, mapping


def build_from_csv(tile_csv, part_csv, proto):
    tiles, tile_map = read_csv(tile_csv)
    parts = None
    part_map = {}
    if part_csv:
        parts, part_map = read_csv(part_csv)
    transitions, stats = build_transitions(tiles, parts, proto)
    stats["tile_columns"] = tile_map
    stats["participant_columns"] = part_map
    return transitions, stats


def audit_cmd(args):
    proto = load_proto(args.protocol)
    transitions, stats = build_from_csv(args.tiles, args.participants, proto)
    splits, n_patients = patient_split(transitions, proto) if transitions else ({}, {})
    visit_align = sum(1 for t in _first_lesions(transitions) if t["alignment"] == "visit")
    lesion_align = sum(1 for t in _first_lesions(transitions) if t["alignment"] == "lesion_level")
    n_path = visit_align + lesion_align
    report = {
        **stats,
        "split_patients": n_patients,
        "pathology_lesions": n_path,
        "alignment_visit": visit_align,
        "alignment_lesion_level": lesion_align,
        "lesion_level_share": (lesion_align / n_path) if n_path else None,
        "protocol_id": proto.get("protocol_id"),
        "gate": dataset_gate({**stats, "n_transitions": len(transitions)}, proto),
        "behavior": behavior_rates(transitions),
    }
    out = Path(args.out)
    dump_json(out, report)
    if args.trajectories:
        save_jsonl(args.trajectories, transitions)
        for name, rows in splits.items():
            save_jsonl(Path(args.trajectories).with_name(f"{name}.jsonl"), rows)
    print(out)
    print(report["gate"], report.get("reasons"))


def _first_lesions(transitions):
    seen = set()
    out = []
    for step in transitions:
        if step["lesion_id"] in seen:
            continue
        seen.add(step["lesion_id"])
        out.append(step)
    return out


def _batch(steps, proto, always_visible, size):
    if not steps:
        raise RuntimeError("empty split")
    pick = [random.choice(steps) for _ in range(size)]
    return encode_batch(pick, proto, always_visible=always_visible)


def train_cmd(args):
    if not args.smoke and os.environ.get("RDA_REMOTE") != "1":
        raise SystemExit("Refuse to train on this machine. Run on remote with RDA_REMOTE=1, or pass --smoke on synthetic data.")
    cfg = load_exp(args.experiment)
    proto = load_proto(args.protocol)
    train_steps = load_jsonl(args.train)
    val_steps = load_jsonl(args.val) if args.val else train_steps
    random.seed(cfg["seed"])
    always = args.method == "rda_no_channel_mask"
    torch = __import__("torch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(feature_dim(), cfg["hidden"], args.method, always_visible=always).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg["lr"])
    print("device", device)
    dual = {"lam_miss": 0.0 if args.method == "rda_no_cost" else cfg["lambda_init"], "lam_over": 0.0 if args.method == "rda_no_cost" else cfg["lambda_init"]}
    n_steps = cfg["smoke_steps"] if args.smoke else cfg["train_steps"]
    if n_steps <= 0:
        raise SystemExit("train_steps is 0; set it in experiment.json or use --smoke")
    logs = []
    for i in range(n_steps):
        batch = _batch(train_steps, proto, always, cfg["batch_size"] if not args.smoke else min(16, len(train_steps)))
        logs.append(train_step(agent, opt, batch, cfg, dual))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{args.method}.pt"
    torch.save({"state": agent.state_dict(), "kind": args.method, "always_visible": always}, ckpt)
    metrics = {
        "method": args.method,
        "smoke": bool(args.smoke),
        "steps": n_steps,
        "device": str(device),
        "last": logs[-1],
        "train_behavior": behavior_rates(train_steps),
        "val": greedy_rates(agent, val_steps, proto, always_visible=always),
        "publication_evidence": False,
    }
    dump_json(out_dir / f"{args.method}_metrics.json", metrics)
    print(ckpt)
    print("publication_evidence=false")


def eval_cmd(args):
    proto = load_proto(args.protocol)
    steps = load_jsonl(args.data)
    dump_json(args.out, {"behavior": behavior_rates(steps), "n": len(steps)})
    print(args.out)


def eval_policy_cmd(args):
    torch = __import__("torch")
    proto = load_proto(args.protocol)
    cfg = load_exp(args.experiment)
    steps = load_jsonl(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)
    kind = args.method or ckpt.get("kind") or Path(args.ckpt).stem
    always = bool(ckpt.get("always_visible", kind == "rda_no_channel_mask"))
    agent = Agent(feature_dim(), cfg["hidden"], kind, always_visible=always).to(device)
    agent.load_state_dict(ckpt["state"])
    agent.eval()
    rates = greedy_rates(agent, steps, proto, always_visible=always)
    report = {
        "method": kind,
        "ckpt": str(args.ckpt),
        "data": str(args.data),
        "split": args.split,
        "device": str(device),
        "logged": behavior_rates(steps),
        "policy": rates,
        "publication_evidence": False,
    }
    dump_json(args.out, report)
    print(args.out)


def _load_agent(ckpt_path, cfg, device, method=""):
    torch = __import__("torch")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    kind = method or ckpt.get("kind") or Path(ckpt_path).stem
    always = bool(ckpt.get("always_visible", kind == "rda_no_channel_mask"))
    agent = Agent(feature_dim(), cfg["hidden"], kind, always_visible=always).to(device)
    agent.load_state_dict(ckpt["state"])
    agent.eval()
    return agent, kind, always


def _policy_class_report(agent, steps, proto, always):
    torch = __import__("torch")
    s, a, *_ = encode_batch(steps, proto, always_visible=always)
    device = next(agent.parameters()).device
    pred = act_greedy(agent, torch.tensor(s, dtype=torch.float32, device=device)).cpu().tolist()
    from .models import protocol_rates

    return {"per_class": per_class(a, pred), **protocol_rates(steps, pred, proto)}


def benchmark_cmd(args):
    if not args.smoke and os.environ.get("RDA_REMOTE") != "1":
        raise SystemExit("Refuse to run full benchmark on this machine. Use RDA_REMOTE=1 or --smoke.")
    proto = load_proto(args.protocol)
    cfg = load_exp(args.experiment)
    train_steps = load_jsonl(args.train)
    val_steps = load_jsonl(args.val)
    test_steps = load_jsonl(args.test) if args.test else []
    splits = {"train": train_steps, "val": val_steps}
    if test_steps:
        splits["test"] = test_steps
    report = {
        "protocol_id": proto.get("protocol_id"),
        "eval_kind": "dermworkup_offline_benchmark",
        "publication_evidence": False,
        "consistency": {name: protocol_consistency(rows) for name, rows in splits.items()},
        "onset": {},
        "policies": {},
    }
    n_steps = 32 if args.smoke else args.onset_steps
    print("train onset", flush=True)
    net = train_onset_scorer(train_steps, hidden=cfg["hidden"], steps=n_steps, seed=cfg["seed"])
    for name, rows in splits.items():
        print("onset", name, flush=True)
        y = onset_labels(rows)
        split_prev = sum(y) / max(len(y), 1)
        heur = onset_heuristics(rows)
        mlp = onset_scores(net, rows)
        block = {
            "majority": score_binary(y, [0.0] * len(y), split_prev),
            "mlp_history": score_binary(y, mlp, split_prev),
        }
        for hname, scores in heur.items():
            block[hname] = score_binary(y, scores, split_prev)
        prev0 = [i for i, s in enumerate(rows) if not s.get("prev_derm_visible")]
        if prev0:
            y0 = [y[i] for i in prev0]
            p0 = sum(y0) / len(y0)
            block["prev0_mlp_history"] = score_binary(y0, [mlp[i] for i in prev0], p0)
            block["prev0_neg_visit"] = score_binary(y0, [heur["neg_visit"][i] for i in prev0], p0)
        report["onset"][name] = block
    print("consistency and onset done", flush=True)
    for split, rows in splits.items():
        y = [int(s["action"]) for s in rows]
        report["policies"].setdefault("majority_watch", {})[split] = {"per_class": per_class(y, [0] * len(y))}
    if args.ckpt_dir:
        torch = __import__("torch")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        methods = list(cfg["methods"]) + list(cfg.get("ablations") or [])
        for method in methods:
            ckpt = Path(args.ckpt_dir) / f"train_{method}" / f"{method}.pt"
            if not ckpt.exists() or method == "behavior":
                continue
            agent, kind, always = _load_agent(ckpt, cfg, device, method)
            print("policy", kind, flush=True)
            report["policies"][kind] = {
                split: _policy_class_report(agent, rows, proto, always) for split, rows in (("val", val_steps), ("test", test_steps)) if rows
            }
    dump_json(args.out, report)
    print(args.out)


def _scan_tile_ids(path):
    df = pd.read_csv(path, nrows=0)
    cols = list(df.columns)
    use = [c for c in cols if "id" in c.lower() or "dermoscopic" in c.lower()][:6]
    if not use:
        return {"columns": cols, "note": "no id-like columns"}
    chunk = pd.read_csv(path, usecols=use, dtype=str)
    out = {"n_rows": int(len(chunk)), "columns": cols}
    for col in use:
        series = chunk[col].fillna("").astype(str)
        out[col] = {
            "nonempty": int((series.str.strip() != "").sum()),
            "contains_isic": int(series.str.lower().str.contains("isic", na=False).sum()),
        }
    return out


def verify_cmd(args):
    if not args.smoke and os.environ.get("RDA_REMOTE") != "1":
        raise SystemExit("Refuse to run full verify on this machine. Use RDA_REMOTE=1 or --smoke.")
    cfg = load_exp(args.experiment)
    splits = {
        "train": load_jsonl(args.train),
        "val": load_jsonl(args.val),
    }
    if args.test:
        splits["test"] = load_jsonl(args.test)
    n_steps = 32 if args.smoke else args.onset_steps
    hidden = cfg["hidden"]
    report = {
        "eval_kind": "dermworkup_offline_verify",
        "publication_evidence": False,
        "split_integrity": split_integrity(splits),
        "id_patterns": {name: id_patterns(rows) for name, rows in splits.items()},
        "visit_acquire_prev0": {name: visit_acquire(rows) for name, rows in splits.items()},
        "onset_seeds": {},
        "onset_ablate": {},
    }
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    eval_splits = {k: v for k, v in splits.items() if k != "train"}
    for seed in seeds:
        print("seed", seed, flush=True)
        net = train_onset_scorer(splits["train"], hidden=hidden, steps=n_steps, seed=seed)
        report["onset_seeds"][str(seed)] = {name: onset_block(net, rows) for name, rows in eval_splits.items()}
    for ablate in ("no_visit", "no_prev", "visit_only"):
        print("ablate", ablate, flush=True)
        net = train_onset_scorer(splits["train"], hidden=hidden, steps=n_steps, seed=cfg["seed"], ablate=ablate)
        report["onset_ablate"][ablate] = {name: onset_block(net, rows, ablate=ablate) for name, rows in eval_splits.items()}
    if args.tiles:
        print("tiles", flush=True)
        report["tile_id_scan"] = _scan_tile_ids(args.tiles)
    dump_json(args.out, report)
    print(args.out)


def audit_ranking_cmd(args):
    if not args.smoke and os.environ.get("RDA_REMOTE") != "1":
        raise SystemExit("Refuse to run ranking audit on this machine. Use RDA_REMOTE=1 or --smoke.")
    splits = {
        "train": load_jsonl(args.train),
        "val": load_jsonl(args.val),
    }
    if args.test:
        splits["test"] = load_jsonl(args.test)
    cfg = load_exp(args.experiment)
    n_steps = 32 if args.smoke else args.onset_steps
    report = run_audit(
        splits,
        train_steps=splits["train"],
        mlp=not args.no_mlp,
        onset_steps=n_steps,
        seed=cfg["seed"],
        hidden=cfg["hidden"],
    )
    dump_json(args.out, report)
    print(args.out)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit")
    a.add_argument("--tiles", required=True)
    a.add_argument("--participants")
    a.add_argument("--protocol", default=str(ROOT / "configs/protocol.json"))
    a.add_argument("--out", default=str(PROJECT / "05_验证记录/trajectory_audit.json"))
    a.add_argument("--trajectories", default="")
    a.set_defaults(func=audit_cmd)

    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--val")
    t.add_argument("--method", default="rda")
    t.add_argument("--experiment", default=str(ROOT / "configs/experiment.json"))
    t.add_argument("--protocol", default=str(ROOT / "configs/protocol.json"))
    t.add_argument("--out", default=str(PROJECT / "05_验证记录/smoke"))
    t.add_argument("--smoke", action="store_true")
    t.set_defaults(func=train_cmd)

    e = sub.add_parser("eval")
    e.add_argument("--data", required=True)
    e.add_argument("--protocol", default=str(ROOT / "configs/protocol.json"))
    e.add_argument("--out", default=str(PROJECT / "05_验证记录/eval.json"))
    e.set_defaults(func=eval_cmd)

    ep = sub.add_parser("eval-policy")
    ep.add_argument("--ckpt", required=True)
    ep.add_argument("--data", required=True)
    ep.add_argument("--out", required=True)
    ep.add_argument("--method", default="")
    ep.add_argument("--split", default="")
    ep.add_argument("--experiment", default=str(ROOT / "configs/experiment.json"))
    ep.add_argument("--protocol", default=str(ROOT / "configs/protocol.json"))
    ep.set_defaults(func=eval_policy_cmd)

    b = sub.add_parser("benchmark")
    b.add_argument("--train", required=True)
    b.add_argument("--val", required=True)
    b.add_argument("--test", default="")
    b.add_argument("--ckpt-dir", default="")
    b.add_argument("--onset-steps", type=int, default=4000)
    b.add_argument("--experiment", default=str(ROOT / "configs/experiment.json"))
    b.add_argument("--protocol", default=str(ROOT / "configs/protocol.json"))
    b.add_argument("--out", required=True)
    b.add_argument("--smoke", action="store_true")
    b.set_defaults(func=benchmark_cmd)

    v = sub.add_parser("verify")
    v.add_argument("--train", required=True)
    v.add_argument("--val", required=True)
    v.add_argument("--test", default="")
    v.add_argument("--tiles", default="")
    v.add_argument("--seeds", default="0,1,2")
    v.add_argument("--onset-steps", type=int, default=4000)
    v.add_argument("--experiment", default=str(ROOT / "configs/experiment.json"))
    v.add_argument("--out", required=True)
    v.add_argument("--smoke", action="store_true")
    v.set_defaults(func=verify_cmd)

    ar = sub.add_parser("audit-ranking")
    ar.add_argument("--train", required=True)
    ar.add_argument("--val", required=True)
    ar.add_argument("--test", default="")
    ar.add_argument("--onset-steps", type=int, default=4000)
    ar.add_argument("--experiment", default=str(ROOT / "configs/experiment.json"))
    ar.add_argument("--out", required=True)
    ar.add_argument("--no-mlp", action="store_true")
    ar.add_argument("--smoke", action="store_true")
    ar.set_defaults(func=audit_ranking_cmd)

    args = p.parse_args(argv)
    if args.cmd == "audit" and not args.trajectories:
        args.trajectories = ""
    args.func(args)


if __name__ == "__main__":
    main()
