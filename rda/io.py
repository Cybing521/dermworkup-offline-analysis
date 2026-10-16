import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(Path(path).read_text())


def protocol(path=None):
    return load_json(path or ROOT / "configs/protocol.json")


def experiment(path=None):
    return load_json(path or ROOT / "configs/experiment.json")


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def norm_col(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def rename_columns(df):
    mapping = {c: norm_col(c) for c in df.columns}
    return df.rename(columns=mapping), mapping
