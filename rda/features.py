LOCATIONS = [
    "head and neck",
    "arm",
    "leg",
    "torso back",
    "torso front",
    "other",
]


def _loc_index(text):
    blob = str(text or "").strip().lower().replace("&", " and ")
    for i, name in enumerate(LOCATIONS):
        if name in blob:
            return i
    if "head" in blob or "neck" in blob:
        return 0
    if "torso" in blob and "back" in blob:
        return 3
    if "torso" in blob:
        return 4
    return len(LOCATIONS) - 1


def _num(meta, *keys):
    for key in keys:
        val = meta.get(key)
        if val is None or val == "":
            continue
        try:
            return float(str(val).split("-")[0].replace("+", "").split()[0])
        except ValueError:
            continue
    return 0.0


def feature_vector(step):
    meta = step.get("meta") or {}
    loc = [0.0] * len(LOCATIONS)
    loc[_loc_index(step.get("location"))] = 1.0
    derm = float(step.get("derm_visible") or 0)
    highrisk = 1.0 if step.get("cohort") == "highrisk" else 0.0
    vec = [
        derm,
        float(step.get("visit") or 0) / 10.0,
        highrisk,
        _num(meta, "number_of_naevi", "number_of_lesions_with_tile_images") / 500.0,
        _num(meta, "total_frecklilng_score", "total_freckling_score") / 12.0,
        _num(meta, "participant_cancer_excision"),
        1.0 if "yes" in str(meta.get("family_history_of_melanoma") or "").lower() else 0.0,
        *loc,
    ]
    return vec


def feature_dim():
    return 7 + len(LOCATIONS)


def next_features(step, proto):
    vec = feature_vector(step)
    if step.get("done"):
        return [0.0] * len(vec)
    nxt = list(vec)
    # next visit index roughly +1; dermoscopy channel from logged next
    nxt[0] = float(step.get("next_derm_visible") or 0)
    nxt[1] = min(1.0, vec[1] + 0.1)
    if not proto["channel"]["mask_missing_dermoscopy"]:
        nxt[0] = max(nxt[0], vec[0])
    return nxt
