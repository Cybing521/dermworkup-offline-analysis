MISSING = {"", "nan", "none", "na", "n/a", "null", "0"}


def present(value):
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in MISSING


def truthy(value):
    if not present(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "positive"}


def has_any(text, needles):
    blob = str(text or "").strip().lower()
    return any(n in blob for n in needles)


def parse_participant_visit(raw, participant_number=""):
    text = str(raw or "").strip()
    participant = str(participant_number or "").strip()
    visit = None
    if "_" in text:
        head, tail = text.rsplit("_", 1)
        digits = "".join(ch for ch in tail if ch.isdigit())
        if digits:
            visit = int(digits)
            if not participant:
                participant = head
    if visit is None:
        digits = "".join(ch for ch in text if ch.isdigit())
        visit = int(digits) if digits else 1
    if not participant:
        participant = text or "unknown"
    return participant, visit


def cohort_of(participant):
    name = str(participant).lower()
    if "highrisk" in name or "high_risk" in name:
        return "highrisk"
    if "general" in name:
        return "general"
    return "unknown"
