"""Synthetic UQ-schema CSVs. Not a clinical dataset."""

from pathlib import Path

import pandas as pd

TILE_COLS = {
    "Uniqie_ID": [],
    "Participant_visit": [],
    "Lesion_Simple_Location": [],
    "Diagnosis": [],
    "Histopathologic_Result": [],
    "Corresponding_Dermoscopic Image_ID": [],
    "Corresponding_LesionID for_next_1_visit": [],
    "Corresponding_LesionID for_next_2_visit": [],
    "Corresponding_LesionID for_next_3_visit": [],
    "Corresponding_LesionID for_next_4_visit": [],
    "Corresponding_LesionID for_next_5_visit": [],
    "Corresponding_LesionID for_next_6_visit": [],
}


def _empty_next():
    return {f"Corresponding_LesionID for_next_{k}_visit": "" for k in range(1, 7)}


def write_fixture(dirpath):
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(uid, pid_visit, loc, diag, hist, derm, **nexts):
        row = {
            "Uniqie_ID": uid,
            "Participant_visit": pid_visit,
            "Lesion_Simple_Location": loc,
            "Diagnosis": diag,
            "Histopathologic_Result": hist,
            "Corresponding_Dermoscopic Image_ID": derm,
            **_empty_next(),
            **nexts,
        }
        rows.append(row)

    # General_1: watch -> new dermoscopy -> still watch (benign)
    add("g1_a1", "General_1_1", "Arm", "benign", "no", "", **{"Corresponding_LesionID for_next_1_visit": "g1_a2", "Corresponding_LesionID for_next_2_visit": "g1_a3"})
    add("g1_a2", "General_1_2", "Arm", "benign", "no", "d_g1_a2", **{"Corresponding_LesionID for_next_1_visit": "g1_a3"})
    add("g1_a3", "General_1_3", "Arm", "benign", "no", "d_g1_a3")

    # HighRisk_2 intervention: watch then excise melanoma, histopath only on last visit
    add("h2_b1", "HighRisk_2_1", "Torso back", "melanoma", "no", "", **{"Corresponding_LesionID for_next_1_visit": "h2_b2"})
    add("h2_b2", "HighRisk_2_2", "Torso back", "melanoma", "yes", "d_h2_b2")

    # HighRisk_3 control: two visits, should drop
    add("h3_c1", "HighRisk_3_1", "Leg", "benign", "no", "", **{"Corresponding_LesionID for_next_1_visit": "h3_c2"})
    add("h3_c2", "HighRisk_3_2", "Leg", "benign", "no", "")

    # General_4: lesion-level histopath on all visits, BCC
    add("g4_d1", "General_4_1", "Head and neck", "basal cell carcinoma", "yes", "d_g4_d1", **{"Corresponding_LesionID for_next_1_visit": "g4_d2"})
    add("g4_d2", "General_4_2", "Head and neck", "basal cell carcinoma", "yes", "d_g4_d2")

    # General_5: single visit, drop
    add("g5_e1", "General_5_1", "Arm", "benign", "no", "")

    pd.DataFrame(rows).to_csv(dirpath / "tiles.csv", index=False)
    parts = pd.DataFrame(
        [
            {"Participant_Number": "General_1", "Study_arm": "", "Number_of_naevi": 40, "Total_frecklilng_score": 6, "Family_history_of melanoma": "No", "Participant_cancer_excision": 0},
            {"Participant_Number": "HighRisk_2", "Study_arm": "intervention", "Number_of_naevi": 120, "Total_frecklilng_score": 9, "Family_history_of melanoma": "Yes", "Participant_cancer_excision": 2},
            {"Participant_Number": "HighRisk_3", "Study_arm": "control", "Number_of_naevi": 80, "Total_frecklilng_score": 5, "Family_history_of melanoma": "Yes", "Participant_cancer_excision": 1},
            {"Participant_Number": "General_4", "Study_arm": "", "Number_of_naevi": 15, "Total_frecklilng_score": 3, "Family_history_of melanoma": "No", "Participant_cancer_excision": 0},
            {"Participant_Number": "General_5", "Study_arm": "", "Number_of_naevi": 10, "Total_frecklilng_score": 2, "Family_history_of melanoma": "No", "Participant_cancer_excision": 0},
        ]
    )
    parts.to_csv(dirpath / "participants.csv", index=False)
    return dirpath / "tiles.csv", dirpath / "participants.csv"
