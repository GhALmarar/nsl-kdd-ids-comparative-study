"""
preprocessing.py — Phase 2: NSL-KDD preprocessing pipeline
===========================================================
MSc AI dissertation — Ghanem Almarar
"Machine Learning-Based Network Intrusion Detection: A Comparative Analysis of
Supervised Algorithms Using the NSL-KDD Dataset"

This module implements the FIXED preprocessing pipeline that is applied identically
before every classifier (RF, SVM, DT). Keeping this constant is the core of the
study design: if the preprocessing is the same, any performance difference is
attributable to the algorithm.

Pipeline (in order — do not reorder):
  1. load KDDTrain+ / KDDTest+ with the 43 official column names
  2. drop `difficulty`
  3. map 39 raw attack labels -> 5 classes (Normal, DoS, Probe, R2L, U2R)
  4. one-hot encode protocol_type / service / flag
  5. align train & test columns (unseen categories -> 0)
  6. LabelEncoder on y
  7. stratified 80/20 split of KDDTrain+  (random_state=42)
  8. MinMaxScaler  — fit on the 80 % train ONLY
  9. SMOTE         — on the 80 % train ONLY  (random_state=42)
 10. RandomForest feature importance -> keep top 20 features
 11. save arrays to data/processed/ and fitted objects to models/

Run:  python -m src.preprocessing
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "results" / "tables"

RANDOM_STATE = 42          # single seed used everywhere for reproducibility
TEST_SIZE = 0.20           # 80/20 split as stated in the proposal
TOP_K_FEATURES = 20        # RF importance -> top 20, as stated in the proposal

# The raw files have NO header row. These are the 41 NSL-KDD features
# (Tavallaee et al., 2009) plus the class label and the difficulty score.
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
assert len(COLUMN_NAMES) == 43

CATEGORICAL = ["protocol_type", "service", "flag"]   # the 3 nominal features

# Standard NSL-KDD taxonomy: every raw label -> one of 4 attack families.
# KDDTest+ contains attack types that never appear in KDDTrain+ (e.g. apache2,
# mscan, snmpguess) — they are included here so the official test set maps cleanly.
ATTACK_MAP = {
    "normal": "Normal",
    # Denial of Service
    **{a: "DoS" for a in ["back", "land", "neptune", "pod", "smurf", "teardrop",
                          "apache2", "udpstorm", "processtable", "worm", "mailbomb"]},
    # Probe / surveillance
    **{a: "Probe" for a in ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]},
    # Remote-to-Local
    **{a: "R2L" for a in ["guess_passwd", "ftp_write", "imap", "phf", "multihop",
                          "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
                          "snmpguess", "snmpgetattack", "httptunnel", "sendmail",
                          "named"]},
    # User-to-Root
    **{a: "U2R" for a in ["buffer_overflow", "loadmodule", "rootkit", "perl",
                          "sqlattack", "xterm", "ps"]},
}
CLASS_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]


# --------------------------------------------------------------------------- #
# Step functions (each one is small, named, and testable)
# --------------------------------------------------------------------------- #
def load_raw(name: str) -> pd.DataFrame:
    """Step 1 — read a raw NSL-KDD file and attach the 43 column names."""
    path = RAW_DIR / name
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    print(f"[load] {name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def drop_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2 — `difficulty` is a per-record metadata score, not a traffic feature."""
    return df.drop(columns=["difficulty"])


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3 — collapse the raw label column into the 5-class target."""
    unknown = set(df["label"].unique()) - set(ATTACK_MAP)
    if unknown:
        raise ValueError(f"Unmapped attack labels found: {sorted(unknown)}")
    df = df.copy()
    df["attack_class"] = df["label"].map(ATTACK_MAP)
    return df.drop(columns=["label"])


def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4 — one-hot encode the 3 nominal features (dtype int, not bool)."""
    return pd.get_dummies(df, columns=CATEGORICAL, dtype=np.uint8)


def align_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 5 — give train & test the same columns in the same order.

    KDDTest+ contains `service` values unseen in training (and vice-versa).
    Columns missing on one side are added and filled with 0. The column order
    of *train* is the reference.
    """
    train, test = train.align(test, join="outer", axis=1, fill_value=0)
    # keep attack_class last and preserve a deterministic order
    cols = [c for c in train.columns if c != "attack_class"] + ["attack_class"]
    return train[cols], test[cols]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["attack_class"]), df["attack_class"]


def summarise_classes(y, title: str) -> pd.Series:
    counts = pd.Series(y).value_counts().reindex(CLASS_ORDER)
    print(f"\n[{title}]")
    print(counts.to_string())
    return counts


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run() -> dict:
    t0 = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1-3: load, drop difficulty, map labels
    train_df = map_labels(drop_difficulty(load_raw("KDDTrain+.txt")))
    test_df = map_labels(drop_difficulty(load_raw("KDDTest+.txt")))

    class_report = {
        "kddtrain_plus": summarise_classes(train_df["attack_class"], "KDDTrain+ class counts").to_dict(),
        "kddtest_plus": summarise_classes(test_df["attack_class"], "KDDTest+ class counts").to_dict(),
    }

    # 4-5: one-hot + align
    train_df, test_df = align_columns(one_hot(train_df), one_hot(test_df))
    print(f"\n[one-hot] feature columns after encoding: {train_df.shape[1] - 1}")

    X_all, y_all = split_xy(train_df)
    X_official, y_official_raw = split_xy(test_df)

    # 6: label encoding (fit on the 5 class names, fixed order for readability)
    le = LabelEncoder().fit(CLASS_ORDER)
    y_all_enc = le.transform(y_all)
    y_official = le.transform(y_official_raw)
    print(f"[label-encode] classes -> {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 7: stratified 80/20 split of KDDTrain+
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X_all, y_all_enc, test_size=TEST_SIZE, stratify=y_all_enc, random_state=RANDOM_STATE
    )
    print(f"[split] train {X_tr.shape[0]:,} | holdout {X_ho.shape[0]:,} | official test {X_official.shape[0]:,}")

    # 8: MinMax scaling — fit on the 80 % train only (no leakage)
    scaler = MinMaxScaler().fit(X_tr)
    feature_names = list(X_tr.columns)
    X_tr_s = pd.DataFrame(scaler.transform(X_tr), columns=feature_names)
    X_ho_s = pd.DataFrame(scaler.transform(X_ho), columns=feature_names)
    X_of_s = pd.DataFrame(scaler.transform(X_official), columns=feature_names)

    # 9: SMOTE — on the 80 % train only
    before = summarise_classes(le.inverse_transform(y_tr), "Train (80 %) BEFORE SMOTE")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_s, y_tr)
    after = summarise_classes(le.inverse_transform(y_tr_bal), "Train (80 %) AFTER SMOTE")
    class_report["train80_before_smote"] = before.to_dict()
    class_report["train80_after_smote"] = after.to_dict()

    # 10: RF feature importance -> top 20
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr_bal, y_tr_bal)
    importances = (
        pd.Series(rf.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
    )
    top_features = importances.head(TOP_K_FEATURES).index.tolist()
    print(f"\n[feature-selection] top {TOP_K_FEATURES} features:")
    print(importances.head(TOP_K_FEATURES).to_string())

    X_train = X_tr_bal[top_features].to_numpy(dtype=np.float32)
    X_holdout = X_ho_s[top_features].to_numpy(dtype=np.float32)
    X_test_official = X_of_s[top_features].to_numpy(dtype=np.float32)

    # 11: persist
    np.savez_compressed(
        PROCESSED_DIR / "nslkdd_processed.npz",
        X_train=X_train, y_train=y_tr_bal,
        X_holdout=X_holdout, y_holdout=y_ho,
        X_test_official=X_test_official, y_test_official=y_official,
    )
    joblib.dump(scaler, MODELS_DIR / "minmax_scaler.pkl")
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
    joblib.dump(top_features, MODELS_DIR / "top20_features.pkl")
    importances.to_csv(TABLES_DIR / "rf_feature_importance_all.csv", header=["importance"])
    importances.head(TOP_K_FEATURES).to_csv(TABLES_DIR / "rf_feature_importance_top20.csv", header=["importance"])
    with open(PROCESSED_DIR / "class_distribution.json", "w") as fh:
        json.dump(class_report, fh, indent=2)
    with open(PROCESSED_DIR / "pipeline_meta.json", "w") as fh:
        json.dump(
            {
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "n_features_after_onehot": len(feature_names),
                "top_k": TOP_K_FEATURES,
                "top_features": top_features,
                "class_encoding": {c: int(i) for c, i in zip(le.classes_, le.transform(le.classes_))},
                "shapes": {
                    "X_train": X_train.shape, "X_holdout": X_holdout.shape,
                    "X_test_official": X_test_official.shape,
                },
                "elapsed_seconds": round(time.time() - t0, 1),
            },
            fh, indent=2,
        )
    print(f"\n[done] saved to {PROCESSED_DIR} and {MODELS_DIR} in {time.time() - t0:.1f}s")
    return {"top_features": top_features, "class_report": class_report}


if __name__ == "__main__":
    run()
