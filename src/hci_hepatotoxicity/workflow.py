import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _validate_labels(frame):
    counts = frame.groupby("chemical_id")["label"].nunique()
    if not counts.empty and counts.max() != 1:
        raise ValueError("Labels disagree within a chemical")


def chemical_profiles(frame, feature_columns):
    required = {"chemical_id", "label", *feature_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    _validate_labels(frame)
    aggregations = {column: "mean" for column in feature_columns}
    aggregations["label"] = "first"
    return frame.groupby("chemical_id", as_index=False, sort=True).agg(aggregations)


def split_chemicals(frame, test_fraction, seed):
    _validate_labels(frame)
    chemicals = frame.groupby("chemical_id", as_index=False)["label"].first()
    train_ids, test_ids = train_test_split(chemicals["chemical_id"], test_size=test_fraction, random_state=seed, stratify=chemicals["label"])
    return frame[frame["chemical_id"].isin(train_ids)].reset_index(drop=True), frame[frame["chemical_id"].isin(test_ids)].reset_index(drop=True)


def scale_from_training(training, query):
    scaler = StandardScaler().fit(training)
    return scaler.transform(training), scaler.transform(query), scaler


def rebalance_training(name, features, labels, params, seed):
    selected = dict(params)
    selected.setdefault("random_state", seed)
    name = name.upper().replace("-", "_")
    if name in {"RF", "KNN", "SVM"}:
        return RandomUnderSampler(**selected).fit_resample(features, labels)
    if name in {"CNN", "CNN_BILSTM"}:
        return SMOTE(**selected).fit_resample(features, labels)
    raise ValueError(f"Unknown model: {name}")


def aggregate_probabilities(frame):
    required = {"chemical_id", "label", "probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    _validate_labels(frame)
    return frame.groupby("chemical_id", as_index=False, sort=True).agg(label=("label", "first"), probability=("probability", "mean"), observations=("probability", "size"))
