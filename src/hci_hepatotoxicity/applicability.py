import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import StandardScaler


class ApplicabilityDomain:
    def __init__(self, support_quantile, inconsistency_quantile):
        self.support_quantile = support_quantile
        self.inconsistency_quantile = inconsistency_quantile

    def fit(self, training, feature_columns):
        if training["chemical_id"].duplicated().any():
            raise ValueError("Training chemicals must be unique")
        self.feature_columns = list(feature_columns)
        values = training[self.feature_columns].to_numpy(dtype=float)
        spread = values.std(axis=0)
        self.keep = spread > 0
        if not self.keep.any():
            raise ValueError("No variable features were found")
        self.scaler = StandardScaler().fit(values[:, self.keep])
        self.training = self.scaler.transform(values[:, self.keep])
        self.labels = training["label"].to_numpy(dtype=int)
        distances = squareform(pdist(self.training)) / np.sqrt(self.training.shape[1])
        positive = distances[distances > 0]
        if positive.size == 0:
            raise ValueError("Training profiles are identical")
        self.bandwidth = float(np.median(positive))
        weights = np.exp(-(distances**2) / (2 * self.bandwidth**2))
        np.fill_diagonal(weights, 0.0)
        totals = weights.sum(axis=1)
        safe_totals = np.maximum(totals, np.finfo(float).tiny)
        disagreement = np.abs(self.labels[:, None] - self.labels[None, :])
        self.training_inconsistency = (weights * disagreement).sum(axis=1) / safe_totals
        support = weights.mean(axis=1)
        interpolated = weights @ self.training_inconsistency / safe_totals
        self.support_threshold = float(np.quantile(support, self.support_quantile))
        self.inconsistency_threshold = float(np.quantile(interpolated, self.inconsistency_quantile))
        return self

    def score(self, query):
        values = query[self.feature_columns].to_numpy(dtype=float)[:, self.keep]
        values = self.scaler.transform(values)
        distances = cdist(values, self.training) / np.sqrt(self.training.shape[1])
        weights = np.exp(-(distances**2) / (2 * self.bandwidth**2))
        totals = weights.sum(axis=1)
        safe_totals = np.maximum(totals, np.finfo(float).tiny)
        support = weights.mean(axis=1)
        inconsistency = weights @ self.training_inconsistency / safe_totals
        within = (support >= self.support_threshold) & (inconsistency <= self.inconsistency_threshold)
        return pd.DataFrame({"chemical_id": query["chemical_id"].to_numpy(), "rho_p": support, "i_y": inconsistency, "within_ad": within})
