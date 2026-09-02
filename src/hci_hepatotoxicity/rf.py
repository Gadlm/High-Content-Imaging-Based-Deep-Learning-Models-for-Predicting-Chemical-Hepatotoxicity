from sklearn.ensemble import RandomForestClassifier


def build_rf(params, seed):
    selected = dict(params)
    selected.setdefault("random_state", seed)
    selected.setdefault("n_jobs", -1)
    return RandomForestClassifier(**selected)
