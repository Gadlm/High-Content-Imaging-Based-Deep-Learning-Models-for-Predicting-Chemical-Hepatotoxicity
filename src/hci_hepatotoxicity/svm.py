from sklearn.svm import SVC


def build_svm(params, seed):
    selected = dict(params)
    selected.setdefault("probability", True)
    selected.setdefault("random_state", seed)
    return SVC(**selected)
