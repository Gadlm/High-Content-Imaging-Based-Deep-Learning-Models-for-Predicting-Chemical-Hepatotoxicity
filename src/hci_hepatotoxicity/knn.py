from sklearn.neighbors import KNeighborsClassifier


def build_knn(params, seed):
    selected = dict(params)
    selected.setdefault("n_jobs", -1)
    return KNeighborsClassifier(**selected)
