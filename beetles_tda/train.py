import fire
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score


def fit_and_score_model(X, y_train, y_test, id_train, id_test):
    X_train = X[id_train, :]
    X_test = X[id_test, :]

    # k means
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X_train)

    # score
    print("Homogeneity score (training):", homogeneity_score(y_train, kmeans.labels_))
    print(
        "Homogeneity score (test):", homogeneity_score(y_test, kmeans.predict(X_test)),
    )


if __name__ == "__main__":
    fire.Fire()
