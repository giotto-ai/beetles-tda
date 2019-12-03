import numpy as np
import pandas as pd
import plotly.graph_objs as gobj
from giotto.diagrams._utils import _subdiagrams
import giotto.diagrams as diag
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score


def plot_diagram(diagram, homology_dimensions=None):
    """Plot a single persistence diagram.

    Parameters
    ----------
    diagram : ndarray, shape (n_points, 3)
        The persistence diagram to plot, where the third dimension along axis 1
        contains homology dimensions, and the other two contain (birth, death)
        pairs to be used as coordinates in the two-dimensional plot.

    homology_dimensions : list of int or None, default: ``None``
        Homology dimensions which will appear on the plot. If ``None``, all
        homology dimensions which appear in `diagram` will be plotted.

    """
    if homology_dimensions is None:
        homology_dimensions = np.unique(diagram[:, 2])

    maximum_persistence = np.where(np.isinf(diagram), -np.inf, diagram).max()

    layout = {
        "title": "Persistence diagram",
        "width": 500,
        "height": 500,
        "xaxis1": {
            "title": "Birth",
            "side": "bottom",
            "type": "linear",
            "range": [0, 1.1 * maximum_persistence],
            "ticks": "outside",
            "anchor": "y1",
            "showline": True,
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e",
        },
        "yaxis1": {
            "title": "Death",
            "side": "left",
            "type": "linear",
            "range": [0, 1.1 * maximum_persistence],
            "ticks": "outside",
            "anchor": "x1",
            "showline": True,
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e",
        },
        "plot_bgcolor": "white",
    }

    fig = gobj.Figure(layout=layout)
    fig.update_xaxes(zeroline=True, linewidth=1, linecolor="black", mirror=False)
    fig.update_yaxes(zeroline=True, linewidth=1, linecolor="black", mirror=False)

    fig.add_trace(
        gobj.Scatter(
            x=np.array([-100 * maximum_persistence, 100 * maximum_persistence]),
            y=np.array([-100 * maximum_persistence, 100 * maximum_persistence]),
            mode="lines",
            line=dict(dash="dash", width=1, color="black"),
            showlegend=False,
            hoverinfo="none",
        )
    )

    for i, dimension in enumerate(homology_dimensions):
        name = "H{}".format(int(dimension))
        subdiagram = _subdiagrams(np.asarray([diagram]), [dimension], remove_dim=True)[
            0
        ]
        diff = subdiagram[:, 1] != subdiagram[:, 0]
        subdiagram = subdiagram[diff]
        fig.add_trace(
            gobj.Scatter(
                x=subdiagram[:, 0], y=subdiagram[:, 1], mode="markers", name=name
            )
        )

    fig.show()


def get_mean_lifetime(X, n_windows, n_points):
    return (
        pd.DataFrame(
            np.column_stack(
                (
                    np.repeat(np.arange(n_windows), n_points),
                    X.reshape(n_windows * n_points, -1),
                )
            ),
            columns=["window", "birth", "death", "dim"],
        )
        .astype({"window": "int32", "dim": "int32"})
        .groupby(["window", "dim"])
        .apply(lambda g: (g["death"] - g["birth"]).mean())
        .to_list()
    )


def get_n_rel_holes(X, n_windows, n_points, frac):
    return (
        pd.DataFrame(
            np.column_stack(
                (
                    np.repeat(np.arange(n_windows), n_points),
                    X.reshape(n_windows * n_points, -1),
                )
            ),
            columns=["window", "birth", "death", "dim"],
        )
        .astype({"window": "int32", "dim": "int32"})
        .groupby(["window", "dim"])
        .apply(lambda g: (g["death"] - g["birth"]))
        .groupby(level=[0, 1])
        .apply(lambda x: len(x.where(x >= frac * x.max()).dropna()))
        .to_list()
    )


def get_max_lifetime(X, n_windows, n_points):
    return (
        pd.DataFrame(
            np.column_stack(
                (
                    np.repeat(np.arange(n_windows), n_points),
                    X.reshape(n_windows * n_points, -1),
                )
            ),
            columns=["window", "birth", "death", "dim"],
        )
        .astype({"window": "int32", "dim": "int32"})
        .groupby(["window", "dim"])
        .apply(lambda g: (g["death"] - g["birth"]).max())
        .to_list()
    )


def get_amplitude(X, metric="wasserstein"):
    wasserstein_amplitudes = diag.Amplitude(metric="wasserstein")
    return wasserstein_amplitudes.fit_transform(X).flatten().tolist()


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
