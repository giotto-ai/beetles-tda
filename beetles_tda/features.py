import giotto.diagrams as diag
import numpy as np
import pandas as pd


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
