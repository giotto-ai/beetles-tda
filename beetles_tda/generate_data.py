import pandas as pd
import numpy as np
import pickle
import fire
from beetles_tda.beetle import BeetlePopulation
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
import gtda.time_series as ts
import gtda.homology as hl
from gtda.pipeline import Pipeline
from beetles_tda.features import (
    get_amplitude,
    get_max_lifetime,
    get_mean_lifetime,
    get_n_rel_holes,
)
import gtda.diagrams as diag
from fastprogress import master_bar, progress_bar


def generate_population_data(beetle_population, data, n_steps, n_series):
    for itr in range(n_series):
        l_0, p_0, a_0 = 98 * np.random.rand(3) + 2
        std_noise = 0.01 * np.random.randn(3) + 0.01
        data = pd.concat(
            [
                data,
                beetle_population.simulate(
                    l_0=l_0, p_0=p_0, a_0=a_0, std_noise=std_noise, n_steps=n_steps
                ),
            ]
        )
    return data


def fixed_noise(n_steps, n_series, args_stable, args_aperiodic):
    # fix seed for reproducibility
    np.random.seed(42)

    pop_stable = BeetlePopulation(**args_stable)
    pop_aperiodic = BeetlePopulation(**args_aperiodic)

    # Define data frame for simulated data. The columns contain
    # the number of current larvae, pupae and adults.
    data = pd.DataFrame(columns=["larvae", "pupae", "adults"])

    # simulate stable case
    data_stable = generate_population_data(pop_stable, data, n_steps, n_series)

    # simulate aperiodic case
    data_stable_aperiodic = generate_population_data(
        pop_aperiodic, data_stable, n_steps, n_series
    )

    # add an id for each time series (the id is only unique within a case)
    data_stable_aperiodic["series_id"] = 2 * sorted(
        (n_steps + 1) * list(range(n_series))
    )
    # add a label of the time series indicated the case it belongs to,
    # i.e. either stable or aperiodic
    data_stable_aperiodic["type"] = n_series * (n_steps + 1) * ["stable"] + n_series * (
        n_steps + 1
    ) * ["aperiodic"]

    with open("data/population_data.pkl", "wb") as file:
        pickle.dump(data_stable_aperiodic, file)


def simulate_data(noise, std, n_steps, n_series, args_stable, args_aperiodic):
    # fix seed for reproducibility
    np.random.seed(42)

    # instantiate instances of beetle class
    pop_stable = BeetlePopulation(**args_stable)
    pop_aperiodic = BeetlePopulation(**args_aperiodic)

    # Define data frame for simulated data. The columns contain
    # the number of current larvae, pupae and adults.
    data = pd.DataFrame(columns=["larvae", "pupae", "adults"])

    # simulate stable case
    for itr in range(n_series):
        l_0, p_0, a_0 = 98 * np.random.rand(3) + 2

        std_noise = std * np.random.randn(3) + noise

        data = pd.concat(
            [
                data,
                pop_stable.simulate(
                    l_0=l_0, p_0=p_0, a_0=a_0, std_noise=std_noise, n_steps=n_steps
                ),
            ]
        )

    # simulate aperiodic case
    for itr in range(n_series):
        l_0, p_0, a_0 = 98 * np.random.rand(3) + 2
        std_noise = 0.1 * np.random.randn(3) + noise
        data = pd.concat(
            [
                data,
                pop_aperiodic.simulate(
                    l_0=l_0, p_0=p_0, a_0=a_0, std_noise=std_noise, n_steps=n_steps
                ),
            ]
        )

    # add an id for each time series (the id is only unique within a case)
    data["series_id"] = 2 * sorted((n_steps + 1) * list(range(n_series)))
    # add a label of the time series indicated the case it belongs to,
    # i.e. either stable or aperiodic
    data["type"] = n_series * (n_steps + 1) * ["stable"] + n_series * (n_steps + 1) * [
        "aperiodic"
    ]

    return data


def varying_noise(n_steps, n_series, args_stable, args_aperiodic):
    # noise parameters
    min_noise = 0.0
    max_noise = 2.1
    step_size = 0.1
    std = 0.1

    parameters_type = "fixed"
    embedding_dimension = 2
    embedding_time_delay = 3
    n_jobs = 1

    window_width = 121 - ((embedding_dimension - 1) * embedding_time_delay + 1)
    # window_stride = 1

    metric = "euclidean"
    max_edge_length = 10
    homology_dimensions = [0, 1]

    epsilon = 0.0

    steps = [
        (
            "embedding",
            ts.TakensEmbedding(
                parameters_type=parameters_type,
                dimension=embedding_dimension,
                time_delay=embedding_time_delay,
                n_jobs=n_jobs,
            ),
        ),
        ("window", ts.SlidingWindow(width=window_width, stride=1)),
        (
            "diagrams",
            hl.VietorisRipsPersistence(
                metric=metric,
                max_edge_length=max_edge_length,
                homology_dimensions=homology_dimensions,
                n_jobs=n_jobs,
            ),
        ),
        ("diagrams_scaler", diag.Scaler()),
        ("diagrams_filter", diag.Filtering(epsilon=epsilon)),
    ]

    pipeline = Pipeline(steps)

    # maximal number of repetitions per noise level (for confidence intervals)
    max_itr = 5

    # data frames to save performance
    perf_train = pd.DataFrame(
        columns={"Score", "Type", "Mean Standard Deviation of Noise"}
    )
    perf_test = pd.DataFrame(
        columns={"Score", "Type", "Mean Standard Deviation of Noise"}
    )

    mb = master_bar(np.arange(min_noise, max_noise, step_size))
    for noise in mb:
        for _ in progress_bar(range(max_itr), parent=mb):
            mb.child.comment = "Repetitions per noise level"
            data = simulate_data(
                noise, std, n_steps, n_series, args_stable, args_aperiodic
            )
            # group data by type and series id
            grouped_data = data.groupby(["type", "series_id"])

            y_true = np.repeat([1, 0], n_series)
            id_train, id_test, y_train, y_test = train_test_split(
                range(2 * n_series), y_true, train_size=0.7, random_state=0
            )

            # classical k-means ###########################################################
            X = data["adults"].values.reshape((2 * n_series, -1))
            # train/test data
            X_train = X[id_train, :]
            X_test = X[id_test, :]

            # k means
            kmeans = KMeans(n_clusters=2, random_state=0)
            kmeans.fit(X_train)

            perf_train = perf_train.append(
                {
                    "Score": homogeneity_score(y_train, kmeans.labels_),
                    "Type": "Classic",
                    "Mean Standard Deviation of Noise": noise,
                },
                ignore_index=True,
            )

            perf_test = perf_test.append(
                {
                    "Score": homogeneity_score(y_test, kmeans.predict(X_test)),
                    "Type": "Classic",
                    "Mean Standard Deviation of Noise": noise,
                },
                ignore_index=True,
            )

            # threshold to determine whether a hole is relevant or not
            frac = 0.7

            # TDA k-means
            features = []
            for name, _ in grouped_data:
                X_filtered = pipeline.fit_transform(
                    grouped_data.get_group(name)["adults"].values
                )
                n_windows, n_points, _ = X_filtered.shape
                features.append(
                    get_mean_lifetime(X_filtered, n_windows, n_points)
                    + get_n_rel_holes(X_filtered, n_windows, n_points, frac=frac)
                    + get_n_rel_holes(X_filtered, n_windows, n_points, frac=0.0)
                    + get_max_lifetime(X_filtered, n_windows, n_points)
                    + get_amplitude(X_filtered)
                )

            # define data matrix for k-means
            X_tda = np.array(features)

            X_tda_train = X_tda[id_train, :]
            X_tda_test = X_tda[id_test, :]

            # k means
            kmeans_tda = KMeans(n_clusters=2, random_state=0)
            kmeans_tda.fit(X_tda_train)

            perf_train = perf_train.append(
                {
                    "Score": homogeneity_score(y_train, kmeans_tda.labels_),
                    "Type": "TDA",
                    "Mean Standard Deviation of Noise": noise,
                },
                ignore_index=True,
            )

            perf_test = perf_test.append(
                {
                    "Score": homogeneity_score(y_test, kmeans_tda.predict(X_tda_test)),
                    "Type": "TDA",
                    "Mean Standard Deviation of Noise": noise,
                },
                ignore_index=True,
            )
        mb.first_bar.comment = "Noise level"

    # write performance metrics to disk
    with open("models/performance_metrics_train.pkl", "wb") as file:
        pickle.dump(perf_train, file)

    with open("models/performance_metrics_test.pkl", "wb") as file:
        pickle.dump(perf_test, file)


if __name__ == "__main__":
    fire.Fire()
