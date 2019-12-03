import pandas as pd
import numpy as np
import pickle
import fire


class BeetlePopulation:
    def __init__(self, b, c_ea, c_el, c_pa, mu_a, mu_l):
        self.b = b
        self.c_ea = c_ea
        self.c_el = c_el
        self.c_pa = c_pa
        self.mu_a = mu_a
        self.mu_l = mu_l

    def get_population(self, l, p, a, std_noise):
        return [
            self.b
            * a
            * np.exp(-self.c_ea * a - self.c_el * l + std_noise[0] * np.random.randn()),
            l * (1 - self.mu_l) * np.exp(std_noise[1] * np.random.randn()),
            (p * np.exp(-self.c_pa * a) + a * (1 - self.mu_a))
            * np.exp(std_noise[2] * np.random.randn()),
        ]

    def simulate(self, l_0, p_0, a_0, std_noise, n_steps):
        data = [[l_0, p_0, a_0]]
        for step in range(n_steps):
            data.append(self.get_population(*data[-1], std_noise))

        return pd.DataFrame(data, columns=["larvae", "pupae", "adults"])


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


def main(n_steps, n_series):

    args_stable = {
        "b": 7.48,
        "c_ea": 0.009,
        "c_pa": 0.004,
        "c_el": 0.012,
        "mu_a": 0.73,
        "mu_l": 0.267,
    }

    args_aperiodic = {
        "b": 11.68,
        "c_ea": 0.009,
        "c_pa": 0.004,
        "c_el": 0.012,
        "mu_a": 0.96,
        "mu_l": 0.267,
    }

    # fix seed for reproducibility
    np.random.seed(42)
    # # number of time steps
    # n_steps = 120
    # # number of series per case
    # n_series = 200

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


if __name__ == "__main__":
    fire.Fire(main)
