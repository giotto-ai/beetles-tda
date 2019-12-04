import pandas as pd
import numpy as np


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
