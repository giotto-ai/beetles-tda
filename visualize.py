import numpy as np
import plotly.graph_objs as gobj
from giotto.diagrams._utils import _subdiagrams
import seaborn as sns
import matplotlib.pyplot as plt


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


def plot_time_series(data, series_id, n_steps):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 8))

    sns.lineplot(
        x=range(n_steps + 1),
        y=data[(data["series_id"] == series_id) & (data["type"] == "stable")]["adults"],
        ax=ax[0],
    ).set_title("Stable")

    sns.lineplot(
        x=range(n_steps + 1),
        y=data[(data["series_id"] == series_id) & (data["type"] == "aperiodic")][
            "adults"
        ],
        ax=ax[1],
    ).set_title("Aperiodic")

    plt.show()
