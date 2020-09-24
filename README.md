![logo](https://raw.githubusercontent.com/giotto-ai/giotto-tda/master/doc/images/tda_logo.svg)

# Analysing Beetle Population Dynamics with Topological Data Analysis

## What is it?
A guide to studying population dynamics with topological data analysis. We aim at correctly
identifying two clusters in an ensemble of time series describing the adult population of the 
[_Tribolium_ flour beetle](https://en.wikipedia.org/wiki/Red_flour_beetle). We compare our approach to a 
baseline approach where we cluster the unmodified time series.

See the accompanying [blog post](https://towardsdatascience.com/the-shape-of-population-dynamics-ba70f253919f) 
for further details.

## Data
Here, we simulate the evolution of the adult population of _Tribolium_ flour beetles. This approach is reasonable and justified as there exist mathematical models describing the population dynamics of _Tribolium_ flour beetles adequately. The beetles' life cycle consists of larva, pupa, and adult stages, with the transition between each stage lasting approximately two weeks. Remarkably, _Tribolium_ flour beetles become cold-hearted in face of overpopulation as they turn cannibalistic by eating unhatched eggs and pupae.

## Feature Creation
We use tools provided by topological data analysis to create several features. For each time series, these features are combined into one vector. The collection of those can then be clustered. The general pipeline is as follows: First, we embed each time series describing an adult population into a higher dimensional space. Next, we calculate the persistence diagrams and extract features from them. More details are given in the [blog post](https://towardsdatascience.com/the-shape-of-population-dynamics-ba70f253919f) and jupyter notebook.

## Model
For both the baseline and our approach, we use k-means to find clusters within the simulated data.

## Results
We show that an approach using TDA features outperforms the baseline. This even holds true in the presence of high environmental uncertainty.

## Getting started
Spin up a virtual environment and install the required libraries:

```
virtualenv -p python3.7 env
pip install -r requirements.txt
```

To make plotly play nice with JupyterLab one also needs to run:

```
# Avoid "JavaScript heap out of memory" errors during extension installation
# (OS X/Linux)
export NODE_OPTIONS=--max-old-space-size=4096
# (Windows)
set NODE_OPTIONS=--max-old-space-size=4096

# Jupyter widgets extension
jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.0 --no-build

# FigureWidget support
jupyter labextension install plotlywidget@1.2.0 --no-build

# and jupyterlab renderer support
jupyter labextension install jupyterlab-plotly@1.2.0 --no-build

# JupyterLab chart editor support (optional)
jupyter labextension install jupyterlab-chart-editor@1.2 --no-build

# Build extensions (must be done to activate extensions since --no-build is used above)
jupyter lab build

# Unset NODE_OPTIONS environment variable
# (OS X/Linux)
unset NODE_OPTIONS
# (Windows)
set NODE_OPTIONS=
```
