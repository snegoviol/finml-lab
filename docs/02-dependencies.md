<!--    
    ================================================================================================
    
    Copyright (c) 2026 Ilya Snegov (aka Sierra Arn)
    
    Dedicated to the public domain under the Creative Commons
    CC0 1.0 Universal Public Domain Dedication (the "Dedication").
    You may obtain a copy of the Dedication at
    
    https://creativecommons.org/publicdomain/zero/1.0/legalcode.en

    ================================================================================================
    docs/02-dependencies.md
    ================================================================================================
-->

# II. Dependencies Overview

> *This document describes the runtime dependencies required by the Financial Machine Learning Laboratory.*

<!--    
    ================================================================================================
-->

## System Dependencies

| Dependency | Repository | What it is | Role in the project |
|---|---|---|---|
| Pixi | [prefix‑dev/pixi](https://github.com/prefix-dev/pixi) | Package and environment manager | 1. Resolves and installs both conda and PyPI dependencies. <br>2. Manages the project's virtual environment and lockfile. |
| pixi-build-python | [prefix‑dev/pixi](https://github.com/prefix-dev/pixi) | Pixi build backend for Python packages | Builds the internal `finml-features` and `finml-lib` packages from their `pyproject.toml` files located in the `packages/` directory. |

<!--    
    ================================================================================================
-->

## Pixi Dependencies

| Dependency | Repository | What it is | Role in the project |
|---|---|---|---|
| CPython | [python/cpython](https://github.com/python/cpython) | The de facto standard implementation of the Python programming language | Executes all source code across every package. |
| **pandas, numpy** | [pandas-dev/pandas](https://github.com/pandas-dev/pandas), [numpy/numpy](https://github.com/numpy/numpy) | Foundational libraries for numerical computation and structured data manipulation | Performs core data transformations, array operations, and tabular data handling across the entire pipeline. |
| **jupyterlab** | [jupyterlab/jupyterlab](https://github.com/jupyterlab/jupyterlab) | The native, web-based development environment for Jupyter Notebooks | Serves as a reliable fallback for executing `.ipynb` files when IDE notebook extensions encounter stability issues or performance bottlenecks. |
| **kagglehub** | [Kaggle/kagglehub](https://github.com/Kaggle/kagglehub) | An official Python client for the Kaggle platform | Facilitates the seamless downloading and management of datasets directly from Kaggle. |
| **pyarrow** | [apache/arrow](https://github.com/apache/arrow) | A cross-language development platform for in-memory data and high-performance I/O | Serves as the backend engine for efficient reading and writing of Parquet files. |
| **statsmodels, scipy** | [statsmodels/statsmodels](https://github.com/statsmodels/statsmodels), [scipy/scipy](https://github.com/scipy/scipy) | Comprehensive libraries for statistical modeling and scientific algorithms | Executes specialized statistical transformations and calculations, such as Autocorrelation Function (ACF) and connected components analysis. |
| **ta-lib** | [TA-Lib/ta-lib-python](https://github.com/TA-Lib/ta-lib-python) | A widely used library for technical analysis of financial market data | Computes standard financial indicators and candlestick pattern recognition features. |
| **xgboost** | [dmlc/xgboost](https://github.com/dmlc/xgboost) | An optimized distributed gradient boosting library | Trains the Gradient Boosting Decision Tree (GBDT) regressor to forecast financial returns. |
| **optuna, rustuna** | [optuna/optuna](https://github.com/optuna/optuna), [optuna/rustuna](https://github.com/optuna/rustuna) | A flexible hyperparameter optimization framework and its high-performance Rust-based accelerator | Automates the search for optimal model configurations, leveraging Rustuna to significantly speed up the TPE sampling process. |
| **matplotlib, seaborn** | [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib), [mwaskom/seaborn](https://github.com/mwaskom/seaborn) | Foundational plotting libraries for creating static, publication-quality graphics | Generates standard statistical plots, distribution comparisons, and feature importance visualizations. |
| **plotly** | [plotly/plotly.py](https://github.com/plotly/plotly.py) | An interactive, browser-based graphing library | Renders dynamic, zoomable charts for deep-dive exploratory data analysis. |
| **ipywidgets** | [jupyter-widgets/ipywidgets](https://github.com/jupyter-widgets/ipywidgets) | HTML widgets for the Jupyter Notebook | Enables interactive controls and dynamic outputs within notebook cells. |
| **jaxtyping** | [google/jaxtyping](https://github.com/google/jaxtyping) | A type-checking extension for array libraries | Provides detailed, shape-aware type hints for developers without imposing runtime performance overhead. |
| **tqdm** | [tqdm/tqdm](https://github.com/tqdm/tqdm) | A fast, extensible progress bar library for Python | Suppresses import warnings generated by Optuna, ensuring clean execution logs (installed as a silent dependency). |
| **hatchling** | [pypa/hatch](https://github.com/pypa/hatch) | A modern, PEP 517 compliant build backend for Python | Serves as the underlying build system invoked by `pixi-build-python` to compile the internal `finml-features` and `finml-lib` modules. |
| **finml-lib** | `Internal` | A centralized utility package of reusable modules extracted from exploratory Jupyter notebooks for the Financial Machine Learning Laboratory. | Keeps notebooks focused on prototyping by abstracting bulky utility code, while providing shared helper functions and data validation across the project. |
| **finml-features** | `Internal` | A streamlined feature engineering module designed to simplify the creation of model-ready inputs for the Financial Machine Learning Laboratory. | Simplifies the data preparation pipeline by abstracting complex transformations into an intuitive interface, generating the exact technical indicators, temporal encodings, and patterns required by the predictive models. |

<!--    
    ================================================================================================
-->
