<!--    
    ================================================================================================
    
    Copyright (c) 2026 Ilya Snegov
    
    Dedicated to the public domain under the Creative Commons
    CC0 1.0 Universal Public Domain Dedication (the "Dedication").
    You may obtain a copy of the Dedication at
    
    https://creativecommons.org/publicdomain/zero/1.0/legalcode.en

    ================================================================================================
    docs/03-project-structure.md
    ================================================================================================
-->

# III. Detailed Project Structure

> *This document details the logical organization of the `finml-lab` codebase, structured as a 
Pixi workspace that bridges interactive Jupyter notebook exploration with reusable internal 
Python packages for financial machine learning.*

<!--    
    ================================================================================================
-->

## Repository Layout

```shell
finml-lab/
├── notebooks/          # Jupyter notebooks for the ML pipeline: data exploration, 
│   │                   # model prototyping, and evaluation.
│   │
│   ├── data/                               # All artifacts generated during notebook execution.
│   │
│   ├── 01-exploratory-data-analysis.ipynb  # Jupyter notebook for exploratory data analysis, 
│   │                                       # target variable engineering, and data quality 
│   │                                       # inspection.
│   │
│   ├── 02-feature-engineering.ipynb        # Jupyter notebook for generating features, and 
│   │                                       # performing feature selection.
│   │
│   ├── 03-model-development.ipynb          # Jupyter notebook for training model and 
│   │                                       # performing hyperparameter optimization.
│   │
│   └── 04-model-evaluation.ipynb           # Jupyter notebook for evaluating the model performance.
│
├── packages/           # Internal Python packages built via pixi-build-python.
│   │
│   ├── finml-lib/                          # Reusable utility package containing shared helper 
│   │                                       # functions, data validation, and common abstractions 
│   │                                       # extracted from notebooks to keep them focused on 
│   │                                       # exploration.
│   │
│   └── finml-features/                     # Lightweight standalone module designed to
│                                           # effortlessly generate model-ready inputs,
│                                           # enabling quick interaction with the trained
│                                           # model without manual pipeline recreation.
│
├── LICENSES/           # Full texts of all licenses used in the project.
│   │
│   ├── Apache-2.0.txt                      # Apache License, Version 2.0.
│   │
│   └── CC0-1.0.txt                         # Creative Commons CC0 1.0 Universal Public Domain
│                                           # Dedication.
│
├── LICENSE.txt         # Licensing overview: states which license applies
│                       # to which files; refers to `LICENSES/`.
│
├── pixi.toml           # Workspace manifest: channels and dependencies.
│
└── pixi.lock           # Fully resolved and reproducible dependency lockfile.
```

Source files are documented with detailed docstrings and/or inline comments to explain the code.

<!--    
    ================================================================================================
-->

## Workspace Overview

### 1. `notebooks/data/`

The `notebooks/data/` directory contains all raw, intermediate, and final data artifacts generated during the machine learning pipeline. The file layout is structured chronologically by processing stage:

```shell
finml-lab/notebooks/data/
├── BTCUSDT.csv                   # Raw historical OHLCV data serving as the initial source of truth.
│
├── train.parquet                 # Initial temporal train split before final preprocessing.
├── valid.parquet                 # Initial temporal validation split before final preprocessing.
├── test.parquet                  # Initial temporal test split before final preprocessing.
│
├── preprocessed_train.parquet    # Final, feature-engineered train set ready for model fitting.
├── preprocessed_valid.parquet    # Final, feature-engineered validation set for hyperparameter tuning.
├── preprocessed_test.parquet     # Final, feature-engineered test set for out-of-sample evaluation.
│
├── optuna_study.db               # SQLite database storing the complete Optuna hyperparameter optimization history.
│
└── finml_gbdt_regressor.json     # Serialized Gradient Boosting Decision Tree (GBDT) model architecture and weights.
```

> **Note:**  
> All `.csv`, `.parquet`, `.db`, and `.json` files are generated artifacts and are gitignored. After a fresh clone, this directory will be empty until the data processing and model development notebooks are executed.

<!--    
    ================================================================================================
-->

### 2. `packages/finml-lib/`

Internal Python package containing reusable utility modules extracted from Jupyter notebooks to keep them focused on exploration. Provides shared helper functions, data validation, and common abstractions used across the entire project.

```shell
finml-lab/packages/finml-lib/
├── pyproject.toml                      # PyPI package manifest: 
│                                       # Hatchling build backend and metadata. 
│
├── pixi.toml                           # Conda package manifest: 
│                                       # `pixi-build-python` build backend, 
│                                       # host and run dependencies.
│
└── src/finml_lib/
    │
    ├── __init__.py                     # Initializes the root directory as a Python module.
    │
    ├── data/                           # Module for data acquisition and formatting.
    │   │
    │   ├── __init__.py                 # Initializes the directory as a Python module.
    │   │
    │   ├── download.py                 # Downloads raw datasets (e.g., via Kaggle) and saves 
    │   │                               # them locally if the file does not already exist.
    │   │
    │   └── styles.py                   # Defines visual styles and formatting configurations for
    │                                   # data plotting.
    │
    ├── features/                       # Module for feature engineering and postprocessing.
    │   │
    │   ├── __init__.py                 # Initializes the directory as a Python module.
    │   │
    │   ├── creation.py                 # Computes technical indicators.
    │   │
    │   └── postprocessing.py           # Applies final transformations to generated features.
    │
    └── target/                         # Module for target variable engineering and analysis.
        │
        ├── __init__.py                 # Initializes the directory as a Python module.
        │
        ├── creation.py                 # Computes the target variable.
        │
        └── statistics.py               # Computes Autocorrelation Function (ACF) metrics 
                                        # for the target variable.
```

<!--    
    ================================================================================================
-->

### 3. `packages/finml-features/`

Lightweight, standalone Python package designed to simplify the creation of model-ready inputs. Enables users to easily install a minimal module and generate the exact features required for interacting with the trained model, without manually recreating the complex transformation pipeline.

```shell
finml-lab/packages/finml-features/
├── pyproject.toml                      # PyPI package manifest: 
│                                       # Hatchling build backend and metadata. 
│
├── pixi.toml                           # Conda package manifest: 
│                                       # `pixi-build-python` build backend, 
│                                       # host and run dependencies.
│
└── src/finml_features/
    │
    └── __init__.py                     # Computes model-ready inputs directly from raw data by 
                                        # encapsulating the complete feature engineering pipeline.
```

<!--    
    ================================================================================================
-->
