<!--    
    ================================================================================================
    
    Copyright (c) 2026 Ilya Snegov
    
    Dedicated to the public domain under the Creative Commons
    CC0 1.0 Universal Public Domain Dedication (the "Dedication").
    You may obtain a copy of the Dedication at
    
    https://creativecommons.org/publicdomain/zero/1.0/legalcode.en

    ================================================================================================
    README.md
    ================================================================================================
-->

# Financial Machine Learning Laboratory

> *A research laboratory for financial machine learning, focused on BTC/USDT return prediction 
using Gradient Boosting Decision Tree (GBDT) regressor.*

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](#license)
[![License: CC0 1.0](https://img.shields.io/badge/License-CC0%201.0-blue.svg)](#license)

<!--    
    ================================================================================================
-->

## Model Overview

The project implements a Gradient Boosting Decision Tree (GBDT) regressor for forecasting hourly BTC/USDT returns, designed as a cross-sectional ranking mechanism for long/short portfolio construction rather than an absolute magnitude predictor.

**Configuration**

| | |
|------------------------------|-------------------------------------------------------------|
| Target Variable              | Hourly forward returns (winsorized at 1st/99th percentiles) |
| Feature Space                | 73 TA-Lib technical indicators + temporal encodings         |
| Feature Categories           | Momentum, volatility, volume, candlestick patterns, time cycles |
| Model Architecture           | XGBoost Regressor with Pseudo-Huber loss                    |
| Hyperparameter Optimization  | Optuna TPE sampler (906 trials, SQLite persistence)         |
| Validation Strategy          | Temporal split (train/valid/test), no lookahead bias        |
| Regularization               | `L1`/`L2` penalties, `min_child_weight`, `early stopping`   |

**Results**

| | |
|----------------------------|---------------------|
| Validation Spearman IC     | 0.0765              |
| Test Spearman IC           | 0.0424 ($p=0.0021$) |
| Test Pearson Correlation   | 0.0293 ($p=0.0330$) |
| Rank-Linear Gap            | 0.0130              |
| Feature Importance Spread  | Top-10 features explain 20.16% of gain |
| Zero-Importance Features   | 7 (ultra-rare candlestick patterns pruned) |

> **Model Interpretation**  
> The model successfully captures non-linear rank-order dependencies ($\rho_S = 0.0424 > \rho_P = 0.0293$, $\Delta = 0.0130$) but exhibits extreme variance shrinkage, with predictions heavily clustered around zero. This conservative behavior, resulting in poor absolute magnitude and volatility forecasting, is a direct and deliberate consequence of the Pseudo-Huber loss function applied to winsorized targets. Consequently, the model is strictly optimized for cross-sectional ranking (Information Coefficient) rather than point-estimate accuracy. The distributed feature importance landscape (28 features for 50% gain) further confirms robust signal aggregation without overfitting to spurious predictors.

> **Want more details?**  
> For the full technical documentation — including the methodological rationale behind data and model choices, the runtime dependency stack, and the detailed repository structure — see [the documentation](./docs/README.md).

<!--    
    ================================================================================================
-->

## Project Structure at a Glance

```shell
finml-lab/
├── notebooks/          # Jupyter notebooks for the ML pipeline: data exploration, 
│                       # model prototyping, and evaluation.
│                       # Each notebook represents one sequential stage of the 
│                       # machine learning lifecycle.
│
├── packages/           # Internal Python packages built via `pixi-build-python`. 
│                       # Each subdirectory represents a self-contained Conda 
│                       # package recipe.
│
├── docs/               # Technical documentation covering workspace dependencies,
│                       # and detailed project structure.
│
├── LICENSES/           # Full texts of all licenses used in the project.
│
├── LICENSE.txt         # Licensing overview: which license applies to which files.
│
├── pixi.toml           # Workspace manifest: channels and dependencies.
│
└── pixi.lock           # Fully resolved and reproducible dependency lockfile.
```

<!--    
    ================================================================================================
-->

## Quick Start

### I. Prerequisites

| Component            | Requirement                               |
| :------------------- | :---------------------------------------- |
| **Package Manager**  | [`Pixi`](https://pixi.prefix.dev/latest/) |
| **Operating System** | GNU/Linux-based system                    |
| **Architecture**     | `x86_64`                                  |

> **Note:**  
> These are not strict requirements, but rather the baseline environment where the code is 
guaranteed to run out of the box. Alternative setups are possible but may require manual 
configuration.

<!--    
    ================================================================================================
-->

### II. Setup

1. **Clone the repository**

    ```bash
    git clone https://github.com/snegoviol/finml-lab.git
    cd finml-lab
    ```

2. **Install dependencies**

    ```bash
    pixi install
    ```

<!--    
    ================================================================================================
-->

### III. Execution Workflow

Once the Pixi environment is installed, you can:
- navigate to the `notebooks/` directory,
- launch the notebooks in your preferred environment (e.g., JupyterLab, VS Code, or via CLI),
- execute them sequentially (from `01` to `04`) to reproduce the end-to-end machine learning pipeline,
- design and execute your own custom experiments,

— all dependencies, Python paths, and environment variables are already configured by Pixi.

> **Want more details?**  
> For a comprehensive breakdown of each notebook's objectives, rendering notes, and direct links to pre-rendered outputs, see the [Notebooks Overview](./docs/README.md#ii-notebooks-overview) section in the documentation.

<!--    
    ================================================================================================
-->

## License

Each file in this repository is licensed under one of the following licenses:

- [Apache License, Version 2.0](LICENSES/Apache-2.0.txt),
- [Creative Commons CC0 1.0 Universal Public Domain Dedication](LICENSES/CC0-1.0.txt).

The applicable license is stated in the header comment of each file. 
See [LICENSE.txt](LICENSE.txt) for details.

<!--    
    ================================================================================================
-->
