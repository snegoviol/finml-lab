<!--    
    ================================================================================================
    
    Copyright (c) 2026 Ilya Snegov
    
    Dedicated to the public domain under the Creative Commons
    CC0 1.0 Universal Public Domain Dedication (the "Dedication").
    You may obtain a copy of the Dedication at
    
    https://creativecommons.org/publicdomain/zero/1.0/legalcode.en

    ================================================================================================
    docs/README.md
    ================================================================================================
-->

# Financial Machine Learning Laboratory

> *This directory holds the full technical documentation for the Financial Machine Learning Laboratory and the predictive models trained within its pipeline. This file serves as the primary entry point: it provides a high-level overview of the model performance and engineering workflow, then directs to the documents covering each layer of the system, from conceptual methodology to empirical evaluation.*

<!--    
    ================================================================================================
-->

## I. Model Overview

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

<!--    
    ================================================================================================
-->

## II. Notebooks Overview

The practical implementation and empirical development of the model described above are detailed in a sequential series of Jupyter notebooks. Each notebook represents a distinct phase in the machine learning lifecycle, progressing from raw data ingestion to final out-of-sample evaluation.

For each phase, both the raw source code (located in the local repository) and a pre-rendered version (with static outputs, plots, and tables saved) are provided.

> **Important Note on Rendering:**  
> While the provided GitHub Gists contain saved outputs for quick review, please be aware of two rendering limitations specific to this platform:  
> 1. **Visual Formatting:**  
Minor visual differences may occur between the GitHub Gist rendering and how the notebook appears in the local development environment.  
> 2. **Interactive Visualizations:**  
GitHub Gist does not support interactive `plotly` graphs. Consequently, the output for these specific cells is entirely omitted in the rendered view, whereas all other cell outputs are fully preserved. To view these visualizations, the notebook must be executed locally.

| Phase & Objective | Source Code | Rendered Output |
| :--- | :--- | :--- |
| **01. Exploratory Data Analysis**<br>To perform EDA, engineer the target variable, and inspect data quality from raw OHLCV data. | [Local Notebook](../notebooks/01-exploratory-data-analysis.ipynb) | [GitHub Gist](https://gist.github.com/snegoviol/9c903f31ffaed21399e4477bfaa174e1) |
| **02. Feature Engineering**<br>To generate TA-Lib technical indicators, create temporal features, and perform feature selection. | [Local Notebook](../notebooks/02-feature-engineering.ipynb) | [GitHub Gist](https://gist.github.com/snegoviol/edc14844e68ce3ce4163295305cfa35b) |
| **03. Model Development**<br>To train the XGBoost regressor and perform hyperparameter optimization via Optuna. | [Local Notebook](../notebooks/03-model-development.ipynb) | [GitHub Gist](https://gist.github.com/snegoviol/aa37eb81b1a3381c0dcd9a4e86252607) |
| **04. Model Evaluation**<br>To evaluate final performance on the holdout test set, including metrics, visualization, and feature importance. | [Local Notebook](../notebooks/04-model-evaluation.ipynb) | [GitHub Gist](https://gist.github.com/snegoviol/40e106a101f478fef607b1ad87b5de98) |

<!--    
    ================================================================================================
-->

## III. Documentation Overview

The supplementary technical documentation for the Financial Machine Learning Laboratory is organized into dedicated files. Each document defines a distinct operational layer of the project, detailing the methodological rationale, runtime dependencies, and repository organization required to reproduce the pipeline.

- To understand the core architectural and methodological choices that define the pipeline, see  
    [01-methodology.md](01-methodology.md).

- To review the runtime dependencies, build backends, and workspace packages, see  
    [02-dependencies.md](02-dependencies.md).

- To understand the detailed project structure, see  
    [03-project-structure.md](03-project-structure.md).

<!--    
    ================================================================================================
-->
