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

- To perform exploratory data analysis, engineer the target variable,  
and inspect data quality from raw cryptocurrency OHLCV data, see  
    [01-exploratory-data-analysis.ipynb](../notebooks/01-exploratory-data-analysis.ipynb).

- To generate technical indicators using TA-Lib, create temporal features,  
and perform feature selection from preprocessed cryptocurrency OHLCV data, see  
    [02-feature-engineering.ipynb](../notebooks/02-feature-engineering.ipynb).

- To train an XGBoost regressor and perform hyperparameter optimization  
via Optuna on the training and validation datasets, see  
    [03-model-development.ipynb](../notebooks/03-model-development.ipynb).

- To evaluate the final model performance on the unseen holdout test dataset,  
including metric calculation, prediction visualization, and feature importance analysis, see  
    [04-model-evaluation.ipynb](../notebooks/04-model-evaluation.ipynb).

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
