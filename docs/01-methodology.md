<!--    
    ================================================================================================
    
    Copyright (c) 2026 Ilya Snegov (aka Sierra Arn)
    
    Dedicated to the public domain under the Creative Commons
    CC0 1.0 Universal Public Domain Dedication (the "Dedication").
    You may obtain a copy of the Dedication at
    
    https://creativecommons.org/publicdomain/zero/1.0/legalcode.en

    ================================================================================================
    docs/01-methodology.md
    ================================================================================================
-->

# Methodology and Design Decisions

> *This document details the foundational architectural and methodological principles of the `finml-lab` pipeline, establishing the strategic rationale behind data selection, model design, and evaluation metrics that govern the end-to-end classical machine learning workflow.*

<!--    
    ================================================================================================
-->

## Core Design Choices

The following table outlines the foundational decisions across the data, modeling, and evaluation layers of the pipeline.

| Decision Area | Selected Approach | Rationale |
| :--- | :--- | :--- |
| **Dataset** | `arthurneuron/cryptocurrency-futures-ohlcv-dataset-1m-2024` (BTCUSDT) | Maximizes data volume for robust statistical analysis. Futures data captures true market sentiment and leverage. |
| **Asset Class** | Cryptocurrency (BTC/USDT) | Deliberate strategic choice to explore continuous 24/7 market dynamics, eliminating the complications of traditional market hour gaps and overnight jumps. |
| **Timeframe** | Hourly (1H) candles | Avoids the micro-structure noise of tick/second-level data (which typically requires Deep Learning) while preserving meaningful intraday trends suitable for classical ML. |
| **Model Architecture** | XGBoost Regressor (GBDT) | State-of-the-art for classical tabular ML. A regressor preserves the continuous nature of returns, allowing for richer visualization and nuanced error analysis compared to a classifier. |
| **Loss Function** | Huber Loss | Standard MSE disproportionately penalizes extreme outliers due to its quadratic nature. Huber loss behaves like MSE for small errors and MAE for large errors, smoothing the optimization landscape. [^1] |
| **Evaluation Metric** | Spearman Rank Correlation (Information Coefficient) | Predicting the exact magnitude of a return is highly difficult. Spearman evaluates how well the model ranks predictions (direction and relative magnitude), which directly correlates with trading strategy profitability. |
| **Lookback Periods** | `SHORT=24` (1 day), `MEDIUM=72` (3 days), `LONG=168` (1 week) | Logically aligned with standard hourly trading horizons, capturing distinct intraday, short-term swing, and weekly market cycles. |
| **Correlation Threshold** | `0.85` | Strikes an optimal balance by removing redundant, highly collinear features while preserving diverse, independent signals for the tree-based model. |

[^1]: The pipeline specifically implements the differentiable Pseudo-Huber variant; the detailed rationale for this substitution is provided in [03-model-development.ipynb](../notebooks/03-model-development.ipynb).

<!--    
    ================================================================================================
-->
