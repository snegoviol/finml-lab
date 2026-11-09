# ==================================================================================================
# 
# Copyright (c) 2026 Ilya Snegov
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# ==================================================================================================
# packages/finml-lib/src/finml_lib/features/postprocessing.py
# ==================================================================================================

import pandas as pd
import numpy as np
from scipy.sparse.csgraph import connected_components

# ==================================================================================================

def analyze_nan_distribution(df: pd.DataFrame) -> bool:
    """
    Analyze the distribution of NaN values in time series data.

    Identify rows containing at least one NaN value and check whether all such
    rows form a single continuous block. Print summary statistics about the
    number and position of NaN rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with time series data and features.

    Returns
    -------
    bool
        True if all NaN rows form a single continuous block or no NaN values
        are present, False if NaN rows are split into multiple blocks.

    Notes
    -----
    The primary purpose of this function is to decide how to handle NaN values
    in the dataset before training a model.

    If all NaN values form a single continuous block at the beginning of the
    data, this is the expected behavior caused by the warm-up period of
    TA-Lib indicators. Most technical indicators require a minimum number of
    previous bars to compute their first valid value, so the initial rows
    naturally contain NaN. In this case, it is safe to simply drop the leading
    NaN rows with `df.dropna()` without losing meaningful information.

    If NaN values appear in multiple blocks or in the middle of the dataset,
    this indicates a more serious issue such as missing bars, gaps in the
    source data, or aggregation errors. In such cases, deeper analysis is
    required before deciding on a handling strategy. Simply dropping NaN rows
    in the middle of a time series is not recommended, because it breaks the
    temporal structure of the data, distorts the meaning of lagged features,
    and can introduce look-ahead bias. Possible strategies for non-consecutive
    NaN values include forward filling, interpolation, or investigating and
    repairing the source data.
    """
    nan_mask = df.isna().any(axis=1).to_numpy()
    nan_positions = np.flatnonzero(nan_mask)

    nan_count = int(nan_positions.size)
    total_rows = len(df)

    print(f"Total rows with NaN values: {nan_count}")

    if nan_count == 0:
        print("No NaN values found in the data.")
        return True

    gaps = np.diff(nan_positions)
    is_continuous = bool((gaps == 1).all())
    block_count = int((gaps != 1).sum() + 1)

    if is_continuous:
        print("All NaN values are located in a single continuous block.")
    else:
        print(
            f"NaN values are distributed across {block_count} blocks "
            f"(gaps detected)."
        )

    first_position = int(nan_positions[0])
    last_position = int(nan_positions[-1])

    print(f"First NaN occurrence: {df.index[first_position]}")
    print(f"Last NaN occurrence: {df.index[last_position]}")
    print(f"First NaN position: {first_position / total_rows:.1%} from start")
    print(f"Last NaN position: {last_position / total_rows:.1%} from start")

    return is_continuous

# ==================================================================================================

def remove_zero_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns with zero variance or constant values.

    Identify and drop columns that contain only a single unique value
    or are entirely composed of missing values. This is particularly
    relevant for TA-Lib candlestick patterns, which often default to zero
    when no pattern is detected.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with zero-variance columns removed.
    """
    constant_columns = [
        col for col in df.columns if df[col].nunique() <= 1
    ]

    print(f"Removed {len(constant_columns)} zero-variance columns.")

    return df.drop(columns=constant_columns)

# ==================================================================================================

def select_features_by_correlation_clustering(
    X: pd.DataFrame,
    feature_importances: pd.Series,
    threshold: float,
) -> list[str]:
    """
    Select representative features using correlation-based clustering.

    Build a graph where features are nodes and edges represent absolute
    correlations above the specified threshold. Identify connected
    components (clusters) and select the feature with the highest
    importance from each cluster.

    Parameters
    ----------
    X : pandas.DataFrame
        Input feature matrix.
    feature_importances : pandas.Series
        Series mapping feature names to their baseline importance scores.
    threshold : float
        Absolute correlation threshold to form edges in the graph.

    Returns
    -------
    list of str
        List of selected representative feature names.

    Raises
    ------
    ValueError
        If the feature_importances index does not contain all columns of X.
    """
    missing_importances = set(X.columns) - set(feature_importances.index)
    if missing_importances:
        raise ValueError(
            f"Missing importances for features: {missing_importances}"
        )

    # Compute absolute correlation matrix and create adjacency matrix
    corr_matrix = X.corr().abs()
    adjacency_matrix = (corr_matrix > threshold).values
    
    # Identify connected components using SciPy (optimized graph traversal)
    n_clusters, cluster_labels = connected_components(
        adjacency_matrix, directed=False
    )
    
    # Select the feature with the highest importance from each cluster
    selected_features = []
    for cluster_id in range(n_clusters):
        cluster_features = X.columns[cluster_labels == cluster_id]
        best_feature = feature_importances.loc[cluster_features].idxmax()
        selected_features.append(best_feature)

    return selected_features

# ==================================================================================================
