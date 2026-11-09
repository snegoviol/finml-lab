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
# packages/finml-lib/src/finml_lib/target/statistics.py
# ==================================================================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf

# ==================================================================================================

def compute_acf_statistics(
    target: pd.Series,
    nlags: int = 24
) -> pd.DataFrame:
    """
    Calculate autocorrelation function (ACF) statistics.

    Computes the ACF for the given target series and returns a DataFrame
    containing summary statistics, including maximum, minimum, mean, and
    standard deviation of the autocorrelation values.

    Parameters
    ----------
    target : pandas.Series
        Input time series data. Must not contain NaN values.
    nlags : int, optional
        Number of lags to compute. Defaults to 24.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the ACF statistics with columns:
        - Metric: Name of the statistic
        - Value: The computed value
        - Lag: The lag at which the value occurs (only for Max/Min)

    Notes
    -----
    Lag 0 is excluded from the summary statistics because the autocorrelation
    of any series with itself at lag 0 is always exactly 1.0. Including it
    would artificially inflate the maximum and mean values, making the
    statistics unrepresentative of the actual lag correlations.
    """
    acf_values = acf(target, nlags=nlags)
    acf_without_lag0 = acf_values[1:]

    max_lag = np.argmax(acf_without_lag0) + 1
    min_lag = np.argmin(acf_without_lag0) + 1

    stats_df = pd.DataFrame({
        'Metric': pd.Series(
            [
                'Max ACF',
                'Min ACF',
                'Mean ACF',
                'Std ACF',
                'Abs Max'
            ], 
            dtype=object
        ),
        'Value': pd.Series(
            [
                acf_without_lag0.max(),
                acf_without_lag0.min(),
                acf_without_lag0.mean(),
                acf_without_lag0.std(),
                np.abs(acf_without_lag0).max()
            ],
            dtype='Float64'
        ),
        'Lag': pd.Series(
            [
                max_lag,
                min_lag,
                np.nan,
                np.nan,
                np.nan
            ], 
            dtype='Int64'
        )
    })

    return stats_df

# ==================================================================================================
