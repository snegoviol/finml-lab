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
# packages/finml-lib/src/finml_lib/target/creation.py
# ==================================================================================================

import pandas as pd

# ==================================================================================================

def add_next_period_return_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a next-period return target column named `target` to the DataFrame.

    The target is defined as:

        target[t] = close[t + 1] / close[t] - 1.0

    For each row, this is the relative change of the close price from the
    current bar to the next bar. In other words, it is the next-period return.

    A future model using this target will try to predict the next-period
    relative price change:

    - positive target values mean that the close price increases;
    - negative target values mean that the close price decreases;
    - zero means that the close price does not change.

    The last row is removed from the output because it does not have a next
    close value, making its target undefined.

    If a column named `target` already exists, the function prints a message
    and returns the input DataFrame unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a `close` column. Rows are expected to be
        sorted chronologically.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the `target` column added and the last row removed.
        If the `target` column already exists, the input DataFrame is returned
        unchanged.

    Raises
    ------
    KeyError
        If the input DataFrame does not contain a `close` column and the
        `target` column does not exist.
    """
    if "target" in df.columns:
        print("Target already exists.")
        return df

    if "close" not in df.columns:
        raise KeyError(f"Input DataFrame does not contain a 'close' column.")

    result_df = df.copy()

    result_df["target"] = (
        result_df["close"].shift(-1) / result_df["close"] - 1.0
    )

    result_df = result_df.iloc[:-1]

    return result_df

# ==================================================================================================
