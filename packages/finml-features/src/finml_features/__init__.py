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
# packages/finml-features/src/finml_features/__init__.py
# ==================================================================================================

import pandas as pd
import numpy as np
import talib

# ==================================================================================================

def _validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that the input DataFrame has a DatetimeIndex and all required
    OHLCV columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to validate.

    Raises
    ------
    TypeError
        If the DataFrame index is not a DatetimeIndex.
    ValueError
        If any of the required OHLCV columns is missing.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"DataFrame index must be a pandas.DatetimeIndex, "
            f"but got {type(df.index).__name__}. "
            f"Please set a DatetimeIndex before calling this function, "
            f"e.g.: df.index = pd.to_datetime(df.index)"
        )

    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"DataFrame is missing required OHLCV columns: {missing_columns}. "
            f"Expected columns: {required_columns}. "
            f"Actual columns: {list(df.columns)}"
        )

# ==================================================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the final, optimized set of 73 predictive features from OHLCV data.

    This function computes the strictly curated feature space derived from a
    rigorous multi-stage feature engineering and selection pipeline. The resulting
    73 indicators are the exact inputs required by the final trained model.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a DatetimeIndex and columns:
        'open', 'high', 'low', 'close', 'volume'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with exactly 73 optimized indicator features. Rows containing
        any NaN value are dropped.

    Raises
    ------
    TypeError
        If the DataFrame index is not a DatetimeIndex.
    ValueError
        If any of the required OHLCV columns is missing.

    Notes
    -----
    The 73 features returned are not an arbitrary collection of technical
    indicators. They represent the non-redundant output of a systematic selection
    framework designed to maximize predictive signal while strictly controlling
    for multicollinearity. The selection methodology comprises three core stages:

    1. **Baseline Model Evaluation:** An initial broad universe of technical
       indicators is evaluated against a baseline model to filter out
       statistically insignificant predictors and pure noise.
    2. **Graph-Theoretic Clustering (Multicollinearity Reduction):** To eliminate
       redundant information, the pairwise correlation matrix of the surviving
       features is thresholded and transformed into an adjacency graph. The
       `scipy.sparse.csgraph.connected_components` algorithm is applied to this
       graph to identify dense clusters of highly correlated features. From each
       connected component, only the single most informative representative is
       retained, ensuring structural orthogonality in the final feature space.
    3. **Final Curation:** The resulting 73 features constitute the minimal,
       low-collinearity feature space on which the final XGBoost regressor was
       trained, validated, and evaluated.

    Feature scaling is intentionally omitted. XGBoost's tree-based architecture
    is invariant to monotonic transformations, making standardization or
    normalization computationally redundant. Rows containing NaN values (arising
    from the warm-up period of rolling indicators) are dropped to ensure
    compatibility with the model's inference pipeline.
    """
    _validate_dataframe(df)

    open_prices = df["open"].to_numpy(dtype=np.float64)
    high_prices = df["high"].to_numpy(dtype=np.float64)
    low_prices = df["low"].to_numpy(dtype=np.float64)
    close_prices = df["close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    input_rows = len(df)

    new_features = {}

    # ----------------------------------------------------------------------------------------------
    # Time-based features 
    # ----------------------------------------------------------------------------------------------

    hour = df.index.hour
    day = df.index.day
    month = df.index.month
    dayofweek = df.index.day_of_week
    dayofyear = df.index.day_of_year

    new_features["hour"] = hour
    new_features["day"] = day
    new_features["dayofweek"] = dayofweek
    new_features["dayofyear"] = dayofyear

    # ----------------------------------------------------------------------------------------------
    # Cyclical encodings 
    # ----------------------------------------------------------------------------------------------

    new_features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    new_features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    new_features["day_sin"] = np.sin(2 * np.pi * day / 31)
    new_features["day_cos"] = np.cos(2 * np.pi * day / 31)

    new_features["month_sin"] = np.sin(2 * np.pi * month / 12)
    new_features["month_cos"] = np.cos(2 * np.pi * month / 12)

    new_features["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    new_features["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)

    # ----------------------------------------------------------------------------------------------
    # Overlap studies 
    # ----------------------------------------------------------------------------------------------

    new_features["MIDPRICE_24"] = talib.MIDPRICE(high_prices, low_prices, timeperiod=24)
    new_features["SAREXT"] = talib.SAREXT(high_prices, low_prices)

    # ----------------------------------------------------------------------------------------------
    # Momentum indicators 
    # ----------------------------------------------------------------------------------------------

    new_features["DX_24"] = talib.DX(high_prices, low_prices, close_prices, timeperiod=24)
    new_features["DX_72"] = talib.DX(high_prices, low_prices, close_prices, timeperiod=72)
    new_features["DX_168"] = talib.DX(high_prices, low_prices, close_prices, timeperiod=168)

    new_features["ADXR_24"] = talib.ADXR(high_prices, low_prices, close_prices, timeperiod=24)
    new_features["ADXR_72"] = talib.ADXR(high_prices, low_prices, close_prices, timeperiod=72)
    new_features["ADXR_168"] = talib.ADXR(high_prices, low_prices, close_prices, timeperiod=168)

    new_features["ADX_168"] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=168)

    new_features["RSI_72"] = talib.RSI(close_prices, timeperiod=72)

    new_features["ROC_24"] = talib.ROC(close_prices, timeperiod=24)
    new_features["ROC_168"] = talib.ROC(close_prices, timeperiod=168)

    new_features["MOM_72"] = talib.MOM(close_prices, timeperiod=72)

    new_features["WILLR_24"] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=24)

    new_features["MFI_24"] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=24)
    new_features["MFI_72"] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=72)
    new_features["MFI_168"] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=168)

    new_features["AROONOSC"] = talib.AROONOSC(high_prices, low_prices, timeperiod=72)

    _, fastd_rsi_72 = talib.STOCHRSI(close_prices, timeperiod=72)
    new_features["STOCHRSI_72_D"] = fastd_rsi_72

    fastk_stochf, _ = talib.STOCHF(high_prices, low_prices, close_prices)
    new_features["STOCHF_K"] = fastk_stochf

    aroon_down, aroon_up = talib.AROON(high_prices, low_prices, timeperiod=72)
    new_features["AROON_up"] = aroon_up
    new_features["AROON_down"] = aroon_down

    _, _, macd_hist = talib.MACD(close_prices)
    new_features["MACD_hist"] = macd_hist

    new_features["TRIX_168"] = talib.TRIX(close_prices, timeperiod=168)

    new_features["ULTOSC"] = talib.ULTOSC(
        high_prices, low_prices, close_prices,
        timeperiod1=24, timeperiod2=72, timeperiod3=168
    )

    # ----------------------------------------------------------------------------------------------
    # Volume indicators 
    # ----------------------------------------------------------------------------------------------

    new_features["OBV"] = talib.OBV(close_prices, volume)
    new_features["ADOSC"] = talib.ADOSC(high_prices, low_prices, close_prices, volume)
    new_features["volume"] = volume

    # ----------------------------------------------------------------------------------------------
    # Volatility 
    # ----------------------------------------------------------------------------------------------
    
    new_features["NATR_24"] = talib.NATR(high_prices, low_prices, close_prices, timeperiod=24)
    new_features["TRANGE"] = talib.TRANGE(high_prices, low_prices, close_prices)

    # ----------------------------------------------------------------------------------------------
    # Cycle indicators 
    # ----------------------------------------------------------------------------------------------

    new_features["HT_DCPERIOD"] = talib.HT_DCPERIOD(close_prices)
    new_features["HT_DCPHASE"] = talib.HT_DCPHASE(close_prices)
    new_features["HT_TRENDMODE"] = talib.HT_TRENDMODE(close_prices)

    ht_inphase, ht_quadrature = talib.HT_PHASOR(close_prices)
    new_features["HT_PHASOR_inphase"] = ht_inphase
    new_features["HT_PHASOR_quadrature"] = ht_quadrature

    ht_sine, ht_leadsine = talib.HT_SINE(close_prices)
    new_features["HT_SINE"] = ht_sine
    new_features["HT_SINE_leadsine"] = ht_leadsine

    # ----------------------------------------------------------------------------------------------
    # Price transform 
    # ----------------------------------------------------------------------------------------------

    new_features["BOP"] = talib.BOP(open_prices, high_prices, low_prices, close_prices)

    # ----------------------------------------------------------------------------------------------
    # Pattern recognition 
    # ----------------------------------------------------------------------------------------------

    pattern_funcs = {
        "CDLSPINNINGTOP": talib.CDLSPINNINGTOP,
        "CDLBELTHOLD": talib.CDLBELTHOLD,
        "CDLLONGLINE": talib.CDLLONGLINE,
        "CDLRICKSHAWMAN": talib.CDLRICKSHAWMAN,
        "CDL3LINESTRIKE": talib.CDL3LINESTRIKE,
        "CDLHIKKAKE": talib.CDLHIKKAKE,
        "CDLSEPARATINGLINES": talib.CDLSEPARATINGLINES,
        "CDLSHORTLINE": talib.CDLSHORTLINE,
        "CDL3INSIDE": talib.CDL3INSIDE,
        "CDLHARAMICROSS": talib.CDLHARAMICROSS,
        "CDLMATCHINGLOW": talib.CDLMATCHINGLOW,
        "CDLHIGHWAVE": talib.CDLHIGHWAVE,
        "CDLADVANCEBLOCK": talib.CDLADVANCEBLOCK,
        "CDLHAMMER": talib.CDLHAMMER,
        "CDLHARAMI": talib.CDLHARAMI,
        "CDL3OUTSIDE": talib.CDL3OUTSIDE,
        "CDLTAKURI": talib.CDLTAKURI,
        "CDLENGULFING": talib.CDLENGULFING,
        "CDLCLOSINGMARUBOZU": talib.CDLCLOSINGMARUBOZU,
        "CDLHIKKAKEMOD": talib.CDLHIKKAKEMOD,
        "CDLINVERTEDHAMMER": talib.CDLINVERTEDHAMMER,
        "CDLSTICKSANDWICH": talib.CDLSTICKSANDWICH,
        "CDLXSIDEGAP3METHODS": talib.CDLXSIDEGAP3METHODS,
    }

    for name, func in pattern_funcs.items():
        new_features[name] = func(open_prices, high_prices, low_prices, close_prices)

    # ----------------------------------------------------------------------------------------------
    # Build result 
    # ----------------------------------------------------------------------------------------------

    result_df = pd.DataFrame(new_features, index=df.index)
    result_df = result_df.dropna()

    print(
        f"create_selected_indicators: "
        f"input rows = {input_rows}, "
        f"output rows (after dropna) = {len(result_df)}"
    )

    return result_df

# ==================================================================================================
