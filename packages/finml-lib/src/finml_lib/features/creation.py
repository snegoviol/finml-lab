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
# packages/finml-lib/src/finml_lib/features/creation.py
# ==================================================================================================

import pandas as pd
import numpy as np
import talib
from jaxtyping import Float64

# ==================================================================================================

def create_moving_averages_indicator_group(
    df: pd.DataFrame,
    close_prices: Float64[np.ndarray, "n"],
    short: int,
    medium: int,
    long: int
) -> pd.DataFrame:
    """
    Create moving average indicator features from the close price.

    Adds SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, and FAMA columns
    to the input DataFrame using the specified short, medium, and long periods.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    close_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of close prices.
    short : int
        Short lookback period.
    medium : int
        Medium lookback period.
    long : int
        Long lookback period.

    Returns
    -------
    pandas.DataFrame
        DataFrame with moving average features added.
    """
    new_features = {
        f"SMA_{short}": talib.SMA(close_prices, timeperiod=short),
        f"SMA_{medium}": talib.SMA(close_prices, timeperiod=medium),
        f"SMA_{long}": talib.SMA(close_prices, timeperiod=long),
        f"EMA_{short}": talib.EMA(close_prices, timeperiod=short),
        f"EMA_{medium}": talib.EMA(close_prices, timeperiod=medium),
        f"EMA_{long}": talib.EMA(close_prices, timeperiod=long),
        f"WMA_{short}": talib.WMA(close_prices, timeperiod=short),
        f"WMA_{medium}": talib.WMA(close_prices, timeperiod=medium),
        f"WMA_{long}": talib.WMA(close_prices, timeperiod=long),
        f"DEMA_{short}": talib.DEMA(close_prices, timeperiod=short),
        f"DEMA_{medium}": talib.DEMA(close_prices, timeperiod=medium),
        f"DEMA_{long}": talib.DEMA(close_prices, timeperiod=long),
        f"TEMA_{short}": talib.TEMA(close_prices, timeperiod=short),
        f"TEMA_{medium}": talib.TEMA(close_prices, timeperiod=medium),
        f"TEMA_{long}": talib.TEMA(close_prices, timeperiod=long),
        f"TRIMA_{short}": talib.TRIMA(close_prices, timeperiod=short),
        f"TRIMA_{medium}": talib.TRIMA(close_prices, timeperiod=medium),
        f"TRIMA_{long}": talib.TRIMA(close_prices, timeperiod=long),
        f"KAMA_{short}": talib.KAMA(close_prices, timeperiod=short),
        f"KAMA_{medium}": talib.KAMA(close_prices, timeperiod=medium),
        f"KAMA_{long}": talib.KAMA(close_prices, timeperiod=long),
    }

    mama, fama = talib.MAMA(close_prices)
    new_features["MAMA"] = mama
    new_features["FAMA"] = fama

    new_df = pd.DataFrame(new_features, index=df.index)

    return pd.concat([df, new_df], axis=1)

# ==================================================================================================

def create_overlap_studies_indicator_group(
    df: pd.DataFrame,
    close_prices: Float64[np.ndarray, "n"],
    high_prices: Float64[np.ndarray, "n"],
    low_prices: Float64[np.ndarray, "n"],
    short: int,
    medium: int,
    long: int
) -> pd.DataFrame:
    """
    Create overlap studies indicator features from price arrays.

    Adds HT_TRENDLINE, SAR, SAREXT, MIDPOINT, MIDPRICE, and Bollinger Bands
    columns to the input DataFrame using the specified short, medium, and long periods.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    close_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of close prices.
    high_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of high prices.
    low_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of low prices.
    short : int
        Short lookback period.
    medium : int
        Medium lookback period.
    long : int
        Long lookback period.

    Returns
    -------
    pandas.DataFrame
        DataFrame with overlap studies features added.
    """
    new_features = {
        "HT_TRENDLINE": talib.HT_TRENDLINE(close_prices),
        "SAR": talib.SAR(
            high_prices,
            low_prices,
            acceleration=0.02,
            maximum=0.2
        ),
        "SAREXT": talib.SAREXT(
            high_prices,
            low_prices
        ),
        f"MIDPOINT_{short}": talib.MIDPOINT(
            close_prices,
            timeperiod=short
        ),
        f"MIDPOINT_{medium}": talib.MIDPOINT(
            close_prices,
            timeperiod=medium
        ),
        f"MIDPOINT_{long}": talib.MIDPOINT(
            close_prices,
            timeperiod=long
        ),
        f"MIDPRICE_{short}": talib.MIDPRICE(
            high_prices,
            low_prices,
            timeperiod=short
        ),
        f"MIDPRICE_{medium}": talib.MIDPRICE(
            high_prices,
            low_prices,
            timeperiod=medium
        ),
        f"MIDPRICE_{long}": talib.MIDPRICE(
            high_prices,
            low_prices,
            timeperiod=long
        ),
    }

    upperband, middleband, lowerband = talib.BBANDS(
        close_prices,
        timeperiod=short
    )
    new_features[f"BBANDS_{short}_upper"] = upperband
    new_features[f"BBANDS_{short}_middle"] = middleband
    new_features[f"BBANDS_{short}_lower"] = lowerband

    upperband, middleband, lowerband = talib.BBANDS(
        close_prices,
        timeperiod=medium
    )
    new_features[f"BBANDS_{medium}_upper"] = upperband
    new_features[f"BBANDS_{medium}_middle"] = middleband
    new_features[f"BBANDS_{medium}_lower"] = lowerband

    upperband, middleband, lowerband = talib.BBANDS(
        close_prices,
        timeperiod=long
    )
    new_features[f"BBANDS_{long}_upper"] = upperband
    new_features[f"BBANDS_{long}_middle"] = middleband
    new_features[f"BBANDS_{long}_lower"] = lowerband

    new_df = pd.DataFrame(new_features, index=df.index)

    return pd.concat([df, new_df], axis=1)

# ==================================================================================================

def create_momentum_index_indicator_group(
    df: pd.DataFrame,
    high_prices: Float64[np.ndarray, "n"],
    low_prices: Float64[np.ndarray, "n"],
    close_prices: Float64[np.ndarray, "n"],
    volume: Float64[np.ndarray, "n"],
    short: int,
    medium: int,
    long: int
) -> pd.DataFrame:
    """
    Create momentum index indicator features from price and volume arrays.

    Adds ADX, ADXR, CCI, DX, MFI, RSI, and STOCHRSI columns to the input
    DataFrame using the specified short, medium, and long periods.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    high_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of high prices.
    low_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of low prices.
    close_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of close prices.
    volume : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of volume.
    short : int
        Short lookback period.
    medium : int
        Medium lookback period.
    long : int
        Long lookback period.

    Returns
    -------
    pandas.DataFrame
        DataFrame with momentum index features added.
    """
    new_features = {
        f"ADX_{short}": talib.ADX(
            high_prices, low_prices, close_prices, timeperiod=short
        ),
        f"ADX_{medium}": talib.ADX(
            high_prices, low_prices, close_prices, timeperiod=medium
        ),
        f"ADX_{long}": talib.ADX(
            high_prices, low_prices, close_prices, timeperiod=long
        ),
        f"ADXR_{short}": talib.ADXR(
            high_prices, low_prices, close_prices, timeperiod=short
        ),
        f"ADXR_{medium}": talib.ADXR(
            high_prices, low_prices, close_prices, timeperiod=medium
        ),
        f"ADXR_{long}": talib.ADXR(
            high_prices, low_prices, close_prices, timeperiod=long
        ),
        f"CCI_{short}": talib.CCI(
            high_prices, low_prices, close_prices, timeperiod=short
        ),
        f"CCI_{medium}": talib.CCI(
            high_prices, low_prices, close_prices, timeperiod=medium
        ),
        f"CCI_{long}": talib.CCI(
            high_prices, low_prices, close_prices, timeperiod=long
        ),
        f"DX_{short}": talib.DX(
            high_prices, low_prices, close_prices, timeperiod=short
        ),
        f"DX_{medium}": talib.DX(
            high_prices, low_prices, close_prices, timeperiod=medium
        ),
        f"DX_{long}": talib.DX(
            high_prices, low_prices, close_prices, timeperiod=long
        ),
        f"MFI_{short}": talib.MFI(
            high_prices, low_prices, close_prices, volume, timeperiod=short
        ),
        f"MFI_{medium}": talib.MFI(
            high_prices, low_prices, close_prices, volume, timeperiod=medium
        ),
        f"MFI_{long}": talib.MFI(
            high_prices, low_prices, close_prices, volume, timeperiod=long
        ),
        f"RSI_{short}": talib.RSI(
            close_prices, timeperiod=short
        ),
        f"RSI_{medium}": talib.RSI(
            close_prices, timeperiod=medium
        ),
        f"RSI_{long}": talib.RSI(
            close_prices, timeperiod=long
        ),
    }

    fastk_rsi, fastd_rsi = talib.STOCHRSI(
        close_prices, timeperiod=short
    )
    new_features[f"STOCHRSI_{short}_K"] = fastk_rsi
    new_features[f"STOCHRSI_{short}_D"] = fastd_rsi

    fastk_rsi, fastd_rsi = talib.STOCHRSI(
        close_prices, timeperiod=medium
    )
    new_features[f"STOCHRSI_{medium}_K"] = fastk_rsi
    new_features[f"STOCHRSI_{medium}_D"] = fastd_rsi

    fastk_rsi, fastd_rsi = talib.STOCHRSI(
        close_prices, timeperiod=long
    )
    new_features[f"STOCHRSI_{long}_K"] = fastk_rsi
    new_features[f"STOCHRSI_{long}_D"] = fastd_rsi

    new_df = pd.DataFrame(new_features, index=df.index)

    return pd.concat([df, new_df], axis=1)

# ==================================================================================================

def create_oscillator_indicator_group(
    df: pd.DataFrame,
    open_prices: Float64[np.ndarray, "n"],
    high_prices: Float64[np.ndarray, "n"],
    low_prices: Float64[np.ndarray, "n"],
    close_prices: Float64[np.ndarray, "n"],
    short: int,
    medium: int,
    long: int
) -> pd.DataFrame:
    """
    Create oscillator indicator features from price arrays.

    Adds APO, AROONOSC, PPO, ULTOSC, CMO, TRIX, WILLR, AROON, BOP, MACD,
    STOCH, and STOCHF columns to the input DataFrame using the specified
    short, medium, and long periods where applicable.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    open_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of open prices.
    high_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of high prices.
    low_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of low prices.
    close_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of close prices.
    short : int
        Short lookback period.
    medium : int
        Medium lookback period.
    long : int
        Long lookback period.

    Returns
    -------
    pandas.DataFrame
        DataFrame with oscillator features added.
    """
    new_features = {
        "APO": talib.APO(close_prices),
        "AROONOSC": talib.AROONOSC(
            high_prices,
            low_prices,
            timeperiod=medium
        ),
        "PPO": talib.PPO(close_prices),
        "ULTOSC": talib.ULTOSC(
            high_prices,
            low_prices,
            close_prices,
            timeperiod1=short,
            timeperiod2=medium,
            timeperiod3=long
        ),
        f"CMO_{short}": talib.CMO(
            close_prices,
            timeperiod=short
        ),
        f"CMO_{medium}": talib.CMO(
            close_prices,
            timeperiod=medium
        ),
        f"CMO_{long}": talib.CMO(
            close_prices,
            timeperiod=long
        ),
        f"TRIX_{short}": talib.TRIX(
            close_prices,
            timeperiod=short
        ),
        f"TRIX_{medium}": talib.TRIX(
            close_prices,
            timeperiod=medium
        ),
        f"TRIX_{long}": talib.TRIX(
            close_prices,
            timeperiod=long
        ),
        f"WILLR_{short}": talib.WILLR(
            high_prices,
            low_prices,
            close_prices,
            timeperiod=short
        ),
        f"WILLR_{medium}": talib.WILLR(
            high_prices,
            low_prices,
            close_prices,
            timeperiod=medium
        ),
        f"WILLR_{long}": talib.WILLR(
            high_prices,
            low_prices,
            close_prices,
            timeperiod=long
        ),
        "BOP": talib.BOP(
            open_prices,
            high_prices,
            low_prices,
            close_prices
        ),
    }

    aroon_down, aroon_up = talib.AROON(
        high_prices,
        low_prices,
        timeperiod=medium
    )
    new_features["AROON_down"] = aroon_down
    new_features["AROON_up"] = aroon_up

    macd, macd_signal, macd_hist = talib.MACD(close_prices)
    new_features["MACD"] = macd
    new_features["MACD_signal"] = macd_signal
    new_features["MACD_hist"] = macd_hist

    slowk, slowd = talib.STOCH(
        high_prices,
        low_prices,
        close_prices
    )
    new_features["STOCH_K"] = slowk
    new_features["STOCH_D"] = slowd

    fastk, fastd = talib.STOCHF(
        high_prices,
        low_prices,
        close_prices
    )
    new_features["STOCHF_K"] = fastk
    new_features["STOCHF_D"] = fastd

    new_df = pd.DataFrame(new_features, index=df.index)

    return pd.concat([df, new_df], axis=1)

# ==================================================================================================

def create_momentum_indicators_indicator_group(
    df: pd.DataFrame,
    high_prices: Float64[np.ndarray, "n"],
    low_prices: Float64[np.ndarray, "n"],
    close_prices: Float64[np.ndarray, "n"],
    short: int,
    medium: int,
    long: int
) -> pd.DataFrame:
    """
    Create momentum indicators features from price arrays.

    Adds MINUS_DI, MINUS_DM, MOM, PLUS_DI, PLUS_DM, ROC, ROCP, ROCR, and
    ROCR100 columns to the input DataFrame using the specified short, medium,
    and long periods.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    high_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of high prices.
    low_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of low prices.
    close_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of close prices.
    short : int
        Short lookback period.
    medium : int
        Medium lookback period.
    long : int
        Long lookback period.

    Returns
    -------
    pandas.DataFrame
        DataFrame with momentum indicators features added.
    """
    new_features = {
        f"MINUS_DI_{short}": talib.MINUS_DI(
            high_prices, low_prices, close_prices, timeperiod=short
        ),
        f"MINUS_DI_{medium}": talib.MINUS_DI(
            high_prices, low_prices, close_prices, timeperiod=medium
        ),
        f"MINUS_DI_{long}": talib.MINUS_DI(
            high_prices, low_prices, close_prices, timeperiod=long
        ),
        f"MINUS_DM_{short}": talib.MINUS_DM(
            high_prices, low_prices, timeperiod=short
        ),
        f"MINUS_DM_{medium}": talib.MINUS_DM(
            high_prices, low_prices, timeperiod=medium
        ),
        f"MINUS_DM_{long}": talib.MINUS_DM(
            high_prices, low_prices, timeperiod=long
        ),
        f"MOM_{short}": talib.MOM(
            close_prices, timeperiod=short
        ),
        f"MOM_{medium}": talib.MOM(
            close_prices, timeperiod=medium
        ),
        f"MOM_{long}": talib.MOM(
            close_prices, timeperiod=long
        ),
        f"PLUS_DI_{short}": talib.PLUS_DI(
            high_prices, low_prices, close_prices, timeperiod=short
        ),
        f"PLUS_DI_{medium}": talib.PLUS_DI(
            high_prices, low_prices, close_prices, timeperiod=medium
        ),
        f"PLUS_DI_{long}": talib.PLUS_DI(
            high_prices, low_prices, close_prices, timeperiod=long
        ),
        f"PLUS_DM_{short}": talib.PLUS_DM(
            high_prices, low_prices, timeperiod=short
        ),
        f"PLUS_DM_{medium}": talib.PLUS_DM(
            high_prices, low_prices, timeperiod=medium
        ),
        f"PLUS_DM_{long}": talib.PLUS_DM(
            high_prices, low_prices, timeperiod=long
        ),
        f"ROC_{short}": talib.ROC(
            close_prices, timeperiod=short
        ),
        f"ROC_{medium}": talib.ROC(
            close_prices, timeperiod=medium
        ),
        f"ROC_{long}": talib.ROC(
            close_prices, timeperiod=long
        ),
        f"ROCP_{short}": talib.ROCP(
            close_prices, timeperiod=short
        ),
        f"ROCP_{medium}": talib.ROCP(
            close_prices, timeperiod=medium
        ),
        f"ROCP_{long}": talib.ROCP(
            close_prices, timeperiod=long
        ),
        f"ROCR_{short}": talib.ROCR(
            close_prices, timeperiod=short
        ),
        f"ROCR_{medium}": talib.ROCR(
            close_prices, timeperiod=medium
        ),
        f"ROCR_{long}": talib.ROCR(
            close_prices, timeperiod=long
        ),
        f"ROCR100_{short}": talib.ROCR100(
            close_prices, timeperiod=short
        ),
        f"ROCR100_{medium}": talib.ROCR100(
            close_prices, timeperiod=medium
        ),
        f"ROCR100_{long}": talib.ROCR100(
            close_prices, timeperiod=long
        ),
    }

    new_df = pd.DataFrame(new_features, index=df.index)

    return pd.concat([df, new_df], axis=1)

# ==================================================================================================

def create_volume_cycle_indicator_group(
    df: pd.DataFrame,
    high_prices: Float64[np.ndarray, "n"],
    low_prices: Float64[np.ndarray, "n"],
    close_prices: Float64[np.ndarray, "n"],
    volume: Float64[np.ndarray, "n"]
) -> pd.DataFrame:
    """
    Create volume and cycle indicator features from price and volume arrays.

    Adds AD, ADOSC, OBV, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, and
    HT_TRENDMODE columns to the input DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    high_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of high prices.
    low_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of low prices.
    close_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of close prices.
    volume : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of volume.

    Returns
    -------
    pandas.DataFrame
        DataFrame with volume and cycle indicator features added.
    """
    new_features = {
        "AD": talib.AD(
            high_prices,
            low_prices,
            close_prices,
            volume
        ),
        "ADOSC": talib.ADOSC(
            high_prices,
            low_prices,
            close_prices,
            volume
        ),
        "OBV": talib.OBV(
            close_prices,
            volume
        ),
        "HT_DCPERIOD": talib.HT_DCPERIOD(close_prices),
        "HT_DCPHASE": talib.HT_DCPHASE(close_prices),
        "HT_TRENDMODE": talib.HT_TRENDMODE(close_prices),
    }

    inphase, quadrature = talib.HT_PHASOR(close_prices)
    new_features["HT_PHASOR_inphase"] = inphase
    new_features["HT_PHASOR_quadrature"] = quadrature

    sine, leadsine = talib.HT_SINE(close_prices)
    new_features["HT_SINE"] = sine
    new_features["HT_SINE_leadsine"] = leadsine

    new_df = pd.DataFrame(new_features, index=df.index)

    return pd.concat([df, new_df], axis=1)

# ==================================================================================================

def create_price_volatility_indicator_group(
    df: pd.DataFrame,
    open_prices: Float64[np.ndarray, "n"],
    high_prices: Float64[np.ndarray, "n"],
    low_prices: Float64[np.ndarray, "n"],
    close_prices: Float64[np.ndarray, "n"],
    short: int,
    medium: int,
    long: int
) -> pd.DataFrame:
    """
    Create price transform and volatility indicator features from price arrays.

    Adds AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE, ATR, NATR, and TRANGE columns
    to the input DataFrame using the specified short, medium, and long periods where applicable.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    open_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of open prices.
    high_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of high prices.
    low_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of low prices.
    close_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of close prices.
    short : int
        Short lookback period.
    medium : int
        Medium lookback period.
    long : int
        Long lookback period.

    Returns
    -------
    pandas.DataFrame
        DataFrame with price transform and volatility features added.
    """
    new_features = {
        "AVGPRICE": talib.AVGPRICE(
            open_prices,
            high_prices,
            low_prices,
            close_prices
        ),
        "MEDPRICE": talib.MEDPRICE(
            high_prices,
            low_prices
        ),
        "TYPPRICE": talib.TYPPRICE(
            high_prices,
            low_prices,
            close_prices
        ),
        "WCLPRICE": talib.WCLPRICE(
            high_prices,
            low_prices,
            close_prices
        ),
        f"ATR_{short}": talib.ATR(
            high_prices,
            low_prices,
            close_prices,
            timeperiod=short
        ),
        f"ATR_{medium}": talib.ATR(
            high_prices,
            low_prices,
            close_prices,
            timeperiod=medium
        ),
        f"ATR_{long}": talib.ATR(
            high_prices,
            low_prices,
            close_prices,
            timeperiod=long
        ),
        f"NATR_{short}": talib.NATR(
            high_prices,
            low_prices,
            close_prices,
            timeperiod=short
        ),
        f"NATR_{medium}": talib.NATR(
            high_prices,
            low_prices,
            close_prices,
            timeperiod=medium
        ),
        f"NATR_{long}": talib.NATR(
            high_prices,
            low_prices,
            close_prices,
            timeperiod=long
        ),
        "TRANGE": talib.TRANGE(
            high_prices,
            low_prices,
            close_prices
        ),
    }

    new_df = pd.DataFrame(new_features, index=df.index)

    return pd.concat([df, new_df], axis=1)

# ==================================================================================================

def create_pattern_recognition_indicator_group(
    df: pd.DataFrame,
    open_prices: Float64[np.ndarray, "n"],
    high_prices: Float64[np.ndarray, "n"],
    low_prices: Float64[np.ndarray, "n"],
    close_prices: Float64[np.ndarray, "n"]
) -> pd.DataFrame:
    """
    Create candlestick pattern recognition features from price arrays.

    Adds 61 candlestick pattern columns to the input DataFrame using TA-Lib
    pattern recognition functions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    open_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of open prices.
    high_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of high prices.
    low_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of low prices.
    close_prices : jaxtyping.Float64[np.ndarray, "n"]
        1D numpy array of close prices.

    Returns
    -------
    pandas.DataFrame
        DataFrame with candlestick pattern recognition features added.
    """
    pattern_functions = [
        talib.CDL2CROWS,
        talib.CDL3BLACKCROWS,
        talib.CDL3INSIDE,
        talib.CDL3LINESTRIKE,
        talib.CDL3OUTSIDE,
        talib.CDL3STARSINSOUTH,
        talib.CDL3WHITESOLDIERS,
        talib.CDLABANDONEDBABY,
        talib.CDLADVANCEBLOCK,
        talib.CDLBELTHOLD,
        talib.CDLBREAKAWAY,
        talib.CDLCLOSINGMARUBOZU,
        talib.CDLCONCEALBABYSWALL,
        talib.CDLCOUNTERATTACK,
        talib.CDLDARKCLOUDCOVER,
        talib.CDLDOJI,
        talib.CDLDOJISTAR,
        talib.CDLDRAGONFLYDOJI,
        talib.CDLENGULFING,
        talib.CDLEVENINGDOJISTAR,
        talib.CDLEVENINGSTAR,
        talib.CDLGAPSIDESIDEWHITE,
        talib.CDLGRAVESTONEDOJI,
        talib.CDLHAMMER,
        talib.CDLHANGINGMAN,
        talib.CDLHARAMI,
        talib.CDLHARAMICROSS,
        talib.CDLHIGHWAVE,
        talib.CDLHIKKAKE,
        talib.CDLHIKKAKEMOD,
        talib.CDLHOMINGPIGEON,
        talib.CDLIDENTICAL3CROWS,
        talib.CDLINNECK,
        talib.CDLINVERTEDHAMMER,
        talib.CDLKICKING,
        talib.CDLKICKINGBYLENGTH,
        talib.CDLLADDERBOTTOM,
        talib.CDLLONGLEGGEDDOJI,
        talib.CDLLONGLINE,
        talib.CDLMARUBOZU,
        talib.CDLMATCHINGLOW,
        talib.CDLMATHOLD,
        talib.CDLMORNINGDOJISTAR,
        talib.CDLMORNINGSTAR,
        talib.CDLONNECK,
        talib.CDLPIERCING,
        talib.CDLRICKSHAWMAN,
        talib.CDLRISEFALL3METHODS,
        talib.CDLSEPARATINGLINES,
        talib.CDLSHOOTINGSTAR,
        talib.CDLSHORTLINE,
        talib.CDLSPINNINGTOP,
        talib.CDLSTALLEDPATTERN,
        talib.CDLSTICKSANDWICH,
        talib.CDLTAKURI,
        talib.CDLTASUKIGAP,
        talib.CDLTHRUSTING,
        talib.CDLTRISTAR,
        talib.CDLUNIQUE3RIVER,
        talib.CDLUPSIDEGAP2CROWS,
        talib.CDLXSIDEGAP3METHODS
    ]

    new_features = {
        func.__name__: func(
            open_prices,
            high_prices,
            low_prices,
            close_prices
        )
        for func in pattern_functions
    }

    new_df = pd.DataFrame(new_features, index=df.index)

    return pd.concat([df, new_df], axis=1)

# ==================================================================================================

def create_time_recognition_indicator_group(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create time-based features from the DataFrame datetime index.

    Adds hour, day, month, dayofweek, dayofyear, candles_since_start, and
    their sine/cosine transformations for cyclical encoding to the input
    DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a DatetimeIndex.

    Returns
    -------
    pandas.DataFrame
        DataFrame with time recognition features added.

    Notes
    -----
    The `candles_since_start` feature calculates calendar hours from the first
    timestamp. This approach works correctly for cryptocurrency futures data
    because crypto markets operate 24/7 without gaps. After aggregating
    1-minute bars into hourly bars, each candle represents exactly 1 hour,
    so the time difference between consecutive candles is always 1 hour.

    For markets with trading breaks (e.g., stock markets with weekends or
    non-trading hours), this formula would count calendar time rather than
    actual trading periods, which may not be appropriate.
    """
    new_features = {
        "hour": df.index.hour,
        "day": df.index.day,
        "month": df.index.month,
        "dayofweek": df.index.day_of_week,
        "dayofyear": df.index.day_of_year,
        "candles_since_start": (
            (df.index - df.index[0]).total_seconds() / 3600
        ),
    }

    hour = new_features["hour"]
    new_features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    new_features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    day = new_features["day"]
    new_features["day_sin"] = np.sin(2 * np.pi * day / 31)
    new_features["day_cos"] = np.cos(2 * np.pi * day / 31)

    month = new_features["month"]
    new_features["month_sin"] = np.sin(2 * np.pi * month / 12)
    new_features["month_cos"] = np.cos(2 * np.pi * month / 12)

    dayofweek = new_features["dayofweek"]
    new_features["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    new_features["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)

    dayofyear = new_features["dayofyear"]
    new_features["dayofyear_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    new_features["dayofyear_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)

    new_df = pd.DataFrame(new_features, index=df.index)

    return pd.concat([df, new_df], axis=1)

# ==================================================================================================
