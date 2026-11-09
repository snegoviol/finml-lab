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
# packages/finml-lib/src/finml_lib/data/styles.py
# ==================================================================================================

import matplotlib.pyplot as plt

# ==================================================================================================

def set_matplotlib_style() -> None:
    """
    Set default matplotlib style parameters for consistent plot appearance.

    Update matplotlib rcParams with predefined values for figure size,
    font sizes, line width, and grid settings.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The function updates matplotlib rcParams and does not return a value.
    """
    plt.rcParams.update({
        "figure.figsize": (20, 10),         # figure size (width, height) in inches
        "font.size": 12,                    # base font size
        "axes.titlesize": 14,               # plot title size (plt.title)
        "axes.labelsize": 12,               # axis labels size (plt.xlabel, plt.ylabel)
        "xtick.labelsize": 10,              # X-axis tick labels size
        "ytick.labelsize": 10,              # Y-axis tick labels size
        "legend.fontsize": 10,              # legend font size
        "figure.titlesize": 16,             # super title size (plt.suptitle)
        "lines.linewidth": 2.0,             # line width for plot lines
        "axes.grid": True,                  # enable grid by default
        "grid.alpha": 0.3,                  # grid line transparency (0.0 to 1.0)
    })

# ==================================================================================================
