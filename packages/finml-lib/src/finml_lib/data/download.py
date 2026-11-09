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
# packages/finml-lib/src/finml_lib/data/download.py
# ==================================================================================================

import os
import kagglehub

# ==================================================================================================

def download_dataset_if_not_exists(
    handle: str,
    filename: str,
    output_dir: str
) -> str:
    """
    Download a dataset file from Kaggle if it does not already exist locally.

    Check whether the specified file exists in the output directory. If it
    exists, return its path without downloading. Otherwise, download the file
    using kagglehub and return the path to the downloaded file.

    Parameters
    ----------
    handle : str
        Kaggle dataset handle (e.g., "username/dataset-name").
    filename : str
        Name of the file to download (e.g., "BTCUSDT.csv").
    output_dir : str
        Directory where the file should be stored.

    Returns
    -------
    str
        Path to the dataset file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist after attempting to download.
    """
    file_path = os.path.join(output_dir, filename)

    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return file_path

    print(f"Downloading dataset from Kaggle...")

    path = kagglehub.dataset_download(
        handle=handle,
        path=filename,
        output_dir=output_dir
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Download failed. File not found at: {path}")

    print(f"Dataset downloaded successfully: {path}")

    return path

# ==================================================================================================
