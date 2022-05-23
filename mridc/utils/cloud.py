# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/cloud.py

import os
from pathlib import Path
from time import sleep

import wget

from mridc.utils import logging


def maybe_download_from_cloud(url, filename, subfolder=None, cache_dir=None, refresh_cache=False) -> str:
    """
    Download a file from a URL if it does not exist in the cache.

    Parameters
    ----------
    url: URL to download the file from.
        str
    filename: What to download. The request will be issued to url/filename
        str
    subfolder: Subfolder within cache_dir. The file will be stored in cache_dir/subfolder. Subfolder can be empty.
        str
    cache_dir: A cache directory where to download. If not present, this function will attempt to create it.
        str, If None (default), then it will be $HOME/.cache/torch/mridc
    refresh_cache: If True and cached file is present, it will delete it and re-fetch
        bool

    Returns
    -------
    If successful - absolute local path to the downloaded file else empty string.
    """
    if cache_dir is None:
        cache_location = Path.joinpath(Path.home(), ".cache/torch/mridc")
    else:
        cache_location = cache_dir
    if subfolder is not None:
        destination = Path.joinpath(cache_location, subfolder)
    else:
        destination = cache_location

    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)

    destination_file = Path.joinpath(destination, filename)

    if os.path.exists(destination_file):
        logging.info(f"Found existing object {destination_file}.")
        if refresh_cache:
            logging.info("Asked to refresh the cache.")
            logging.info(f"Deleting file: {destination_file}")
            os.remove(destination_file)
        else:
            logging.info(f"Re-using file from: {destination_file}")
            return str(destination_file)
    # download file
    wget_uri = url + filename
    logging.info(f"Downloading from: {wget_uri} to {str(destination_file)}")
    # NGC links do not work everytime so we try and wait
    i = 0
    max_attempts = 3
    while i < max_attempts:
        i += 1
        try:
            wget.download(wget_uri, str(destination_file))
            if os.path.exists(destination_file):
                return str(destination_file)
            return ""
        except Exception as e:
            logging.info(f"Download from cloud failed. Attempt {i} of {max_attempts}")
            logging.info(f"Error: {e}")
            sleep(0.05)
            continue
    raise ValueError("Not able to download url right now, please try again.")
