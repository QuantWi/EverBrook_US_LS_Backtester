# =========================================================
# Data Access
# =========================================================


from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import yfinance as yf

class GetStockData:
    """Load data and lookup tables from disk."""

    def __init__(self, data_source: str = "Yahoo") -> None:
        self.data_source = data_source

    def _get_yahoo_data(self) -> pd.DataFrame:
        """Read SPX500 pricing data from pickle."""
        pickle_file_path = (
            "/content/drive/MyDrive/Colab Notebooks/China_LS/all_sp500_dataframes.pkl"
        )
        try:
            df = pd.read_pickle(pickle_file_path)
            print(f"Successfully loaded pricing data from {pickle_file_path}")
            return df
        except FileNotFoundError:
            print(f"Error: The file {pickle_file_path} was not found.")
        except Exception as e:
            print(f"An error occurred while reading the pickle file: {e}")
        return pd.DataFrame()

    def _get_sector(self) -> Dict[str, str] | None:
        """Return a dict Ticker -> Sector (static, not time-varying)."""
        path = "/content/drive/MyDrive/Colab Notebooks/China_LS/sector_code.json"
        with open(path, "r") as f:
            return json.load(f)

    def _get_russell3000_data(self) -> pd.DataFrame:
        """Read Russell 3000 pricing (and drop CHRD explicitly)."""
        pickle_file_path = (
            "/content/drive/MyDrive/Colab Notebooks/China_LS/russell3000_prices.pkl"
        )
        try:
            df = pd.read_pickle(pickle_file_path)
            print(f"Successfully loaded pricing data from {pickle_file_path}")
        except FileNotFoundError:
            print(f"Error: The file {pickle_file_path} was not found.")
            return pd.DataFrame()
        except Exception as e:
            print(f"An error occurred while reading the pickle file: {e}")
            return pd.DataFrame()

        ticker_removed = ['DEC', 'CHRD','IVT']

        # Find which tickers exist in the dataframe
        existing_tickers = set(df["Ticker"].unique())
        removed = existing_tickers.intersection(ticker_removed)

        if removed:
            # Use ~ instead of ! for negation
            df = df[~df["Ticker"].isin(ticker_removed)].copy()
            print(f"Removed tickers: {', '.join(removed)}")
        else:
            print("No tickers found to remove.")

        return df

    def _get_russell3000_sector(self) -> Dict[str, str] | None:
        """Return a dict Ticker -> Sector (static)."""
        path = "/content/drive/MyDrive/Colab Notebooks/China_LS/sectorcode_russell3000.json"
        with open(path, "r") as f:
            return json.load(f)

    def _get_neutralize_data(self) -> pd.DataFrame:
        """Load factor table used for neutralization."""
        pickle_file_path = (
            "/content/drive/MyDrive/Colab Notebooks/China_LS/factors_r3000_full.pkl"
        )
        try:
            df = pd.read_pickle(pickle_file_path)
            print(f"Successfully loaded pricing data from {pickle_file_path}")
            return df
        except FileNotFoundError:
            print(f"Error: The file {pickle_file_path} was not found.")
        except Exception as e:
            print(f"An error occurred while reading the pickle file: {e}")
        return pd.DataFrame()