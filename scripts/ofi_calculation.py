"""
ofi_calculation.py

Implements multi-level OFI calculations based on the methodology
described in 'Cross-impact of order flow imbalance in equity markets'
by Rama Cont, Mihai Cucuringu & Chao Zhang.

Here, we only have one asset (BTC), but we still demonstrate the
multi-level OFI approach with up to 5 levels (both bids and asks).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def reconstruct_price(df: pd.DataFrame, level: int, side: str) -> pd.Series:
    """
    Reconstruct approximate absolute price for the specified side and level.
    Because the data columns store distances from midpoint, we do:
       price = midpoint + distance
    If side is 'bid', distance is typically negative. 
    If side is 'ask', distance is typically positive.

    :param df: The DataFrame containing 'midpoint' and e.g. 'bids_distance_<level>'
    :param level: The level index (0 to 14 in the data)
    :param side: 'bid' or 'ask'
    :return: A Series of approximate absolute prices
    """
    col_name = f"{side}s_distance_{level}"
    return df['midpoint'] + df[col_name]

def reconstruct_size(df: pd.DataFrame, level: int, side: str) -> pd.Series:
    """
    Reconstruct approximate size (notional) for the specified side and level.
    The data columns store notional directly, e.g. 'bids_notional_<level>'. 
    We'll treat that as the 'size' for OFI computation.

    :param df: The DataFrame containing e.g. 'bids_notional_<level>'
    :param level: The level index (0 to 14 in the data)
    :param side: 'bid' or 'ask'
    :return: A Series with notional (which we treat as size)
    """
    col_name = f"{side}s_notional_{level}"
    return df[col_name]

def compute_ofi_for_btc(df: pd.DataFrame, max_levels: int = 5) -> pd.DataFrame:
    """
    Compute multi-level OFI for BTC. We'll create columns: OFI_level_<i>.

    :param df: DataFrame with midpoint, bids_distance_i, asks_distance_i,
               bids_notional_i, asks_notional_i columns.
    :param max_levels: how many levels of order book to use (out of up to 15)
    :return: DataFrame with new columns for OFI at each level
    """
    # Sort by system_time just to ensure chronological order
    df.sort_values('system_time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    for level in range(max_levels):
        # Reconstruct approximate bid/ask price
        bid_price = reconstruct_price(df, level, side='bid')
        ask_price = reconstruct_price(df, level, side='ask')

        # Reconstruct approximate size
        bid_size = reconstruct_size(df, level, side='bid')
        ask_size = reconstruct_size(df, level, side='ask')

        # SHIFT by 1 row to get previous state
        bid_price_prev = bid_price.shift(1)
        ask_price_prev = ask_price.shift(1)
        bid_size_prev = bid_size.shift(1)
        ask_size_prev = ask_size.shift(1)

        # OFI formula for level = 
        #  ( (bid_price - bid_price_prev) * (bid_size + bid_size_prev) / 2 )
        # - ( (ask_price - ask_price_prev) * (ask_size + ask_size_prev) / 2 )
        df[f'OFI_level_{level}'] = (
            (bid_price - bid_price_prev) * (bid_size + bid_size_prev) / 2
            - (ask_price - ask_price_prev) * (ask_size + ask_size_prev) / 2
        )
    
    # Fill NaN for the first row
    df.fillna(0, inplace=True)

    return df

def integrate_multi_level_ofi(df: pd.DataFrame, max_levels: int = 5, method: str = 'sum') -> pd.DataFrame:
    """
    Integrate multi-level OFI columns into a single metric using PCA or
    another method (sum by default).

    :param df: DataFrame with OFI_level_i columns
    :param max_levels: how many levels of the order book to consider
    :param method: 'PCA', 'sum', or 'avg'
    :return: DataFrame with an aggregated OFI column
    """
    ofi_cols = [f'OFI_level_{level}' for level in range(max_levels)]
    ofi_matrix = df[ofi_cols].values

    if method == 'PCA':
        pca = PCA(n_components=1)
        ofi_agg = pca.fit_transform(ofi_matrix)
        df['OFI_aggregated'] = ofi_agg
    elif method == 'sum':
        df['OFI_aggregated'] = df[ofi_cols].sum(axis=1)
    elif method == 'avg':
        df['OFI_aggregated'] = df[ofi_cols].mean(axis=1)
    else:
        raise ValueError("Unknown integration method. Choose from ['PCA','sum','avg'].")
    
    return df
