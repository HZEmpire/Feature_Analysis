"""
cross_impact_analysis.py

In the original multi-asset framework, this script would handle
cross-asset regressions. Here we demonstrate a single-asset version:
we analyze how aggregated OFI relates to future returns for BTC.

We will compute short-term returns from midpoint and run a simple
linear regression as an example.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

def compute_short_term_returns(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Compute short-term returns for BTC based on the midpoint column.
    E.g. horizon=1 means we shift by 1 row to compute next-step return.

    :param df: DataFrame containing 'midpoint' column
    :param horizon: how many rows to shift for future returns
    :return: DataFrame with an added 'future_ret_<horizon>' column
    """
    # Example: (midpoint_{t+horizon} / midpoint_t) - 1
    df[f'future_ret_{horizon}'] = df['midpoint'].shift(-horizon) / df['midpoint'] - 1
    df.dropna(inplace=True)
    return df

def run_regression_ofi_vs_returns(df: pd.DataFrame, horizon: int = 1) -> dict:
    """
    Runs a simple linear regression of future returns on the aggregated OFI
    for BTC. We retrieve and return the R^2 and coefficient.

    :param df: DataFrame containing 'OFI_aggregated' and 'future_ret_<horizon>' 
    :param horizon: The horizon used for returns
    :return: Dict with regression results
    """
    y_col = f'future_ret_{horizon}'
    X = df[['OFI_aggregated']].values
    y = df[y_col].values

    reg = LinearRegression()
    reg.fit(X, y)
    r2 = reg.score(X, y)

    results = {
        'Coefficient': reg.coef_[0],
        'Intercept': reg.intercept_,
        'R^2': r2
    }
    return results

def visualize_ofi_vs_returns(df: pd.DataFrame, horizon: int = 1):
    """
    Create a scatter plot of aggregated OFI vs future returns for BTC,
    with a simple regression line.

    :param df: DataFrame containing 'OFI_aggregated' and 'future_ret_<horizon>'
    :param horizon: The horizon used for returns
    """
    y_col = f'future_ret_{horizon}'
    
    plt.figure(figsize=(7, 5))
    sns.regplot(x='OFI_aggregated', y=y_col, data=df, scatter_kws={'alpha':0.5})
    plt.title(f"BTC Future Returns (horizon={horizon}) vs. Aggregated OFI")
    plt.xlabel("OFI_aggregated")
    plt.ylabel(f"Future Return (horizon={horizon})")
    plt.tight_layout()
    plt.show()
