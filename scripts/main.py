"""
main.py

Main orchestration script that ties together data loading, 
OFI calculation, regression analysis, and saving results for BTC data.
"""

import os
from scripts.data_preprocessing import load_and_preprocess_data
from scripts.ofi_calculation import compute_ofi_for_btc, integrate_multi_level_ofi
from scripts.cross_impact_analysis import (
    compute_short_term_returns,
    run_regression_ofi_vs_returns,
    visualize_ofi_vs_returns
)

def main():
    # 1) Load data
    csv_path = os.path.join('data', 'BTC_1min.csv')
    df = load_and_preprocess_data(csv_path)

    # 2) Compute multi-level OFI for up to 5 levels
    df = compute_ofi_for_btc(df, max_levels=5)

    # 3) Integrate multi-level OFI into one metric (using 'sum' by default)
    df = integrate_multi_level_ofi(df, max_levels=5, method='sum')

    # 4) Compute short-term returns for horizon=1
    df = compute_short_term_returns(df, horizon=1)

    # 5) Run a simple linear regression of future returns on aggregated OFI
    results = run_regression_ofi_vs_returns(df, horizon=1)
    print("Regression results for horizon=1:")
    print(f"  Coefficient: {results['Coefficient']:.6f}")
    print(f"  Intercept: {results['Intercept']:.6f}")
    print(f"  R^2: {results['R^2']:.6f}")

    # 6) Visualize the relationship
    visualize_ofi_vs_returns(df, horizon=1)
    
    # You can also save the figure or store outputs in the results folder
    # ...

if __name__ == "__main__":
    main()
