# BTC Multi-level OFI Analysis
Author: [Haozhou Xu](https://hzempire.github.io/)

This project demonstrates how to:
1. Load BTC order book data.
2. Compute **multi-level Order Flow Imbalance (OFI)**.
3. Aggregate OFI into a single metric.
4. Perform a simple regression analysis of OFI vs. future returns.

## Project Structure
```bash
project/
├── data/
│   ├── BTC_1min.csv             # The BTC dataset (1-minute interval)
├── notebooks/
│   └── exploratory_analysis.ipynb  # Jupyter Notebook demonstrating EDA & analysis
├── scripts/
│   ├── data_preprocessing.py    # Data loading & basic cleaning
│   ├── ofi_calculation.py       # Multi-level OFI calculation
│   ├── cross_impact_analysis.py # Regression & visualization (single-asset version)
│   └── main.py                  # Main pipeline script
├── results/
│   ├── figures/                 # Charts & figures generated by the scripts/notebook
│   └── (other outputs)          
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview & usage
```

### 1. Data
- **BTC_1min.csv**: BTC order book data (1-minute intervals).

### 2. Notebooks
- **exploratory_analysis.ipynb**: Demonstrates data exploration and OFI analysis steps.

### 3. Scripts
- **data_preprocessing.py**: Loads and cleans `BTC_1min.csv`.
- **ofi_calculation.py**: Reconstructs prices/sizes from distance/notional columns, calculates multi-level OFI.
- **cross_impact_analysis.py**: Computes future returns, runs a simple regression, and visualizes.
- **main.py**: Orchestrates the entire process (load -> OFI -> regression -> plot).

### 4. Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run main script:
```bash
python -m scripts.main
```

### 5. Notes
- By default, up to 5 levels of the order book are used.
- Future returns are computed via `(midpoint_{t+1}/midpoint_t) - 1`.
- You can adjust parameters (levels, horizon, etc.) in the scripts.


