"""
data_preprocessing.py

This script handles reading and cleaning the BTC dataset.
We demonstrate using BTC_1min.csv as an example.
"""

import pandas as pd

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load raw BTC L2 data from a CSV file, clean it, and return a DataFrame.

    :param filepath: Path to BTC_1min.csv
    :return: Cleaned DataFrame ready for OFI calculation
    """
    df = pd.read_csv(filepath)
    
    # Convert system_time to datetime
    df['system_time'] = pd.to_datetime(df['system_time'])
    
    # Sort by time
    df.sort_values(by='system_time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop any rows with missing values (if any)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
