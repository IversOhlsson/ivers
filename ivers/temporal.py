"""
Filename: temporal.py

Code for the "leaky" and "AllForOne" splits

Functions "leaky_endpoint_split" and "allforone_endpoint_split" used when
only generate one test/train split

Functions "allforone_folds_endpoint_split" and "leaky_folds_endpoint_split"
where it is possible split data into multiple sections and concistently 
increase the train set

Author: Philip Ivers Ohlsson
License: MIT License 
"""

import pandas as pd
from pandas import DataFrame
import logging
from typing import Tuple, List, Dict
import numpy as np
import os
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_aggregation_rules(df: DataFrame, exclude_columns: List[str]) -> dict:
    """
    Determine aggregation rules for the DataFrame columns based on their data types.
    """
    return {col: ('mean' if df[col].dtype in [np.float64, np.int64] else 'first')
            for col in df.columns if col not in exclude_columns}

def allforone_endpoint_split(df: pd.DataFrame, split_size: float, smiles_column: str,
                             endpoint_date_columns: Dict[str, str], date_aggregation: str = 'min', 
                             exclude_columns: List[str] = []) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into training and testing sets based on the specified aggregation of endpoint dates.

    Parameters:
    - df: Input DataFrame containing compound data.
    - split_size: Proportion of the dataset to include in the test split.
    - smiles_column: Column name containing the SMILES representation of compounds.
    - endpoint_date_columns: Dictionary mapping endpoint names to their respective date columns.
    - date_aggregation: Method to aggregate date ('min' for earliest, 'max' for latest, 'mean' for average).
    - exclude_columns: List of columns to exclude from aggregation (default is empty).

    Returns:
    - Tuple containing the training and testing DataFrames.
    """
    # Convert specified date columns to datetime objects
    for column in endpoint_date_columns.values():
        df[column] = pd.to_datetime(df[column], errors='coerce')

    # Aggregate dates per specified method
    if date_aggregation == 'mean':
        df['tmp_date'] = df[list(endpoint_date_columns.values())].mean(axis=1, skipna=True)
    elif date_aggregation == 'max':
        df['tmp_date'] = df[list(endpoint_date_columns.values())].max(axis=1, skipna=True)
    else:
        df['tmp_date'] = df[list(endpoint_date_columns.values())].min(axis=1, skipna=True)

    # Group by SMILES column and find the corresponding date per compound
    compound_dates = df.groupby(smiles_column)['tmp_date'].min().sort_values().reset_index()

    # Calculate split index for test set based on split size and sorted dates
    split_index = int(len(compound_dates) * split_size)
    test_compounds = set(compound_dates.iloc[:split_index][smiles_column])

    # Create initial splits based on test compounds
    df_test = df[df[smiles_column].isin(test_compounds)]
    df_train = df[~df[smiles_column].isin(test_compounds)]

    # Determine aggregation rules for other columns
    aggregation_rules = {
        col: 'mean' if df[col].dtype in [np.float64, np.int64] and col not in exclude_columns + [smiles_column]
        else 'first'
        for col in df.columns if col not in ['tmp_date'] + list(endpoint_date_columns.values())
    }

    # Group by SMILES and apply aggregation to finalize train and test DataFrames
    all_train_df = df_train.groupby(smiles_column, as_index=False).agg(aggregation_rules)
    all_test_df = df_test.groupby(smiles_column, as_index=False).agg(aggregation_rules)

    # Remove temporary column
    df.drop(columns=['tmp_date'], inplace=True)

    return all_train_df, all_test_df
def allforone_folds_endpoint_split(df: DataFrame, num_folds: int, smiles_column: str, 
                                    endpoint_date_columns: Dict[str, str], chemprop: bool, 
                                    save_path: str, aggregation: str, 
                                    feature_columns: List[str] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Splits a DataFrame into multiple train/test sets (folds) with progressively increasing training data.
    
    For each fold, the test set size is reduced progressively based on the fold index, and the split is 
    determined by aggregating endpoint dates according to the specified aggregation method. Optionally, if 
    chemprop processing is enabled, the function extracts the specified feature columns to save separate 
    feature and target CSV files for each fold.
    
    Args:
        df: The input DataFrame containing compound data.
        num_folds: The number of folds for the cross-validation split.
        smiles_column: Name of the column containing compound identifiers (e.g., SMILES strings).
        endpoint_date_columns: Dictionary mapping endpoint names to their respective date columns.
        chemprop: Boolean flag indicating whether the output is intended for chemprop processing.
        save_path: Directory path to save the resulting CSV files for each fold.
        aggregation: Method to aggregate date values; must be one of 'first', 'last', or 'avg'.
        feature_columns: List of feature column names to extract (required if chemprop is True).
    
    Returns:
        A list of tuples, each containing the training and testing DataFrames for a fold.
    """
    if aggregation not in ['first', 'last', 'avg']:
        raise ValueError("Aggregation method must be 'first', 'last', or 'avg'.")
    cv_splits = []
    for fold in range(1, num_folds + 1):
        # Decrease the test size progressively: later folds have larger training sets.
        split_size = 1 - (fold / (num_folds + 1))
        
        # Generate the train/test split using the allforone_endpoint_split function.
        train_df, test_df = allforone_endpoint_split(df, split_size, smiles_column, endpoint_date_columns, aggregation)

        if chemprop:
            if feature_columns is None:
                raise ValueError("feature_columns must be provided when chemprop is True.")
            # Extract features using the helper function.
            train_features = extract_features(train_df, smiles_column, feature_columns)
            test_features = extract_features(test_df, smiles_column, feature_columns)
            # Define target sets including the smiles and endpoint columns.
            train_targets = train_df[[smiles_column] + list(endpoint_date_columns.keys())]
            test_targets = test_df[[smiles_column] + list(endpoint_date_columns.keys())]

            # Save features and targets as CSV files.
            train_features.to_csv(os.path.join(save_path, f'train_features_fold{fold}.csv'), index=False)
            test_features.to_csv(os.path.join(save_path, f'test_features_fold{fold}.csv'), index=False)
            train_targets.to_csv(os.path.join(save_path, f'train_targets_fold{fold}.csv'), index=False)
            test_targets.to_csv(os.path.join(save_path, f'test_targets_fold{fold}.csv'), index=False)
        else:
            # Save complete train/test DataFrames.
            desired_columns = [smiles_column] + list(endpoint_date_columns.keys()) + feature_columns

            train_df = train_df[desired_columns]
            test_df = test_df[desired_columns]

            train_df.to_csv(os.path.join(save_path, f'train_fold{fold}.csv'), index=False)
            test_df.to_csv(os.path.join(save_path, f'test_fold{fold}.csv'), index=False)

        cv_splits.append((train_df, test_df))

    return cv_splits

def leaky_endpoint_split(df: pd.DataFrame, split_size: float, smiles_column: str, endpoint_date_columns: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a DataFrame by identifying test compounds and splitting the DataFrame based on the earliest date across multiple endpoints,
    each with its own date column. This method ensures consistent test set selection across different endpoints by converting all date columns
    to datetime format before aggregation.

    Args:
        df: DataFrame to be processed.
        split_size: Fraction of the DataFrame to include in the test set.
        smiles_column: Name of the column containing compound identifiers.
        endpoint_date_columns: Dictionary mapping endpoint names to their respective date columns.

    Returns:
        Tuple containing the training and testing DataFrames.
    """
    
    # Convert all date columns to datetime
    for date_column in endpoint_date_columns.values():
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Create a new column for the earliest date across all endpoints
    df['earliest_date'] = df[list(endpoint_date_columns.values())].min(axis=1)
    
    # Filter out rows where the earliest date is null
    df_filtered = df[df['earliest_date'].notnull()]

    # Sorting data to pick the most recent entries as the test set
    df_sorted = df_filtered.sort_values(by='earliest_date', ascending=False)
    
    test_size = int(len(df_sorted) * split_size)
    test_indices = df_sorted.iloc[:test_size].index
    train_indices = df_sorted.iloc[test_size:].index
    
    # Create test and train dataframes
    test_df = df.loc[test_indices].copy()
    train_df = df.loc[train_indices].copy()

    # Define aggregation rules without earliest_date
    aggregation_rules = {col: 'mean' if df[col].dtype in [np.float64, np.int64] else 'first' for col in df.columns if col not in  ['earliest_date'] and col != smiles_column}
    
    # Drop the 'earliest_date' column before aggregation
    test_df.drop(columns=['earliest_date'], inplace=True)
    train_df.drop(columns=['earliest_date'], inplace=True)

    # Aggregation to ensure no duplicate SMILES in the sets
    all_train_df = train_df.groupby(smiles_column).agg(aggregation_rules).reset_index()
    all_test_df = test_df.groupby(smiles_column).agg(aggregation_rules).reset_index()

    return all_train_df, all_test_df

def expand_df_to_endpoints(df: pd.DataFrame, endpoint_date_columns: Dict[str, str]) -> pd.DataFrame:
    """
    Transforms the DataFrame so that each row corresponds to a single endpoint,
    setting null for other endpoint values. Rows where all endpoint values are None are not included.
    It also removes any duplicate rows and rows where all endpoint values are None post transformation.
    
    Args:
        df: Original DataFrame.
        endpoint_date_columns: Dictionary mapping endpoint values to their respective date columns.
    
    Returns:
        DataFrame with expanded rows for each endpoint, maintaining NaN where data is missing,
        removing rows where all endpoints are None, and eliminating duplicates.
    """
    rows = []
    for idx, row in df.iterrows():
        if any(row[value_col] is not None for value_col in endpoint_date_columns.keys()):
            for value_col, date_col in endpoint_date_columns.items():
                new_row = {col: None for col in df.columns}  # Initialize all columns to None
                new_row.update(row)  # Update with original data

                # Reset other endpoints and their dates to None
                for vc, dc in endpoint_date_columns.items():
                    new_row[vc] = None
                    new_row[dc] = None

                new_row[value_col] = row[value_col]  # Set current endpoint
                new_row[date_col] = row[date_col]    # Set current endpoint's date

                rows.append(new_row)

    new_df = pd.DataFrame(rows)
    new_df = new_df.drop_duplicates().reset_index(drop=True)

    # Remove rows where all endpoint values are None
    endpoints = list(endpoint_date_columns.keys())  # List of all endpoint columns
    new_df = new_df.dropna(subset=endpoints, how='all')  # Drop rows where all endpoints are NaN

    return new_df

def extract_features(df: pd.DataFrame, smiles_column: str, feature_columns: List[str]) -> pd.DataFrame:
    """
    Extract features from the DataFrame.

    Args:
        df: The original DataFrame.
        smiles_column: Column name containing the SMILES strings.
        feature_columns: List of columns to be used as features.

    Returns:
        A DataFrame containing the SMILES and features.
    """
    return df[[smiles_column] + feature_columns]

def leaky_folds_endpoint_split(df: DataFrame, num_folds: int, smiles_column: str, endpoint_date_columns: Dict[str, str], chemprop: bool, save_path: str, feature_columns: List[str] = None) -> List[Tuple[DataFrame, DataFrame]]: 
    """
    Process a DataFrame by splitting it into multiple train/test sets for cross-validation, with the training set growing progressively.
    The size of the test set decreases with each fold, increasing the training data size.

    Args:
        df: DataFrame to be processed.
        num_folds: Number of folds for cross-validation.
        smiles_column: Name of the column containing compound identifiers.
        endpoint_date_columns: Dictionary of endpoint names to their respective date columns.
        chemprop: Boolean to indicate if data is for chemprop.
        save_path: Path to save the resulting dataframes.
        feature_columns: List of columns to be used as features.
    Returns:
        List of tuples containing training and testing DataFrames for each fold.
    """
    splits = []
    df = expand_df_to_endpoints(df, endpoint_date_columns)

    # test comment
    for fold in range(1, num_folds + 1 ):
        split_size = 1 - (fold / (num_folds + 1))  # Decrease the test size progressively
        
        # Use the leaky_endpoint_split function to generate each fold's split
        train_df, test_df = leaky_endpoint_split(df, split_size, smiles_column, endpoint_date_columns)
        
        if chemprop:
            train_features = extract_features(train_df, smiles_column, feature_columns)
            test_features = extract_features(test_df, smiles_column, feature_columns)

            # Include smiles_column in the targets
            train_targets = train_df[[smiles_column] + list(endpoint_date_columns.keys())]
            test_targets = test_df[[smiles_column] + list(endpoint_date_columns.keys())]

            # Save features and targets
            train_features.to_csv(os.path.join(save_path, f'train_features_fold{fold}.csv'), index=False)
            test_features.to_csv(os.path.join(save_path, f'test_features_fold{fold}.csv'), index=False)
            train_targets.to_csv(os.path.join(save_path, f'train_targets_fold{fold}.csv'), index=False)
            test_targets.to_csv(os.path.join(save_path, f'test_targets_fold{fold}.csv'), index=False)
        else:
            desired_columns = [smiles_column] + list(endpoint_date_columns.keys()) + feature_columns

            train_df = train_df[desired_columns]
            test_df = test_df[desired_columns]

            train_df.to_csv(os.path.join(save_path, f'train_fold{fold}.csv'), index=False)
            test_df.to_csv(os.path.join(save_path, f'test_fold{fold}.csv'), index=False)

        splits.append((train_df, test_df))

    return splits
