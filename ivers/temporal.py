import pandas as pd
from pandas import DataFrame
import logging
from typing import Tuple, List, Dict
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_aggregation_rules(df: DataFrame, exclude_columns: List[str]) -> dict:
    """
    Determine aggregation rules for the DataFrame columns based on their data types.
    """
    return {col: ('mean' if df[col].dtype in [np.float64, np.int64] else 'first')
            for col in df.columns if col not in exclude_columns}


def allforfree_endpoint_split(df_list: List[pd.DataFrame], split_size: float, smiles_column: str, date_column: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Process a list of DataFrames by identifying test compounds and splitting each DataFrame based on a unified test compound list.
    
    Args:
        df_list: List of DataFrames to be processed.
        split_size: Fraction of each DataFrame to include in the test set.
        smiles_column: Name of the column containing compound identifiers.
        date_column: Name of the column containing the dates of publication or experiment.
    
    Returns:
        Tuple containing lists of training and testing DataFrames.
    """
    all_test_compounds = set()
    initial_test_compounds_counts = []
    
    # First pass: Identify all test compounds across DataFrames
    for df in df_list:
        test_size = int(len(df) * split_size)
        df[date_column] = pd.to_datetime(df[date_column])
        df_sorted = df.sort_values(by=date_column, ascending=False)
        new_test_compounds = set(df_sorted.iloc[:test_size][smiles_column].unique())
        initial_test_compounds_counts.append(len(new_test_compounds))
        all_test_compounds.update(new_test_compounds)
    
    # Second pass: Split DataFrames into training and testing using the unified test compound set
    train_dfs = []
    test_dfs = []
    for df in df_list:
        df_test = df[df[smiles_column].isin(all_test_compounds)]
        df_train = df[~df[smiles_column].isin(all_test_compounds)]
        train_dfs.append(df_train)
        test_dfs.append(df_test)
    
    # Optionally print additional info for each DataFrame
    for i, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
        additional_compounds = len(test_df) - initial_test_compounds_counts[i]
        print(f"DataFrame {i}: Additional compounds due to unified test set = {additional_compounds}, Training set size = {len(train_df)}, Test set size = {len(test_df)}")
    
    return train_dfs, test_dfs


def allforfree_folds_endpoint_split(df: pd.DataFrame, num_folds: int, smiles_column: str, endpoint_date_columns: Dict[str, str]) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Process a DataFrame by splitting it into multiple train/test sets for cross-validation, with the training set growing progressively.
    Args:
        df: DataFrame to be processed.
        num_folds: Number of folds for cross-validation.
        smiles_column: Name of the column containing compound identifiers.
        endpoint_date_columns: Dictionary of endpoint names to their respective date columns.
    Returns:
        List of tuples containing training and testing DataFrames for each fold.
    """
    cv_splits = []
    for fold in range(1, num_folds + 1):
        split_size = 1 - (fold / num_folds)  # Decrease the test size progressively
        train_df, test_df = allforfree_endpoint_split(df, split_size, smiles_column, endpoint_date_columns)
        cv_splits.append((train_df, test_df))
    return cv_splits


def leaky_endpoint_split(df: DataFrame, split_size: float, smiles_column: str, endpoint_date_columns: Dict[str, str]) -> Tuple[DataFrame, DataFrame]:
    """
    Process a DataFrame by identifying test compounds and splitting the DataFrame for multiple endpoints each with its own date column.

    Args:
        df: DataFrame to be processed.
        split_size: Fraction of the DataFrame to include in the test set for each endpoint.
        smiles_column: Name of the column containing compound identifiers.
        endpoint_date_columns: Dictionary of endpoint names to their respective date columns.

    Returns:
        Tuple containing the training and testing DataFrames.
    """
    
    test_compounds_by_endpoint = {}
    all_test_compounds = set()
    
    # Identify test compounds for each endpoint
    for endpoint, date_column in endpoint_date_columns.items():
        df_sorted = df.sort_values(by=date_column, ascending=False)
        test_size = int(len(df_sorted) * split_size)
        new_test_compounds = set(df_sorted.iloc[:test_size][smiles_column].unique())
        all_test_compounds.update(new_test_compounds)
        test_compounds_by_endpoint[endpoint] = new_test_compounds

    train_dfs = []
    test_dfs = []
    
    # Split the DataFrame into training and testing sets for each endpoint
    for endpoint in endpoint_date_columns.keys():
        test_compounds = test_compounds_by_endpoint[endpoint]
        test_df = df[df[smiles_column].isin(test_compounds)]
        train_df = df[~df[smiles_column].isin(test_compounds)]
        train_dfs.append(train_df)
        test_dfs.append(test_df)

    # Concatenate all training and testing DataFrames
    all_train_df = pd.concat(train_dfs, axis=0, ignore_index=True, sort=False)
    all_test_df = pd.concat(test_dfs, axis=0, ignore_index=True, sort=False)

    # Aggregation rules to apply
    aggregation_rules = {col: 'mean' for col in all_train_df.columns if col != smiles_column}

    # Group by SMILES and apply aggregation
    all_train_df = all_train_df.groupby(smiles_column, as_index=False).agg(aggregation_rules)
    all_test_df = all_test_df.groupby(smiles_column, as_index=False).agg(aggregation_rules)

    return all_train_df, all_test_df


def leaky_folds_endpoint_split(df: DataFrame, num_folds: int, smiles_column: str, endpoint_date_columns: Dict[str, str]) -> List[Tuple[DataFrame, DataFrame]]:
    """
    Process a DataFrame by splitting it into multiple train/test sets for cross-validation, with the training set growing progressively.
    The size of the test set decreases with each fold, increasing the training data size.

    Args:
        df: DataFrame to be processed.
        num_folds: Number of folds for cross-validation.
        smiles_column: Name of the column containing compound identifiers.
        endpoint_date_columns: Dictionary of endpoint names to their respective date columns.

    Returns:
        List of tuples containing training and testing DataFrames for each fold.
    """

    cv_splits = []

    for fold in range(1, num_folds + 1):
        split_size = 1 - (fold / num_folds)  # Decrease the test size progressively

        # Use the leaky_endpoint_split function to generate each fold's split
        train_df, test_df = leaky_endpoint_split(df, split_size, smiles_column, endpoint_date_columns)

        # Append the current fold's train and test DataFrames to the list
        cv_splits.append((train_df, test_df))

    return cv_splits
