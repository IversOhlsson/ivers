import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_aggregation_rules(df: pd.DataFrame, exclude_columns: List[str]) -> dict:
    """
    Determine aggregation rules for the DataFrame columns based on their data types.
    Args:
        df: DataFrame to define aggregation rules for.
        exclude_columns: List of columns to exclude from aggregation.
    Returns:
        A dictionary with column names as keys and aggregation methods as values.
    """
    return {col: ('mean' if df[col].dtype in [np.float64, np.int64] else 'first') for col in df.columns if col not in exclude_columns}

def aggregate_data(df: pd.DataFrame, smiles_column: str) -> pd.DataFrame:
    """
    Aggregate the DataFrame based on predefined rules and group by the SMILES column.
    Args:
        df: DataFrame to aggregate.
        smiles_column: Column name for SMILES identifiers.
    Returns:
        Aggregated DataFrame.
    """
    aggregation_rules = get_aggregation_rules(df, [smiles_column])
    return df.groupby(smiles_column, as_index=False).agg(aggregation_rules)

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
