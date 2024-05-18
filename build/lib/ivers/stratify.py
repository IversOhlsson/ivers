import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def aggregate_dataframe(df: DataFrame, smiles_column: str, exclude_columns: List[str], aggregation_rules: Dict[str, str]) -> DataFrame:
    """
    Aggregate the dataframe based on specified rules, excluding some columns.

    Args:
        df: The original DataFrame to be aggregated.
        smiles_column: Name of the column containing SMILES strings for grouping.
        exclude_columns: List of column names to exclude from aggregation.
        aggregation_rules: Dictionary specifying how to aggregate columns.

    Returns:
        Aggregated DataFrame.
    """
    aggregation = {col: 'mean' if df[col].dtype in [np.float64, np.int64] and col not in exclude_columns and col != smiles_column 
                else 'first' for col in df.columns}
    aggregation.update(aggregation_rules)
    return df.groupby(smiles_column, as_index=False).agg(aggregation)

def create_stratify_key(df: DataFrame, order: List[str], label_column: Optional[str] = None) -> Tuple[DataFrame, Dict[str, str]]:
    """
    Generate stratify key columns in the DataFrame based on the order of endpoint columns and an optional label column.
    Adds 'StratifyKey' for full stratification and 'endpoint_stratify_key' for endpoints only.

    Args:
        df: DataFrame to operate on.
        order: Ordered list of endpoint columns.
        label_column: Optional column name to include in the stratify key for further grouping.

    Returns:
        Tuple containing the modified DataFrame and a dictionary mapping original column names to their abbreviations.
    """
    col_abbreviations = {col: chr(65 + i) for i, col in enumerate(order)}
    
    # Create endpoint_stratify_key which includes only the endpoint columns
    df['endpoint_stratify_key'] = df.apply(
        lambda row: '-'.join(col_abbreviations[col] for col in order if pd.notnull(row[col])), axis=1)

    # Create StratifyKey which includes both endpoints and the label column if present
    df['StratifyKey'] = df['endpoint_stratify_key'].copy()
    if label_column:
        df['StratifyKey'] = df.apply(
            lambda row: row['endpoint_stratify_key'] + '-' + str(row[label_column]) if pd.notnull(row[label_column]) else row['endpoint_stratify_key'],
            axis=1)
    
    return df, col_abbreviations

def split_dataframe(df: DataFrame, test_size: float, random_state: int = 42) -> Tuple[DataFrame, DataFrame]:
    """
    Split the DataFrame into training and testing datasets, stratifying by 'StratifyKey'.

    Args:
        df: DataFrame to split.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random state for reproducibility.
        include_labels: Labels to exclusively include in the test dataset.
        exclude_labels: Labels to exclude from the test dataset.

    Returns:
        A tuple containing the training and testing datasets.
    """
    
    class_counts = df['StratifyKey'].value_counts()
    small_classes = class_counts[class_counts < 2].index
    # ----------------- Concat Small Classes ----------------- #
    if small_classes.any():
        logging.info(f"Small classes found: {small_classes.tolist()}, merging into a single class for stratification.")
        df.loc[df['StratifyKey'].isin(small_classes), 'StratifyKey'] = 'ConcatSmall'
    # -------------------------------------------------------- #

    # ----------------- Remove small classes ----------------- #
    #df_filtered = df[~df['StratifyKey'].isin(small_classes)]
    # -------------------------------------------------------- #

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['StratifyKey'])
    return df_train, df_test

def stratify_endpoint(df: DataFrame, order: List[str], 
                        smiles_column: str, 
                        exclude_columns: List[str], 
                        aggregation_rules: Dict[str, str], 
                        test_size: float, 
                        random_state: int = 42, 
                        label_column: Optional[str] = None) -> Tuple[DataFrame, DataFrame, Dict[str, str]]:
    """
    Manage the distribution of the DataFrame by aggregating, generating a stratify key with optional label considerations, and splitting into train/test datasets. Additionally returns column abbreviation mapping.

    Args:
        df: The original DataFrame.
        order: Ordered list of endpoint columns for stratification.
        smiles_column: Column name containing the SMILES strings for aggregation grouping.
        exclude_columns: Columns to exclude from mean aggregation.
        aggregation_rules: Specific aggregation rules for some columns.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random state for reproducibility.
        label_column: Optional column for additional grouping in stratification.
    Returns:
        A tuple containing the training and testing datasets, and a dictionary of column abbreviations.
    """
    df_processed = aggregate_dataframe(df, smiles_column, exclude_columns, aggregation_rules)
    df_processed, col_abbreviations = create_stratify_key(df_processed, order, label_column)
    train_df, test_df = split_dataframe(df_processed, test_size, random_state)
    
    return train_df, test_df, col_abbreviations

