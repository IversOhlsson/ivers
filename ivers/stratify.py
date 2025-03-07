"""
Filename: stratify.py

Function "stratify_endpoint" for generate train / test taking into account endpoint distribution 

Function "stratify_split_and_cv" for generate cross-validation that takes into account the endpoint distribution 

Author: Philip Ivers Ohlsson
License: MIT License
"""

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def aggregate_dataframe(df: DataFrame, smiles_column: str, 
                        exclude_columns: Optional[List[str]] = None, 
                        aggregation_rules: Optional[Dict[str, str]] = None) -> DataFrame:
    """
    Aggregate the dataframe based on specified rules, excluding some columns.

    Args:
        df: The original DataFrame to be aggregated.
        smiles_column: Name of the column containing SMILES strings for grouping.
        exclude_columns: List of column names to exclude from aggregation, defaults to an empty list.
        aggregation_rules: Dictionary specifying how to aggregate columns, defaults to an empty dictionary.

    Returns:
        Aggregated DataFrame.
    """
    if exclude_columns is None:
        exclude_columns = []
    if aggregation_rules is None:
        aggregation_rules = {}

    # Define default aggregation based on column data types
    aggregation = {col: 'mean' if df[col].dtype in [np.float64, np.int64] and col not in exclude_columns and col != smiles_column 
                   else 'first' for col in df.columns}

    # Update with provided aggregation rules
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
    
    df['endpoint_stratify_key'] = df.apply(
        lambda row: '-'.join(col_abbreviations[col] for col in order if pd.notnull(row[col])), axis=1)

    if label_column:
        df['StratifyKey'] = df.apply(
            lambda row: row['endpoint_stratify_key'] + '-' + str(row[label_column]) if pd.notnull(row[label_column]) else row['endpoint_stratify_key'],
            axis=1)
    else:
        df['StratifyKey'] = df['endpoint_stratify_key']

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
    #small_classes = class_counts[class_counts < 2].index
    required_min_samples = 2 * (n_splits + 1)

    small_classes = value_counts[value_counts < required_min_samples].index.tolist()
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

def stratify_endpoint_cv(df: pd.DataFrame, 
                         order: List[str], 
                         smiles_column: str, 
                         exclude_columns: List[str], 
                         aggregation_rules: Dict[str, str], 
                         n_splits: int, 
                         random_state: int = 42, 
                         label_column: Optional[str] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]]:
    """
    Manage the distribution of the DataFrame by aggregating, generating a stratify key with optional label considerations, and performing k-fold cross-validation. Additionally returns column abbreviation mappings for each fold.

    Args:
        df: The original DataFrame.
        order: Ordered list of endpoint columns for stratification.
        smiles_column: Column name containing the SMILES strings for aggregation grouping.
        exclude_columns: Columns to exclude from mean aggregation.
        aggregation_rules: Specific aggregation rules for some columns.
        n_splits: Number of folds for cross-validation.
        random_state: Random state for reproducibility.
        label_column: Optional column for additional grouping in stratification.
    Returns:
        A list of tuples, each containing the training and testing datasets for a fold, and a dictionary of column abbreviations.
    """
    df_processed = aggregate_dataframe(df, smiles_column, exclude_columns, aggregation_rules)
    df_processed, col_abbreviations = create_stratify_key(df_processed, order, label_column)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    for train_index, test_index in skf.split(df_processed, df_processed['StratifyKey']):
        train_df = df_processed.iloc[train_index]
        test_df = df_processed.iloc[test_index]
        splits.append((train_df, test_df, col_abbreviations))
    
    return splits

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import List, Dict, Tuple, Optional

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

def stratify_split_and_cv(df: pd.DataFrame, 
                          endpoints: List[str], 
                          smiles_column: str, 
                          exclude_columns: List[str], 
                          aggregation_rules: Dict[str, str], 
                          test_size: float, 
                          n_splits: int, 
                          random_state: int = 1337, 
                          label_column: Optional[str] = None,
                          chemprop: bool = False,
                          save_path: str = './', 
                          feature_columns: List[str] = None) -> Tuple[pd.DataFrame, List[Tuple[pd.DataFrame, pd.DataFrame]], Dict[str, str]]:

    """
    Split the data into a test set and perform cross-validation on the remaining data, ensuring stratified distribution of the endpoints.
    The resulting dataframes will be saved at the specified path with suffixes for different folds.

    Args:
        df: The original DataFrame.
        endpoints: Ordered list of endpoint columns for stratification.
        smiles_column: Column name containing the SMILES strings for aggregation grouping.
        exclude_columns: Columns to exclude from mean aggregation.
        aggregation_rules: Specific aggregation rules for some columns.
        test_size: Proportion of the dataset to include in the test split.
        n_splits: Number of folds for cross-validation.
        random_state: Random state for reproducibility.
        label_column: Optional column for additional grouping in stratification.
        chemprop: Boolean to indicate if data is for chemprop.
        save_path: Path to save the resulting dataframes.

    Returns:
        A tuple containing the test dataset, a list of tuples (each with training and validation datasets for a fold), and a dictionary of column abbreviations.
    """
    df_processed = aggregate_dataframe(df, smiles_column, exclude_columns, aggregation_rules)
    df_processed, col_abbreviations = create_stratify_key(df_processed, endpoints, label_column)
    
    # Check for small classes -------------------------------- #
    value_counts = df_processed['StratifyKey'].value_counts()
    print(value_counts)
    if any(value_counts < 2*(n_splits + 1)):
        small_classes = value_counts[value_counts < n_splits].index.tolist()
        raise ValueError(f"Classes {small_classes} not enought data points, {2 * (n_splits+1)} needed for {n_splits} splits.")
    # -------------------------------------------------------- #

    remaining_df, test_df = train_test_split(
        df_processed, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df_processed['StratifyKey']
    )
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(remaining_df, remaining_df['StratifyKey'])):
        train_df = remaining_df.iloc[train_index]
        val_df = remaining_df.iloc[val_index]
        splits.append((train_df, val_df))
        
        # Extracting and saving features and targets for Chemprop
        if chemprop:
            if feature_columns is None:
                feature_columns = [col for col in df.columns if col not in exclude_columns + [smiles_column, 'StratifyKey']]

            train_features = extract_features(train_df, smiles_column, feature_columns)
            val_features = extract_features(val_df, smiles_column, feature_columns)
            train_targets = train_df[endpoints]
            val_targets = val_df[endpoints]
            
            # Save features and targets
            train_features.to_csv(os.path.join(save_path, f'train_features_fold{fold+1}.csv'), index=False)
            val_features.to_csv(os.path.join(save_path, f'val_features_fold{fold+1}.csv'), index=False)
            train_targets.to_csv(os.path.join(save_path, f'train_targets_fold{fold+1}.csv'), index=False)
            val_targets.to_csv(os.path.join(save_path, f'val_targets_fold{fold+1}.csv'), index=False)
        else:
            train_df.to_csv(os.path.join(save_path, f'train_fold{fold+1}.csv'), index=False)
            val_df.to_csv(os.path.join(save_path, f'val_fold{fold+1}.csv'), index=False)
    
    if chemprop:
        test_features = extract_features(test_df, smiles_column, feature_columns)
        test_targets = test_df[endpoints]
        test_features.to_csv(os.path.join(save_path, 'test_features.csv'), index=False)
        test_targets.to_csv(os.path.join(save_path, 'test_targets.csv'), index=False)
    else:
        test_df.to_csv(os.path.join(save_path, 'test.csv'), index=False)
    
    return test_df, splits, col_abbreviations
