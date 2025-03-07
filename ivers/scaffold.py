"""
Filename: scaffold.py

Function "balanced_scaffold_cv" scaffold splits 

Author: Philip Ivers Ohlsson
License: MIT License
"""

import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

def find_balanced_folds_dp(n_splits, scaffold_to_indices):
    # Sort groups by size in descending order
    sorted_scaffolds = sorted(scaffold_to_indices.items(), key=lambda x: len(x[1]), reverse=True)

    # Calculate total number of indices
    total_indices = sum(len(indices) for _, indices in sorted_scaffolds)
    
    # Initialize DP table
    # dp[i][j] will hold the minimum largest fold size achievable with the first i items and j folds
    dp = [[float('inf')] * (n_splits + 1) for _ in range(len(sorted_scaffolds) + 1)]
    scaffold_sum = [0] * (len(sorted_scaffolds) + 1)
    
    # Calculate prefix sums of scaffold sizes
    for i in range(1, len(sorted_scaffolds) + 1):
        scaffold_sum[i] = scaffold_sum[i - 1] + len(sorted_scaffolds[i - 1][1])
    
    # Base case: 0 items into any number of folds has cost 0
    for j in range(n_splits + 1):
        dp[0][j] = 0
    
    # Fill DP table
    for i in range(1, len(sorted_scaffolds) + 1):
        for k in range(1, n_splits + 1):
            # We consider all possible ways to form the k-th fold
            for j in range(0, i):
                max_fold_size = max(dp[j][k - 1], scaffold_sum[i] - scaffold_sum[j])
                if max_fold_size < dp[i][k]:
                    dp[i][k] = max_fold_size
    
    # Recover the partitions from the DP table
    folds = [[] for _ in range(n_splits)]
    current_fold = n_splits
    last = len(sorted_scaffolds)
    
    while current_fold > 0:
        for i in range(last):
            if dp[i][current_fold - 1] <= dp[last][current_fold] - (scaffold_sum[last] - scaffold_sum[i]):
                # Assign scaffolds i+1 to last to the current fold
                for index in range(i + 1, last + 1):
                    folds[current_fold - 1].extend(sorted_scaffolds[index - 1][1])
                last = i
                current_fold -= 1
                break 
    
    return folds

def find_balanced_folds(n_splits, scaffold_to_indices):
    # Sort groups by size in descending order
    sorted_scaffolds = sorted(scaffold_to_indices.items(), key=lambda x: len(x[1]), reverse=True)

    # Initialize empty folds
    folds = [[] for _ in range(n_splits)]

    # Iterate over sorted scaffolds
    for scaffold, indices in sorted_scaffolds:
        # Find the smallest fold
        smallest_fold = min(folds, key=len)
        # Add indices to the smallest fold
        smallest_fold.extend(indices)

    return folds

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

def balanced_scaffold(df: pd.DataFrame, 
                      endpoints: List[str], 
                      smiles_column: str, 
                      n_splits: int, 
                      random_state: int = 42,
                      feature_columns: List[str] = None,
                      exclude_columns: List[str] = [], 
                      chemprop: bool = False,
                      save_path: str = './') -> Tuple[pd.DataFrame, List[Tuple[pd.DataFrame, pd.DataFrame]], Dict[str, int]]:

    invalid_smiles_indices = []

    scaffold_to_indices = defaultdict(list)
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_column])
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            scaffold_to_indices[scaffold_smiles].append(idx)
        else:
            invalid_smiles_indices.append(idx)

    if invalid_smiles_indices:
        print(f"Removing {len(invalid_smiles_indices)} invalid SMILES:")
        print(df.loc[invalid_smiles_indices, smiles_column].tolist())
        df = df.drop(index=invalid_smiles_indices).reset_index(drop=True)

    folds_indices = find_balanced_folds(n_splits, scaffold_to_indices)

    # Continue as before with creating DataFrame splits...
    results = []
    all_indices = set(range(len(df)))
    for fold in folds_indices:
        test_indices = set(fold)
        train_indices = all_indices - test_indices
        train_df = df.iloc[list(train_indices)]
        test_df = df.iloc[list(test_indices)]
        results.append((train_df, test_df))

    # The remainder of your existing function here...

    return df, results, {i: len(fold) for i, fold in enumerate(folds_indices)}
