import unittest
import sys
sys.path.append('C:\\Users\\philip.ivers.ohlsson.TSS\\Documents\\ivers')
import os
import numpy as np
from ivers.scaffold import balanced_scaffold
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_smiles_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

class TestBalancedScaffoldCV(unittest.TestCase):
    def setUp(self):
        # Create a test dataset
        csv_path = "C:\\Users\\philip.ivers.ohlsson.TSS\\Downloads\\archive (1)\\SMILES_Big_Data_Set.csv"
        self.df = generate_smiles_from_csv(csv_path)
        self.smiles_column = 'SMILES'
        self.n_splits = 3  # Number of folds

    def test_function_savepahts(self):
        # create test outputs folder
        os.mkdir('test_outputs')
        _, results, fold_sizes = balanced_scaffold(self.df, 
                                                      endpoints=[], 
                                                      smiles_column=self.smiles_column, 
                                                      n_splits=self.n_splits, 
                                                      random_state=42, 
                                                      chemprop=True,
                                                      save_path='test_outputs/')

        # Check if files are created
        for i in range(self.n_splits):
            self.assertTrue(os.path.exists(f'test_outputs/train_features_fold{i + 1}.csv'))
            self.assertTrue(os.path.exists(f'test_outputs/train_targets_fold{i + 1}.csv'))
            self.assertTrue(os.path.exists(f'test_outputs/test_features_fold{i + 1}.csv'))
            self.assertTrue(os.path.exists(f'test_outputs/test_targets_fold{i + 1}.csv'))

        # Clean up test files after test run remove based on number of splits
        for i in range(self.n_splits):
            os.remove(f'test_outputs/train_features_fold{i + 1}.csv')
            os.remove(f'test_outputs/train_targets_fold{i + 1}.csv')
            os.remove(f'test_outputs/test_features_fold{i + 1}.csv')
            os.remove(f'test_outputs/test_targets_fold{i + 1}.csv')
        os.rmdir('test_outputs')

    def test_function_execution(self):
        # Test the execution of the function
        _, results, fold_sizes = balanced_scaffold(self.df, 
                                                      endpoints=[], 
                                                      smiles_column=self.smiles_column, 
                                                      n_splits=self.n_splits, 
                                                      random_state=42, 
                                                      save_path=None)

        # Test number of results
        self.assertEqual(len(results), self.n_splits, "Should create exactly n_splits folds.")
        # check number of data points is same as original 
        self.assertEqual(sum([len(train_df) + len(test_df) for train_df, test_df in results])/self.n_splits, len(self.df), "Should have same number of data points as original dataset.")
        
if __name__ == '__main__':
    unittest.main()
