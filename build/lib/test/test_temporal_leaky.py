import unittest
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import List, Tuple
from ivers.temporal import leaky_endpoint_split, leaky_folds_endpoint_split
import os
import shutil
import tempfile

class TestEndpointSplits(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame with multiple endpoints, each with its own date column
        data = {
            'SMILES': ['C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
            'MISC': ['C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
            'dummy_exclude_column1': ['C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
            'Activity': [1.2, 3.4, 5.6, 7.8, 9.0, None, None, None, None, None],
            'Activity_Date': [
                datetime(2020, 1, 1),
                datetime(2020, 2, 1),
                datetime(2020, 3, 1),
                datetime(2020, 4, 1),
                datetime(2020, 5, 1),
                None, None, None, None, None
            ],
            'Toxicity': [None, None, None, None, None, 0.5, 2.5, 5.0, 7.5, None],
            'Toxicity_Date': [
                None, None, None, None, None,
                datetime(2020, 1, 15),
                datetime(2020, 2, 15),
                datetime(2020, 3, 15),
                datetime(2020, 4, 15),
                None
            ]
        }
        self.df = pd.DataFrame(data)
        self.split_size = 0.4
        self.smiles_column = 'SMILES'
        self.endpoint_date_columns = {
            'Activity': 'Activity_Date',
            'Toxicity': 'Toxicity_Date'
        }
        self.chemprop = True
        self.num_folds = 2
        self.save_path = tempfile.mkdtemp()
        #self.save_path = './'
        self.exclude_columns = ['dummy_exclude_column1']

    def tearDown(self):
        # Clean up the directory after tests
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)

    def test_leaky_endpoint_split(self):
        # Test the basic split function for a single DataFrame with multiple date columns
        train_df, test_df = leaky_endpoint_split(self.df, self.split_size, self.smiles_column, self.endpoint_date_columns, self.exclude_columns)

    def test_leaky_endpoint_split_folds(self):
        # Test the fold split function for a single DataFrame with multiple date columns
        results = leaky_folds_endpoint_split(self.df, self.num_folds, self.smiles_column, self.endpoint_date_columns, self.exclude_columns, self.chemprop, self.save_path)
        
        # Expect 2 tuples in the results since we have 2 folds
        self.assertEqual(len(results), self.num_folds)
        
        # Validate the sizes of each fold's train and test sets
        for fold, (train_df, test_df) in enumerate(results, start=1):
            # Check if the files were created
            self.assertTrue(os.path.exists(os.path.join(self.save_path, f'train_features_fold{fold}.csv')))
            self.assertTrue(os.path.exists(os.path.join(self.save_path, f'test_features_fold{fold}.csv')))
            self.assertTrue(os.path.exists(os.path.join(self.save_path, f'train_targets_fold{fold}.csv')))
            self.assertTrue(os.path.exists(os.path.join(self.save_path, f'test_targets_fold{fold}.csv')))
            
            # deleting the files after checking
            os.remove(os.path.join(self.save_path, f'train_features_fold{fold}.csv'))
            os.remove(os.path.join(self.save_path, f'test_features_fold{fold}.csv'))
            os.remove(os.path.join(self.save_path, f'train_targets_fold{fold}.csv'))
            os.remove(os.path.join(self.save_path, f'test_targets_fold{fold}.csv'))

            
if __name__ == '__main__':
    unittest.main()
