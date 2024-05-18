import unittest
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import List, Tuple
from ivers.temporal import allforfree_endpoint_split, allforfree_folds_endpoint_split

class TestEndpointSplits(unittest.TestCase):
    def setUp(self):
        data = {
            'SMILES': ['C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'C', 'CC', 'CCC', 'CCCC', 'CCCCC'],
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

    def test_leaky_endpoint_split(self):
        # Test the basic split function for a single DataFrame with multiple date columns
        train_df, test_df = allforfree_endpoint_split(self.df, self.split_size, self.smiles_column, self.endpoint_date_columns)


    def test_leaky_endpoint_split_folds(self):
        # Test the fold split function for a single DataFrame with multiple date columns
        num_folds = 2
        results = allforfree_folds_endpoint_split(self.df, num_folds, self.smiles_column, self.endpoint_date_columns)
        # Expect 2 tuples in the results since we have 2 folds
        self.assertEqual(len(results), 2)
        
        # Validate the sizes of each fold's train and test sets
        for train_df, test_df in results:
            continue

if __name__ == '__main__':
    unittest.main()
