import unittest
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from ivers.stratify import stratify_endpoint_cv, stratify_split_and_cv

class TestStratifyFunctions(unittest.TestCase):

    @staticmethod
    def generate_sample_data(num_samples: int = 1000) -> pd.DataFrame:
        np.random.seed(42)
        data = {
            'SMILES': [f'CCO{str(i)}' for i in range(num_samples)],
            'endpoint1': np.random.randint(0, 3, size=num_samples),
            'endpoint2': np.random.randint(0, 3, size=num_samples),
            'feature1': np.random.rand(num_samples),
            'feature2': np.random.rand(num_samples),
            'label': np.random.choice(['A', 'B'], size=num_samples)
        }
        return pd.DataFrame(data)

    def setUp(self):
        self.df = self.generate_sample_data()
        self.order = ['endpoint1', 'endpoint2']
        self.smiles_column = 'SMILES'
        self.exclude_columns = ['feature1', 'feature2']
        self.aggregation_rules = {
            'endpoint1': 'mean',
            'endpoint2': 'mean',
            'feature1': 'mean',
            'feature2': 'mean'
        }
        self.test_size = 0.2
        self.n_splits = 5
        self.random_state = 42
        self.label_column = 'label'

    def test_stratify_endpoint_cv(self):
        splits = stratify_endpoint_cv(
            self.df, 
            self.order, 
            self.smiles_column, 
            self.exclude_columns, 
            self.aggregation_rules, 
            self.n_splits, 
            self.random_state, 
            self.label_column
        )

        self.assertEqual(len(splits), self.n_splits)
        for train_df, test_df, col_abbreviations in splits:
            self.assertGreater(len(train_df), 0)
            self.assertGreater(len(test_df), 0)
            self.assertIsInstance(col_abbreviations, dict)
            self.assertTrue(set(self.df.columns).issubset(set(train_df.columns)))
            self.assertTrue(set(self.df.columns).issubset(set(test_df.columns)))

    def test_stratify_split_and_cv(self):
        test_df, splits, col_abbreviations = stratify_split_and_cv(
            self.df, 
            self.order, 
            self.smiles_column, 
            self.exclude_columns, 
            self.aggregation_rules, 
            self.test_size, 
            self.n_splits, 
            self.random_state, 
            self.label_column
        )

        self.assertGreater(len(test_df), 0)
        self.assertEqual(len(splits), self.n_splits)
        self.assertIsInstance(col_abbreviations, dict)

        for train_df, val_df in splits:
            self.assertGreater(len(train_df), 0)
            self.assertGreater(len(val_df), 0)
            self.assertTrue(set(self.df.columns).issubset(set(train_df.columns)))
            self.assertTrue(set(self.df.columns).issubset(set(val_df.columns)))

if __name__ == '__main__':
    unittest.main()
