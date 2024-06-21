import unittest
import pandas as pd
import sys
sys.path.append('C:\\Users\\philip.ivers.ohlsson.TSS\\Documents\\ivers')

from ivers.temporal import leaky_endpoint_split, leaky_folds_endpoint_split

class TestLeakyFoldsEndpointSplit(unittest.TestCase):
    def setUp(self):
        """Create mock data for testing."""
        self.num_records = 100
        self.data = {
            'SMILES': [f'compound_{i}' for i in range(self.num_records)],
            'Endpoint1_Date': pd.date_range(start='2021-01-01', periods=self.num_records, freq='M'),
            'Activity': [1 if i % 2 == 0 else 0 for i in range(self.num_records)]
        }
        self.df = pd.DataFrame(self.data)
        self.num_folds = 5
        self.smiles_column = 'SMILES'
        self.endpoint_date_columns = {'Activity': 'Endpoint1_Date'}
        self.exclude_columns = []
        self.chemprop = False
        self.save_path = './'

    def test_fold_sizes(self):
        """Test if the number of folds and the size of each fold are correct."""
        splits = leaky_folds_endpoint_split(
            self.df,
            self.num_folds,
            self.smiles_column,
            self.endpoint_date_columns,
            self.exclude_columns,
            self.chemprop,
            self.save_path
        )

        # Check if the correct number of folds are created
        self.assertEqual(len(splits), self.num_folds)

        # Check the size of each fold
        expected_test_sizes = [int(self.num_records * (1 - i / self.num_folds)) for i in range(1, self.num_folds + 1)]
        for i, (_, test_df) in enumerate(splits):
            self.assertEqual(test_df.shape[0], expected_test_sizes[i])

    def test_output_files(self):
        """Test if the files are correctly saved to the directory."""
        import os
        import glob

        # Execute function
        splits = leaky_folds_endpoint_split(
            self.df,
            self.num_folds,
            self.smiles_column,
            self.endpoint_date_columns,
            self.exclude_columns,
            self.chemprop,
            self.save_path
        )

        # Check for created files
        files = glob.glob(os.path.join(self.save_path, '*.csv'))
        self.assertTrue(len(files) > 0, "No files were saved")


if __name__ == '__main__':
    unittest.main()
