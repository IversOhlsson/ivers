import unittest
import pandas as pd
import os
import shutil
import tempfile
from ivers.temporal import leaky_endpoint_split, leaky_folds_endpoint_split

def create_dataframe():
    """Generate a sample dataframe for testing."""
    data = {
        'smiles': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'],
        'date_1': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', 
                   '2020-06-01', '2020-07-01', '2020-08-01', '2020-01-01', '2020-01-01'],
        'date_2': ['2020-01-15', '2020-02-15', '2020-03-15', '2020-04-15', '2020-05-15', 
                   '2020-06-15', '2020-07-15', '2020-08-15', '2020-09-15', '2020-09-15'],
        'date_3': ['2022-01-15', '2022-02-15', '2022-03-15', '2022-04-15', '2022-05-15', 
                   '2022-06-15', '2022-07-15', '2022-08-15', '2022-09-15', '2022-09-15'],
        'value_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, None],
        'value_2': [10, None, 30, None, 50, None, 70, 80, 90, None],
        'value_3': [1001, None, None, None, None, None, None, None, None, 1000],
        'feature_1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 91],
        'feature_2': [100, 200, 300, 400, 500, 600, 700, 800, 900, 901]
    }
    return pd.DataFrame(data)

class TestLeakyFoldsEndpointSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize once for all tests."""
        cls.df = create_dataframe()
        cls.num_folds = 3
        cls.smiles_column = 'smiles'
        cls.endpoint_date_columns = {'value_1': 'date_1', 'value_2': 'date_2', 'value_3' : 'date_3'}
        cls.feature_columns = ['feature_1']
        cls.chemprop = True
        cls.save_path = tempfile.mkdtemp()  # Use a temporary directory for files

    @classmethod
    def tearDownClass(cls):
        """Cleanup after all tests."""
        shutil.rmtree(cls.save_path)
        print(f"Deleted directory and all files in: {cls.save_path}")

    def test_correct_number_of_folds_generated(self):
        """Ensure the correct number of folds are generated."""
        result = leaky_folds_endpoint_split(self.df, self.num_folds, self.smiles_column, 
                                            self.endpoint_date_columns, 
                                            self.chemprop, self.save_path, self.feature_columns)
        self.assertEqual(len(result), self.num_folds, "Incorrect number of folds returned.")

    def test_c9_presence_in_all_folds(self):
        """Check that compound 'C9' is present in all folds correctly."""
        result = leaky_folds_endpoint_split(self.df, self.num_folds, self.smiles_column, 
                                            self.endpoint_date_columns,
                                            self.chemprop, self.save_path, self.feature_columns)
        found_in_train = found_in_test = False
        for train, test in result:
            if 'C9' in train[self.smiles_column].values:
                found_in_train = True
            if 'C9' in test[self.smiles_column].values:
                found_in_test = True

        self.assertTrue(found_in_train, "'C9' not found in any train set across all folds.")
        self.assertTrue(found_in_test, "'C9' not found in any test set across all folds.")
    
    def test_new_endpoint_data_always_in_test_set(self):
        """Ensure that all data from a newly added endpoint with limited entries are always in the test set."""
        # Generate the folds with the updated dataframe and endpoint
        result = leaky_folds_endpoint_split(self.df, self.num_folds, self.smiles_column,
                                            self.endpoint_date_columns,
                                            self.chemprop, self.save_path, self.feature_columns)

        # Check that all new data is in the test sets
        for fold_index, (train, test) in enumerate(result):
            
            # Check that 'value_3' is only in the test sets
            train_values = train['value_3'].dropna().values
            test_values = test['value_3'].dropna().values
            new_values = [1000, 1001]

            assert all(val in test_values for val in new_values), "New endpoint data not found in all test sets."
            assert not any(val in train_values for val in new_values), "New endpoint data incorrectly appears in train sets."



if __name__ == '__main__':
    unittest.main()
