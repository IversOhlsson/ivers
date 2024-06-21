import os
import sys
import unittest
import pandas as pd

# Append the directory containing your custom modules to the Python path
sys.path.append('C:\\Users\\philip.ivers.ohlsson.TSS\\Documents\\ivers')

from ivers.stratify import stratify_split_and_cv

class TestStratifySplitAndCV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Current working directory: ", os.getcwd())
        # Create a sample DataFrame, this is a common setup for all tests
        cls.df = pd.DataFrame({
            "smiles": ["C1=CC=CC=C1", "O=C(O)C", "CNC", "CCO", "O=C=O", "C=C", "C#N", "CCC", "C1CC1", "C1CCC1"],
            "y_var1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y_var2": [5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y_var3": [None, None, None, None, None, None, None, None, None, 6.0],
            "misc": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "label": ["Group1", "Group1", "Group2", "Group2", "Group1", "Group1", "Group1", "Group2", "Group2", "Group2"]
        })

    def test_insufficient_classes_for_splits(self):
        endpoints = ["y_var1", "y_var2", "y_var3"]  # Added y_var3 to endpoints
        n_splits = 3
        
        # Expecting ValueError because there are not enough classes for the splits
        with self.assertRaises(ValueError) as context:
            stratify_split_and_cv(self.df, endpoints, "smiles", [], {}, 0.2, n_splits)

        self.assertTrue(f"Classes ['A-B-C'] not enought data points, {2 * (n_splits+1)} needed for {n_splits} splits." in str(context.exception))

    def test_successful_stratify_split(self):
        endpoints = ["y_var1", "y_var2"]
        n_splits = 3
        
        try:
            test_df, splits, col_abbreviations = stratify_split_and_cv(self.df, endpoints, "smiles", [], {}, 0.2, n_splits)
        except ValueError as e:
            self.fail(f"Unexpected error occurred: {str(e)}")

        self.assertIsInstance(test_df, pd.DataFrame)
        self.assertEqual(len(splits), n_splits)
        self.assertIsInstance(col_abbreviations, dict)

if __name__ == '__main__':
    unittest.main()
