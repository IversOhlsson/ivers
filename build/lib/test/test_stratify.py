import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from ivers.stratify import stratify_endpoint

class TestManageDistribution(unittest.TestCase):

    def setUp(self):
        # Create a more complex DataFrame with NaN values and labels
        self.df = pd.DataFrame({
    "smiles": ["C1=CC=CC=C1", "O=C(O)C", "CNC", "CCO", "O=C=O", "C=C", "C#N", "CCC", "C1CC1", "C1CCC1"],
    "y_var1": [1.0, 2.0, pd.NA, 4.0, 5.0, pd.NA, 7.0, 8.0, 9.0, 10.0],
    "y_var2": [5.0, pd.NA, 3.0, pd.NA, 1.0, 2.0, 5.0, 4.0, 6.0, 11.0],
    "misc": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "label": ["Group1", "Group1", "Group2", "Group2", "Group1", "Group1", "Group1", "Group2", "Group2", "Group2"]  
})

        # Setup for manage_distribution
        self.order = ['y_var1', 'y_var2']
        exclude_columns = ['misc']
        aggregation_rules = {'y_var1': 'mean', 'y_var2': 'mean'}
        test_size = 0.4
        random_state = 42
        label_column = 'label'

        # Execute manage_distribution
        self.train_df, self.test_df, col_abbreviations = stratify_endpoint(
            self.df, self.order, "smiles", exclude_columns, aggregation_rules, test_size, random_state, label_column
        )

    def test_train_test_split_sizes(self):
        # Check the size of train and test splits
        self.assertEqual(len(self.train_df) + len(self.test_df), len(self.df))

    def test_distribution(self):
        # Generate combinations of endpoint outcomes and labels to check distribution
        train_counts = self.train_df.groupby(['StratifyKey', 'label']).size().unstack(fill_value=0)
        test_counts = self.test_df.groupby(['StratifyKey', 'label']).size().unstack(fill_value=0)

        # Normalize counts to compare proportions
        train_norm = train_counts.div(train_counts.sum(axis=1), axis=0)
        test_norm = test_counts.div(test_counts.sum(axis=1), axis=0)

        # Check if the stratify process is intact across combinations
        for label in train_norm.columns:
            for key in train_norm.index:
                if label in test_norm.columns and key in test_norm.index:
                    train_ratio = train_norm.at[key, label]
                    test_ratio = test_norm.at[key, label]
                    self.assertLess(abs(train_ratio - test_ratio), 0.2)  # Ensure similar distributions

    def test_data_integrity(self):
        # Validate data integrity
        combined_df = pd.concat([self.train_df, self.test_df]).drop_duplicates()
        self.assertEqual(len(combined_df), len(self.df))

    def test_small_class_handling(self):
        # Count the occurrences of 'ConcatSmall' in the training and testing dataframes
        train_small_class_count = (self.train_df['StratifyKey'] == 'ConcatSmall').sum()
        test_small_class_count = (self.test_df['StratifyKey'] == 'ConcatSmall').sum()

        self.assertGreater(train_small_class_count, 0, "No small classes were merged in the training set")
        self.assertGreater(test_small_class_count, 0, "No small classes were merged in the testing set")

if __name__ == '__main__':
    unittest.main()
