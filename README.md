

# Ivers
Ivers is a toolkit designed to help split your data into training and testing sets in a way that mimics real-world situations, making it incredibly useful for fields like cheminformatics and bioinformatics. It introduces unique methods for creating these splits, such as Leaky and All for Free, to ensure that models train on data that closely reflects actual scenarios they will encounter, improving their effectiveness.

## Features
### Temporal Leaky
Simulates real-world scenarios by permitting forward-leakage in data, subtly influencing how future models may learn from past data. Allows controlled forward leakage, meaning a compound may appear in both training ($T$) and test ($X$) sets if associated with different endpoints at distinct times. For compound $c$, endpoint $e$, and measurement date $D_{c,e}$, assignments to training or test sets are determined by a global threshold date $D_{\text{thresh}}$:

```math
S(c, e) = 
\begin{cases} 
T & \text{if } D_{c,e} < D_{\text{thresh}} \\[6pt]
X & \text{if } D_{c,e} \geq D_{\text{thresh}}
\end{cases}
```
### Temporal AllForFree
Enforces strict temporal independence, ideal for cases needing accurate long-term model projections without any future leakage. Ensures strict temporal independence by assigning each compound entirely to either training or test sets based on its earliest recorded endpoint date, ${\min D_c}$:

```math
S(c) = 
\begin{cases} 
T & \text{if } \min D_c < D_{\text{thresh}} \\[6pt]
X & \text{if } \min D_c \geq D_{\text{thresh}}
\end{cases}
```

### Temporal Fold Split
Progressively increases the training set size across multiple folds, strictly following the chronological order for more robust model evaluation over time.

### Stratified Endpoint Split
Ensures that splits are balanced according to endpoint distributions across categories (e.g., chemical scaffolds or bioactivity classes).

### Balanced Scaffold CV
Enhances representativeness in cross-validation splits, particularly useful in cheminformatics where scaffold distributions can vary significantly.

## Installation
### From Source:
```bash
git clone https://github.com/IversOhlsson/ivers.git
cd ivers
pip install -r requirements.txt
```
### From Pip
```bash
pip install ivers
```

## Usage Example
The library includes several functions tailored for different splitting strategies:

- `stratify_endpoint`, `stratify_split_and_cv`: These functions generate train/test and cross-validation splits that respect endpoint distribution.
- `leaky_endpoint_split`, `allforone_endpoint_split`: Used for generating a single train/test split with respective temporal dynamics.
- `allforone_folds_endpoint_split`, `leaky_folds_endpoint_split`: Enable multiple sectional splits, increasing training data size consistently.
- `balanced_scaffold_cv`: Supports balanced scaffold cross-validation, enhancing data representativeness in splits.
```bash
import pandas as pd
from ivers.temporal import (
    allforone_endpoint_split, 
    allforone_folds_endpoint_split, 
    leaky_endpoint_split, 
    leaky_folds_endpoint_split
)
from ivers.stratify import stratify_split_and_cv  
from ivers.scaffold import balanced_scaffold

# Sample DataFrame to demonstrate usage
# ----- Create a sample DataFrame -----
data = {
    'smiles':  [ 
        'C1CCCCC1', 'C1=CC=CC=C1', 'CCO', 'CCN',   'CCC', 
        'C1CCOC1', 'C1=CC=CN=C1', 'CCOC',  'CCNC',  'CCCC', 
        'C1CCNCC1', 'C1=CC=CC=N1', 'CCCO', 'CCCN',   'CCCCC', 
        'C1CCSCC1',  'C1=CC=COC1', 'CCCCO', 'CCCCN', 'CCCCCC', 
        'C1CCCN1', 'C1=CC=CC=C1O', 'CCOCC', 'CCNCC',  'CCCCCCC', 
        'C1CCCNC1', 'C1=CC=C(O)C=C1','CCCCCO', 'CCCCCN', 'CCCCCCCC' ],
    'value': [
        5.0, 7.2, 3.1, 4.8, 6.0,
        5.1, 7.3, 3.2, 4.9, 6.1,
        5.2, 7.4, 3.3, 5.0, 6.2,
        5.3, 7.5, 3.4, 5.1, 6.3,
        5.4, 7.6, 3.5, 5.2, 6.4,
        5.5, 7.7, 3.6, 5.3, 6.5
    ],
    'date_col1': ['2021-01-10', '2021-02-15', '2021-03-20', '2021-04-25', '2021-05-30'] * 6,
    'date_col2': ['2021-01-12', '2021-02-18', '2021-03-25', '2021-04-28', '2021-06-02'] * 6,
    'endpoint1': [100, 200, 150, 180, 170] * 6,
    'endpoint2': [110, 210, 160, 190, 180] * 6,
    'other_info': ['A', 'B', 'C', 'D', 'E'] * 6,
    'exclude_col': ['ignore'] * 30
}

df = pd.DataFrame(data)

# Mapping from endpoint names to their respective date columns
endpoint_date_columns = {
    'endpoint1': 'date_col1',
    'endpoint2': 'date_col2'
}

# Allforone: Simple train/test split
train_df, test_df = allforone_endpoint_split(
    df.copy(),
    split_size=0.5,
    smiles_column='smiles',
    endpoint_date_columns=endpoint_date_columns
)

# Leaky: Train/test split with data leakage
train_df_leaky, test_df_leaky = leaky_endpoint_split(
    df.copy(),
    split_size=0.5,
    smiles_column='smiles',
    endpoint_date_columns=endpoint_date_columns
)

# Allforone: Multiple folds with additional features for Chemprop
folds_splits = allforone_folds_endpoint_split(
    df.copy(),
    num_folds=2,
    smiles_column='smiles',
    endpoint_date_columns=endpoint_date_columns,
    chemprop=True,
    save_path='.',
    aggregation='first',
    feature_columns=feature_columns
)
# Leaky: Multiple folds with leaky data handling
folds_leaky = leaky_folds_endpoint_split(
    df.copy(),
    num_folds=2,
    smiles_column='smiles',
    endpoint_date_columns=endpoint_date_columns,
    chemprop=False,
    save_path='.',         # adjust this path as needed
    feature_columns=feature_columns
)

# ----- Stratified Split and CV -----
test_size = 0.4   
n_splits = 2      

aggregation_rules = {}

test_df_strat, cv_splits_strat, col_abbreviations = stratify_split_and_cv(
    df.copy(),
    endpoints=endpoints,
    smiles_column='smiles',
    exclude_columns=exclude_columns,
    aggregation_rules=aggregation_rules,
    test_size=test_size,
    n_splits=n_splits,
    random_state=1337,
    label_column=None,
    chemprop=False,
    save_path='.',
    feature_columns=feature_columns
)

# ----- Balanced Scaffold Cross-Validation -----
df_scaffold, scaffold_splits, scaffold_fold_counts = balanced_scaffold(
    df=df,
    endpoints=endpoints,
    smiles_column='smiles',
    n_splits=n_splits,
    random_state=1337,
    feature_columns=feature_columns,
    exclude_columns=exclude_columns,
    chemprop=False,
    save_path='.'
)
```

###  Integration with Chemprop
If you're using Chemprop for molecular property predictions, Ivers can directly output data splits that are compatible with Chemprop's training scripts. Just set chemprop=True in any of the splitting functions to enable this feature. This integration simplifies the process of preparing your dataset for model training, making it easier to get reliable results.



## Reference
