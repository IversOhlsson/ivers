

# Ivers
Ivers is a toolkit designed for creating realistic training and testing splits for machine learning models, particularly in cheminformatics and bioinformatics. It provides specialized methods, such as **Leaky** and **AllForOne**, to ensure models are trained on data that closely reflects real-world scenarios, improving their predictive performance.

## Features
### Temporal Leaky
This method allows controlled forward leakage in data, simulating how future models may learn from past observations. A compound can appear in both the training ($T$) and test ($X$) sets if its endpoints are measured at different times. Given a compound $c$, an endpoint $e$, and a measurement date $D_{c,e}$, the split is determined based on a global threshold date, $D_{\text{thresh}}$.


```math
S(c, e) = 
\begin{cases} 
T & \text{if } D_{c,e} < D_{\text{thresh}} \\[6pt]
X & \text{if } D_{c,e} \geq D_{\text{thresh}}
\end{cases}
```
### Temporal AllForFree
Ensures strict temporal independence, making it ideal for scenarios that require accurate long-term model projections without future data leakage. Each compound is assigned exclusively to either the training or test set based on its earliest recorded endpoint date, ${\min D_c}$.


```math
S(c) = 
\begin{cases} 
T & \text{if } \min D_c < D_{\text{thresh}} \\[6pt]
X & \text{if } \min D_c \geq D_{\text{thresh}}
\end{cases}
```

### Temporal Fold Split
Splits data chronologically, progressively increasing the training set size across multiple folds. This approach ensures a more robust evaluation of model performance over time.

### Stratified Endpoint Split
Balances training and test sets based on endpoint distributions, ensuring that endpoints remain well-represented.

### Balanced Scaffold CV
Enhances the representativeness of cross-validation splits by considering scaffold distributions, making it particularly useful for cheminformatics applications.


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

- `stratify_endpoint`, `stratify_split_and_cv`: Generate train/test and cross-validation splits while maintaining endpoint balance.
- `leaky_endpoint_split`, `allforone_endpoint_split`: Create single train/test splits while incorporating temporal dynamics.
- `allforone_folds_endpoint_split`, `leaky_folds_endpoint_split`: Generate progressive training splits that increase dataset size over time.
- `balanced_scaffold_cv`: Supports balanced scaffold-based cross-validation, improving data representativeness.
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
Ivers supports direct integration with **Chemprop**, allowing you to generate training and test splits that work seamlessly with its training scripts. Just set `chemprop=True` in any splitting function to output Chemprop-compatible files, making dataset preparation more straightforward.








## Reference
