
Here's the updated documentation encapsulated in a code block for clarity:

vbnet
Copy code
# Ivers

Ivers offers a suite of tools designed for managing data splits while maintaining endpoint distributions, and introduces two novel temporal split techniques: 'Leaky' and 'All for Free'. This library ensures that data splits are suitable for realistic scenarios and rigorous testing needs in various applications. It was utilized to generate data splits in the research outlined in the [linked paper](https://github.com/IversOhlsson/ivers).

## Features
  - **Temporal Leaky**: Simulates real-world scenarios by allowing forward-leakage in data, which might subtly influence future models.
  - **Temporal AllForFree**: Ensures strict temporal separation, keeping training data completely independent of the test set—ideal for accurate long-term model predictions.
  - **Temporal Fold Split**: Progressively increases the training set size across multiple folds, adhering to the temporal sequence, enhancing model robustness over time.
  - **Stratified Endpoint Split**: Introduces a stratified approach to splitting, crucial for consistent endpoint distribution across different categories in datasets—beneficial in fields like cheminformatics and bioinformatics.


## Code Functions
The library includes several functions tailored for different splitting strategies:

- `stratify_endpoint`, `stratify_split_and_cv`: These functions generate train/test and cross-validation splits that respect endpoint distribution.
- `leaky_endpoint_split`, `allforone_endpoint_split`: Used for generating a single train/test split with respective temporal dynamics.
- `allforone_folds_endpoint_split`, `leaky_folds_endpoint_split`: Enable multiple sectional splits, increasing training data size consistently.
- `balanced_scaffold_cv`: Supports balanced scaffold cross-validation, enhancing data representativeness in splits.


## Integration with Chemprop

- Activating the `chemprop` configuration allows the library to generate splits that are directly compatible with the Chemprop framework, facilitating seamless integration and usage.

## Getting Started or Contributing

To begin using Ivers, clone the repository and set up the necessary dependencies:

```bash
git clone https://github.com/IversOhlsson/ivers.git
cd ivers
pip install -r requirements.txt
```

## Installation via pip
You can also install the package via pip:
```bash
pip install ivers
```
We welcome contributions! Feel free to open issues or pull requests on our GitHub repository.

## Guide

## Reference
when using this library, please cite the following paper:
```
@article{Ivers_1,
  title={PlaceHolder},
  author={PlaceHolder},
  journal={PlaceHolder},
  volume={PlaceHolder},
  number={PlaceHolder},
  pages={PlaceHolder},
  year={PlaceHolder},
  publisher={PlaceHolder}
}
```
