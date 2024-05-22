# Ivers


This project offers tools for managing data splits, ensuring endpoint distributions are maintained, and presents two novel temporal split techniques: 'leaky' and 'all for free' splits. See the explanation below. 

**Note**: This library was used in this paper [PlaceHolder](https://github.com/IversOhlsson/ivers) to generate data splits for the Chemprop library.

## Features
  - **Temporal Leaky**: Allows for forward-leakage in your data to simulate real-world scenarios where future data might influence the model subtly.
  - **Temporal AllForFree**: Provides a stricter temporal separation, ensuring that the training data is entirely independent of the test set, suitable for rigorous testing of model predictions over time.
  - **Temporal Fold Split**: Implements a novel approach to increasing the training set size successively across multiple folds based on the temporal time sequence
  - **Stratified Endpoint Split**: Our library introduces a stratified endpoint split, crucial for maintaining a consistent distribution of data across different categories or endpoints in your datasets. Especially useful in scenarios where endpoint distributions are critical, such as in cheminformatics and bioinformatics.
  - **Cross-Validation Support**: Integrates capabilities to ensure that each cross-validation split maintains endpoint distribution, ideal for developing models that are generalizable across varied data conditions.

## Integration with Chemprop

- By setting the `chemprop` variable to `true`, the library will generate splits compatible with the Chemprop library. This ensures that the features and train-test splits are generated in a way that can easily be used with Chemprop.

## Getting Started or Contributing

To get started with this library, clone the repository and install the required dependencies:

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

