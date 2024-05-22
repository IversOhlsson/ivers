# Ivers


This project offers tools for managing data splits, ensuring endpoint distributions are maintained, and presents two novel temporal split techniques: 'leaky' and 'all for free' splits. See the explanation below. 

**Note**: This library was used in this paper [PlaceHolder](https://github.com/IversOhlsson/ivers) to generate data splits for the Chemprop library.

## Features
  - **Temporal Leaky**: This approach offers a more relaxed temporal separation, permitting a slight overlap between the training and test sets for different endpoints, allowing the model to be trained and tested on the same compound but for different endpoints, with each endpoint receiving x% of data. This setup closely simulates real-world scenarios where a model might access one endpoint before another.
  - **Temporal AllForFree**:  Initially, we add x% of data for each endpoint. Subsequently, if any endpoint for a compound has been added, we include all remaining endpoints for that compound.
  - **Temporal Fold Split**: Available for both leaky and allforfree where we split the data into folds, ensuring that each fold maintains the selected temporal separation between the training and test sets. In this approach, we gradually increase the size of the training set by moving the pointer that separates the test and train data.
  - **Stratified Endpoint Split**: Our library introduces a stratified endpoint split, crucial for maintaining a consistent distribution of data across different categories or endpoints in your datasets. Especially useful in scenarios where endpoint distributions are critical or where dataset might be imbalanced such as in chemoinformatics.
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
install the package via pip:
```bash
pip install ivers
```
We welcome contributions! Feel free to open issues or pull requests on our GitHub repository.

