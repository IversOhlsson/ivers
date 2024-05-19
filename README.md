# Ivers


This project offers tools for managing data splits, ensuring endpoint distributions are maintained, and presents two novel temporal split techniques: 'leaky' and 'all for free' splits. See the explanation below.

**Note**: This library was used in this paper [PlaceHolder](https://github.com/IversOhlsson/ivers) to generate data splits for the Chemprop library.


## Features

- **Temporal Splits**: Supports two types of temporal data splits:
  - **Leaky**: Allows the same input data to be used in both training and test sets, but not for the same endpoint.
  - **AllForFree**: Provides a stricter temporal separation, ensuring that the input data is entirely independent of the test set. The logic involves taking x% from all the endpoints and then adding all other endpoints for that input data that has already been added.
  
- **Stratified Endpoint Split**:
  - **Main Feature**: Introduces a stratified endpoint split, crucial for maintaining a consistent distribution of data across different categories and/or endpoints in your datasets.
  
- **General**: Includes support to generate splits for x number of cross-validation splits.

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

