# Ivers

This project provides a robust library for managing data splits in machine learning projects with a focus on maintaining balanced distributions across various conditions. It is specifically designed to interface seamlessly with the Chemprop library, enhancing its capabilities for chemical property prediction.

## Features

- **Temporal Splits**: Supports two types of temporal data splits:
  - **Leaky**: Allows for forward-leakage in your data to simulate real-world scenarios where future data might influence the model subtly.
  - **AllForFree**: Provides a stricter temporal separation, ensuring that the training data is entirely independent of the test set, suitable for rigorous testing of model predictions over time.

- **Num Fold Split**: Implements a novel approach to increasing the training set size successively across multiple folds. This feature allows for progressive training, where each subsequent fold includes the data from all previous folds, enhancing model robustness.

- **Stratified Endpoint Split**: 
  - **Main Feature**: Our library introduces a stratified endpoint split, crucial for maintaining a consistent distribution of data across different categories or endpoints in your datasets.
  - **Utility**: Especially useful in scenarios where endpoint distributions are critical, such as in cheminformatics and bioinformatics.
  - **Cross-Validation Support**: Integrates capabilities to ensure that each cross-validation split maintains endpoint distribution, ideal for developing models that are generalizable across varied data conditions.

## Integration with Chemprop

- **Seamless Compatibility**: Designed to be used in conjunction with the Chemprop library, this feature allows users to leverage our data splitting strategies directly within their Chemprop workflows.
- **Enhanced Data Handling**: By integrating our splitting mechanisms, users can ensure that their chemical property prediction models are trained on well-structured and strategically partitioned datasets.

## Getting Started

To get started with this library, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourrepository/projectname.git
cd projectname
pip install -r requirements.txt
```