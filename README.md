# Cardiovascular Disease Detection using Deep Learning

This project aims to build a deep learning model to predict cardiovascular diseases based on a dataset containing various medical attributes like blood pressure, cholesterol levels, and more. Using a Deep Neural Network (DNN), the model achieves high accuracy in detecting the presence of cardiovascular disease from the given features.

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The project is focused on leveraging a deep learning approach to detect cardiovascular disease from medical data. It utilizes the following steps:
1. Preprocessing of the dataset.
2. Building a Deep Neural Network (DNN) model to predict the target class (cardiovascular disease).
3. Evaluating the model performance using various metrics such as accuracy, precision, recall, and F1-score.

## Tech Stack

- **Python**: Core programming language for the project.
- **TensorFlow**: Deep learning library used to build and train the DNN model.
- **Keras**: High-level API of TensorFlow used for model building.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: Used for preprocessing and evaluation metrics.
  
## Dataset

The dataset consists of 70,000 data points with the following attributes:
- `id`, `age`, `gender`, `height`, `weight`, `ap_hi` (systolic blood pressure), `ap_lo` (diastolic blood pressure), `cholesterol`, `gluc` (glucose level), `smoke`, `alco`, `active`, and `cardio` (target label: 1 for disease, 0 for no disease).

Source of the dataset: [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

## Model Architecture

The Deep Neural Network (DNN) consists of:
- Input layer: Accepts 11 input features.
- Hidden layers: Multiple fully connected layers with ReLU activation.
- Output layer: A single neuron with sigmoid activation to classify the presence or absence of cardiovascular disease.

The optimizer used is Adam, and binary cross-entropy is used as the loss function.

## Results

The model achieves the following performance metrics on the test set:
- **Accuracy**: 94.43%

## Installation

To run this project, ensure you have the following dependencies installed:

```bash
pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. Clone this repository:
    ```bash
    git clone https://github.com/soumik-saha/Cardiovascular_Disease_Detection.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Cardiovascular_Disease_Detection
    ```

3. Run the Jupyter notebook:
    ```bash
    jupyter notebook Cardiovascular_Disease_Detection_Deep_Learning.ipynb
    ```

4. Follow the steps in the notebook to train and evaluate the model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you would like to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

