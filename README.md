# FashionMNIST Classification Project

This repository contains code for an image classification project on the FashionMNIST dataset. The aim of this project is to classify grayscale images of different clothing items into ten distinct classes with the highest accuracy possible. As of now, the project has achieved a maximum accuracy of **95.05%** on the validation set, which gives **7th** place comparing to results at [FashionMNIST repository](https://github.com/zalandoresearch/fashion-mnist) 

## Project Details

FashionMNIST is a dataset that comprises 60,000 28x28 grayscale images of ten different types of clothing, created by Zalando Research. It is a great dataset for benchmarking machine learning algorithms, especially in image classification tasks.

Our project employs a variety of techniques including data augmentation, hyperparameter tuning, and ensemble methods to maximize the accuracy. It also showcases the best practices for training, validating, and testing machine learning models.

## Requirements

- Python 3.8+
- NumPy
- Pandas
- TensorFlow
- Keras
- Scikit-learn
- Matplotlib

Please install the necessary packages by running `pip install -r requirements.txt` in your local environment.

## Contents

This repository contains the following files:

- `main.py`: The primary script for model training and validation.
- `model.py`: Contains the definition of the neural network model used for the task.
- `dataset.py`: Contains the dataset class representing prepared dataset.
- `const.py`: Constant parameters for the project.
- `requirements.txt`: List of python dependencies.

## Usage

To use this project, follow these steps:

1. Clone this repository to your local machine.
2. Install the necessary Python packages.
3. Run `python main.py`.

The script will automatically download the FashionMNIST dataset and place it inside `/data`, preprocess it, train the model, and finally evaluate the model's performance on the validation set (or only validate the best model, that is included - depending on parameters inside `const.py` file)

## Results

Our best model achieves an accuracy of 95.05% on the validation set. Logs from this training can be found inside `/logs` directory.

## Contact

For any further questions, please feel free to reach out. 

## Acknowledgements

I want to express our gratitude to Zalando Research for providing this interesting dataset. If you find this project helpful, please give a star and share it!
