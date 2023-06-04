# FashionMNIST Classification Project

This repository contains code for an image classification project on the FashionMNIST dataset. The aim of this project is to classify grayscale images of different clothing items into ten distinct classes with the highest accuracy possible. As of now, the project has achieved a maximum accuracy of **95.05%** on the validation set, which gives **7th** place comparing to results at [FashionMNIST repository](https://github.com/zalandoresearch/fashion-mnist).

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

My best model achieves an accuracy of 95.05% on the validation set. Logs from this training can be found inside `/logs` directory.

## Data Augmentation Techniques

In this project, I used a combination of data augmentation techniques to increase the diversity of my training data and improve model generalization. These methods manipulate the training images in ways that change their appearance but not their labels, which effectively increases the size and variability of my training set.

The following augmentation techniques were employed:

### Random Transformations

I applied random transformations to my images, which includes flips, rotations, and affine transformations. 

- **Flips**: Each image was randomly flipped horizontally.
- **Rotations**: Each image was randomly rotated by an angle within a specified range.
- **Affine Transformations**: This includes mainly translation (shifting the position of the image).

### Normalization

Normalization was carried out on the images, which is a process that changes the range of pixel intensity values. Usually the image data is normalized to fall in the range [0,1] or [-1,1]. In this case, I used the mean and standard deviation of my dataset for the normalization process, which makes my model's training process more stable and faster.

### Random Erasing

Random erasing is another data augmentation technique that was employed in this project. This method randomly selects a rectangle region in an image and erases its pixels with random values. It has been shown to enhance the model's robustness to occlusion and improve its ability to focus on more discriminative regions.

Combining these data augmentation techniques helped the model achieve a higher accuracy on the FashionMNIST dataset.

## Contact

For any further questions, please feel free to reach out. 

## Acknowledgements

I want to express my gratitude to Zalando Research for providing this interesting dataset. If you find this project helpful, please give a star and share it!
