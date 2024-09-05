# Food Classification using Convolutional Neural Networks (CNN)

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to classify images of food into 11 categories. The task involves building a CNN architecture, preprocessing the data, and experimenting with different network structures to achieve optimal performance on the **Food-11 dataset**.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Network Architecture](#network-architecture)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to build and train a **CNN** to classify images of food into 11 categories, including Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetables/Fruits. The network is trained on the **Food-11** dataset, and data augmentation techniques are used to improve the model's generalization ability.

## Dataset

The dataset used is the **Food-11 dataset**, which can be downloaded from the following sources:
- [EPFL Food Dataset](https://mmspg.epfl.ch/food-image-datasets)
- [Kaggle Food-11 Dataset](https://www.kaggle.com/vermaavi/food11/data)

The dataset consists of images categorized into the following 11 food classes:
1. Bread
2. Dairy product
3. Dessert
4. Egg
5. Fried food
6. Meat
7. Noodles/Pasta
8. Rice
9. Seafood
10. Soup
11. Vegetables/Fruits

### Preprocessing

1. Extract labels from image filenames using a custom `get_label` function.
2. Load and decode images into JPEG format, resize them to **150x150** pixels.
3. Normalize the pixel values to a range of **0-1**.
4. Apply data augmentation (random horizontal flip, random rotation by 10%) to introduce variability during training.
5. Split the data into **training** and **validation** sets.

## Methodology

### Network Architecture

The **CNN** architecture for this task is built using **TensorFlow** and **Keras** libraries. The network consists of the following layers:

1. **Input Layer**: Accepts images of size **150x150x3**.
2. **Convolutional Layers**:
   - Four convolutional layers with increasing filter sizes: 32, 64, 128, and 256 filters.
   - Each convolutional layer is followed by **Batch Normalization** and **Max Pooling** layers.
3. **Global Average Pooling Layer**: Reduces spatial dimensions before the dense layers.
4. **Dense Layer**: A fully connected layer with 512 units, using L2 regularization and a **dropout layer** for regularization.
5. **Output Layer**: A dense layer with **11 units** corresponding to the 11 food classes, using **Softmax activation** to predict class probabilities.

### Training

- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Loss Function**: Sparse categorical cross-entropy.
- **Metrics**: Accuracy.
- **Early Stopping**: Used with a patience of 100 epochs to avoid overfitting.
- The model was trained for up to 100 epochs.

## Results

The model achieved an accuracy of **72%** on the validation set. The following figures represent the model's performance:

- **Figure 1**: Accuracy Testing Plot.
- **Figure 2**: Confusion Matrix (Part 1).
- **Figure 3**: Confusion Matrix (Part 2).
- **Figure 4**: Accuracy and Loss graphs showing the model's training process.

## Conclusion

The CNN achieved an accuracy of **72%** on the **Food-11** dataset, showing reasonable classification performance. Further improvements can be made by experimenting with deeper architectures, additional data augmentation techniques, and fine-tuning hyperparameters. 

## Installation

1. Clone the repository:
   

2. Install the required dependencies:
    

3. Download the **Food-11 dataset** and place it in the `data/` directory:
    ```bash
    data/
        |-- food11/
    ```

## Usage

1. **Preprocess the dataset**:
    

2. **Train the CNN model**:
   

3. **Evaluate the model**:
    The training and validation accuracy, along with the confusion matrix, will be displayed.

