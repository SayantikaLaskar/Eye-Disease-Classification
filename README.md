# Eye Disease Classification Using CNN, VGG16, and ResNet50

## Overview

This repository contains a project focused on the classification of eye diseases using convolutional neural networks (CNNs). The models implemented include a custom CNN, VGG16, and ResNet50. These models are trained and evaluated on a dataset of eye images to diagnose various eye conditions.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [License](#license)

## Introduction

The goal of this project is to develop an automated system for diagnosing eye diseases from images using deep learning techniques. Accurate and timely diagnosis of eye conditions is crucial for effective treatment and prevention of vision loss. This project leverages the power of CNNs, specifically the VGG16 and ResNet50 architectures, to classify various eye diseases.

## Features

- Implementation of a custom CNN model
- Implementation of pre-trained VGG16 and ResNet50 models
- Data preprocessing and augmentation
- Training and validation routines
- Performance evaluation metrics
- Visualization of results

## Requirements

- Python 3.7 or higher
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- OpenCV

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/eye-disease-classification.git
   cd eye-disease-classification
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Dataset

The link to the dataset:
https://www.kaggle.com/datasets/kondwani/eye-disease-dataset

## Usage

1. **Data Preprocessing:**
   ```sh
   python preprocess_data.py
   ```

2. **Train Custom CNN Model:**
   ```sh
   python train_cnn.py
   ```

3. **Train VGG16 Model:**
   ```sh
   python train_vgg16.py
   ```

4. **Train ResNet50 Model:**
   ```sh
   python train_resnet50.py
   ```

## Training

Training scripts for each model are provided. Modify the hyperparameters and paths in the respective scripts if needed.

- **Custom CNN:**
  ```sh
  python train_cnn.py
  ```

- **VGG16:**
  ```sh
  python train_vgg16.py
  ```

- **ResNet50:**
  ```sh
  python train_resnet50.py
  ```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
