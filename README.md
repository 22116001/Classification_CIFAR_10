# Galaxy Classification

## Overview

This project aims to classify galaxy images into ten categories using Convolutional Neural Networks (CNNs). The dataset includes 17,736 images, each labeled into one of the following categories:

1. Disturbed Galaxies
2. Merging Galaxies
3. Round Smooth Galaxies
4. In-between Round Smooth Galaxies
5. Cigar Shaped Smooth Galaxies
6. Barred Spiral Galaxies
7. Unbarred Tight Spiral Galaxies
8. Unbarred Loose Spiral Galaxies
9. Edge-on Galaxies without Bulge
10. Edge-on Galaxies with Bulge

## Problem Statement

Classify galaxy images into the correct category with high accuracy using deep learning models.

## Dataset

- **Images**: 256x256x3
- **Labels**: 10 categories

## Requirements

- numpy
- pandas
- torch
- torchvision
- matplotlib
- scikit-image
- PIL (Pillow)

## Directory Structure

```
/content/drive/MyDrive/Galaxy prediction/
  ├── images.npy
  └── labels.npy
```

## Instructions

1. **Mount Google Drive**:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Load Data**:
    ```python
    images = np.load('/content/drive/MyDrive/Galaxy prediction/images.npy')
    labels = np.load('/content/drive/MyDrive/Galaxy prediction/labels.npy')
    ```

3. **Data Preprocessing**:
    - Normalize images.
    - Apply transformations.

4. **Create Dataset and DataLoader**:
    ```python
    train_dataset = CustomImageDataset(train_images, train_labels, transform=data_transforms['train'])
    valid_dataset = CustomImageDataset(valid_images, valid_labels, transform=data_transforms['validation'])
    ```

5. **Model Setup**:
    - Load and modify ResNet50 for 10 classes.
    - Define loss function and optimizer.

6. **Train Model**:
    ```python
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, device, num_epochs=15)
    ```

## Results

- Best validation accuracy: **86.44%**

## Notes

- Experiment with different models and hyperparameters.
- Implement advanced data augmentation techniques.
