FAI-Trainer
===========

FAI-Trainer is a Python package designed to streamline the process of data preparation and model training for image classification tasks using PyTorch and torchvision. The package includes functionality for converting image formats, resizing images, removing duplicates, splitting datasets into training and validation sets, and training a ResNet50 model.

Features
--------

*   **Data Preparation**: Convert image formats, resize images, remove duplicates, and ensure no corrupted images.
*   **Dataset Splitting**: Automatically split datasets into training and validation sets with a specified ratio.
*   **Model Training**: Train a ResNet50 model on the prepared dataset with configurable batch size, number of epochs, and image dimensions.
*   **Progress Tracking**: Visual progress tracking for both training and validation phases.
*   **Model Testing**: Load a trained model to test on a specific image or the validation dataset, and save the results.

Installation
------------

To install the FAI-Trainer package, use pip:

    pip install fai-trainer

Usage
-----

### Data Preparation

To prepare the data with a specified image size (default dataset directory is `datasets`):

    fai-trainer --prepare-data --image-size 224 224

To specify a different dataset directory:

    fai-trainer --prepare-data --dataset-dir path/to/your/dataset --image-size 224 224

### Model Training

To train the model with a specified batch size and number of epochs:

    fai-trainer --train --batch-size 64 --epochs 20 --image-size 224 224

### Full Pipeline

To run both data preparation and model training in sequence:

    fai-trainer --prepare-data --train --batch-size 64 --epochs 20 --image-size 224 224

### Model Testing

To test the model on a specific image:

    fai-trainer --test --image-path path/to/your/image.jpg --image-size 224 224

To test the model on the validation dataset:

    fai-trainer --test --image-size 224 224

Directory Structure
-------------------

Ensure your dataset directory has the following structure:

    
    datasets/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── class3/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
        

Contributing
------------

Contributions are welcome! Please open an issue or submit a pull request on GitHub.