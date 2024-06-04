FAI-Trainer
-----------

FAI-Trainer is a Python package designed to streamline the process of data preparation and model training for image classification tasks using PyTorch and torchvision. The package includes functionality for converting image formats, resizing images, removing duplicates, splitting datasets into training and validation sets, and training a ResNet50 model.

**Authors:** Nizamuddin Mohamed (@webnizam) [GitHub: webnizam](https://github.com/webnizam) | Michael Stattelman (@mstatt) [GitHub: mstatt](https://github.com/mstatt)

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

### Data Preparation and Model Training

To run both data preparation and model training in sequence:

    fai-trainer --prepare-data --train --batch-size 32 --epochs 3 --image-size 96 96

### Model Testing

To test the model on a specific image:

    fai-trainer --test --image-path path/to/your/image.jpg --image-size 96 96

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

©️2024 Falcons.AI | Vition.AI