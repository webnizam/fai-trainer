FAI-Trainer
-----------
FAI-Trainer, the ultimate one-liner for image classification tasks in PyTorch! This user-friendly Python package simplifies the entire data preparation and model training process, allowing you to focus on what matters most - developing accurate models.

With FAI-Trainer, you can effortlessly convert your images into a format that's perfect for machine learning, resize them to the ideal size, and even eliminate duplicate images from your dataset. And if you're worried about splitting your data into training and validation sets? Don't be! FAI-Trainer takes care of that too.

Also: with just a a single line of code, you can train a powerful ResNet50 model using PyTorch and torchvision. No more tedious setup or manual configuration required. Just load your dataset, specify some basic parameters, and let FAI-Trainer do the rest.

Whether you're a seasoned AI expert needing a quick trained model turnaround or just starting out with deep learning, FAI-Trainer is designed to make image classification tasks easier than ever before.

**Authors:** Nizamuddin Mohamed (@webnizam) [GitHub: webnizam](https://github.com/webnizam) | Michael Stattelman (@mstatt) [GitHub: mstatt](https://github.com/mstatt)

Features
--------

*   **Multi-Device Support**: Automatic selection of the best available device (XPU, CUDA, MPS, or CPU) for training and inference
*   **Data Preparation**: Convert image formats, resize images, remove duplicates, and ensure no corrupted images.
*   **Dataset Splitting**: Automatically split datasets into training and validation sets with a specified ratio.
*   **Model Training**: Train a ResNet50 model on the prepared dataset with configurable batch size, number of epochs, and image dimensions.
*   **Progress Tracking**: Visual progress tracking for both training and validation phases.
*   **Model Testing**: Load a trained model to test on a specific image or the validation dataset, and save the results.

System Requirements
-----------------

*   Python 3.8 or higher
*   For GPU acceleration:
    - NVIDIA GPU with CUDA support
    - Intel GPU with XPU support (requires intel-extension-for-pytorch)
    - Apple Silicon with MPS support (requires macOS 12.3+)
*   4GB RAM minimum (8GB+ recommended)

Installation
------------

To install the basic FAI-Trainer package:

    pip install fai-trainer

For Intel GPU acceleration support:

    pip install fai-trainer[xpu]

If Intel XPU support fails to install or initialize, the package will automatically fall back to other available devices in this order:
1. NVIDIA CUDA GPU (if available)
2. Apple Silicon MPS (if available)
3. CPU (always available as fallback)

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

### Directory Clean up

To clean up any prior processing or training:

    fai-trainer --clean

Contributing
------------

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

©️2024 Falcons.AI | Vition.AI
