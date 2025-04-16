FAI-Trainer
-----------
FAI-Trainer, the ultimate one-liner for image classification tasks in PyTorch! This user-friendly Python package simplifies the entire data preparation and model training process, allowing you to focus on what matters most - developing accurate models.

With FAI-Trainer, you can effortlessly convert your images into a format that's perfect for machine learning, resize them to the ideal size, and even eliminate duplicate images from your dataset. And if you're worried about splitting your data into training and validation sets? Don't be! FAI-Trainer takes care of that too.

Also: with just a single line of code, you can train a powerful Vision Transformer (ViT-B/16) model using PyTorch and torchvision. No more tedious setup or manual configuration required. Just load your dataset, specify some basic parameters, and let FAI-Trainer do the rest.

Whether you're a seasoned AI expert needing a quick trained model turnaround or just starting out with deep learning, FAI-Trainer is designed to make image classification tasks easier than ever before.

**Authors:** Nizamuddin Mohamed (@webnizam) [GitHub: webnizam](https://github.com/webnizam) | Michael Stattelman (@mstatt) [GitHub: mstatt](https://github.com/mstatt)

Features
--------

*   **Multi-Model Support**: Choose from multiple architectures including:
    - Vision Transformer (ViT-B/16) (default)
    - ResNet50
    - Vision Transformer (ViT-L/16)
    - EfficientNetV2-S
    - ConvNeXt-Tiny
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

To run data preparation and training with the default ViT-B/16:

    fai-trainer --prepare-data --train --batch-size 16 --epochs 3 --image-size 224 224

To train with a different model architecture:

    fai-trainer --train --model-type resnet50 --batch-size 32 --epochs 5
    fai-trainer --train --model-type efficientnet_v2_s --batch-size 32 --epochs 3
    fai-trainer --train --model-type convnext_tiny --batch-size 32 --epochs 3

Available model types:
- vit_b_16 (Vision Transformer Base, default)
- resnet50 (ResNet50)
- vit_l_16 (Vision Transformer Large)
- efficientnet_v2_s (EfficientNetV2 Small)
- convnext_tiny (ConvNeXt Tiny)

To train without using pretrained weights:

    fai-trainer --train --model-type vit_b_16 --no-pretrained

### Model Testing

To test the model on a specific image:

    fai-trainer --test --image-path path/to/your/image.jpg --model-type vit_b_16

Note: When testing, make sure to use the same model type that was used for training.

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

©️2025 Falcons.AI | Vition.AI
