import os
import shutil
from PIL import Image, ImageOps
from .utils import (
    convert_png_to_jpg,
    convert_webp_to_jpg,
    convert_jpeg_to_jpg,
    resize_images_in_subdirectories,
    verify_jpg_images,
    suffle_image_names,
    convert_images_to_rgb,
    find_subdirectories,
    split_dataset,
    remove_duplicates,
    equalize_image_counts,
)


def augment_image(image_path):
    """Create multiple versions of an image with various transformations."""
    base_image = Image.open(image_path)
    images = []

    # Original
    images.append(base_image)

    # Horizontal flip
    images.append(ImageOps.mirror(base_image))

    # Vertical flip
    images.append(ImageOps.flip(base_image))

    # Rotations
    images.append(base_image.rotate(90, expand=True))
    images.append(base_image.rotate(180, expand=True))
    images.append(base_image.rotate(270, expand=True))

    return images


def save_augmented_images(images, base_path, base_name):
    """Save augmented images with appropriate names."""
    for i, img in enumerate(images):
        img_path = os.path.join(base_path, f"{base_name}_aug_{i}.jpg")
        img.save(img_path)


def augment_images_in_directory(directory):
    """Augment all images in the specified directory."""
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".jpg"):
                file_path = os.path.join(subdir, file)
                base_name = os.path.splitext(file)[0]
                augmented_images = augment_image(file_path)
                save_augmented_images(augmented_images, subdir, base_name)
                os.remove(file_path)  # Remove the original file if not needed


def prepare_data(
    main_directory, image_size=(224, 224), processed_directory="processed_data"
):
    # Create a copy of the original dataset
    if not os.path.exists(processed_directory):
        shutil.copytree(main_directory, processed_directory)
    print("Processing directory created.")

    subdirectories = find_subdirectories(processed_directory)
    for subdir in subdirectories:
        print(f"Preparing data in {subdir}")
        convert_png_to_jpg(subdir)
        convert_webp_to_jpg(subdir)
        convert_jpeg_to_jpg(subdir)
        resize_images_in_subdirectories(subdir, size=image_size)
        verify_jpg_images(subdir)
        suffle_image_names(subdir)
        convert_images_to_rgb(subdir)
        remove_duplicates(subdir)
        augment_images_in_directory(subdir)  # Augment images in each subdirectory
    print("Data preparation completed.")

    equalize_image_counts(processed_directory)
    print("Image counts equalized across classes.")

    # Split the data into train and validation sets
    train_path = os.path.join(processed_directory, "train")
    val_path = os.path.join(processed_directory, "validation")
    split_dataset(processed_directory, train_path, val_path, val_ratio=0.2)
    print("Data split into training and validation sets.")


if __name__ == "__main__":
    main_directory = "./data_directory"
    prepare_data(main_directory)
