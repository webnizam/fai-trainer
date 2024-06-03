import os
import shutil
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


def prepare_data(main_directory, image_size=(300, 300)):
    # Backup initial dataset
    backup_directory = main_directory + "_backup"
    if not os.path.exists(backup_directory):
        shutil.copytree(main_directory, backup_directory)
    print("Backup of the initial dataset created.")

    subdirectories = find_subdirectories(main_directory)
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
    print("Data preparation completed.")

    equalize_image_counts(main_directory)
    print("Image counts equalized across classes.")

    # Split the data into train and validation sets
    train_path = os.path.join(main_directory, "train")
    val_path = os.path.join(main_directory, "validation")
    split_dataset(main_directory, train_path, val_path, val_ratio=0.2)
    print("Data split into training and validation sets.")


if __name__ == "__main__":
    main_directory = "./data_directory"
    prepare_data(main_directory)
