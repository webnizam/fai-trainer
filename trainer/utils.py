import os
import shutil
import random
import numpy as np
from PIL import Image, ImageChops
from tqdm import tqdm
import matplotlib.pyplot as plt


def convert_png_to_jpg(directory):
    """Convert PNG images to JPG format in the specified directory."""
    for file in os.listdir(directory):
        if file.endswith(".png"):
            png_path = os.path.join(directory, file)
            jpg_path = os.path.join(directory, file.replace(".png", ".jpg"))
            img = Image.open(png_path)
            img.convert("RGB").save(jpg_path, "JPEG")
            os.remove(png_path)


def convert_webp_to_jpg(directory):
    """Convert WEBP images to JPG format in the specified directory."""
    for file in os.listdir(directory):
        if file.endswith(".webp"):
            webp_path = os.path.join(directory, file)
            jpg_path = os.path.join(directory, file.replace(".webp", ".jpg"))
            img = Image.open(webp_path)
            img.convert("RGB").save(jpg_path, "JPEG")
            os.remove(webp_path)


def convert_jpeg_to_jpg(directory):
    """Convert JPEG images to JPG format in the specified directory."""
    for file in os.listdir(directory):
        if file.lower().endswith(".jpeg"):
            jpeg_path = os.path.join(directory, file)
            jpg_path = os.path.join(directory, file.replace(".jpeg", ".jpg"))
            img = Image.open(jpeg_path)
            img.convert("RGB").save(jpg_path, "JPEG")
            os.remove(jpeg_path)


def count_images_in_directory(directory, extensions=[".jpg"]):
    """Count the number of images in the given directory with specified extensions."""
    num_images = 0
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in extensions):
            num_images += 1
    return num_images


def find_subdirectories(directory):
    """Find all subdirectories within a given directory."""
    return [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]


def find_truncated_images(directory):
    """Find truncated images in the specified directory."""
    truncated_images = []
    for file in tqdm(os.listdir(directory)):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            try:
                Image.open(file_path).load()
            except Exception as e:
                truncated_images.append(file)
    return truncated_images


def resize_images_in_subdirectories(directory, size=(48, 48)):
    """Resize all images in all subdirectories of the given directory."""
    for subdir in os.listdir(directory):
        subdirectory_path = os.path.join(directory, subdir)
        if os.path.isdir(subdirectory_path):
            for filename in tqdm(os.listdir(subdirectory_path)):
                file_path = os.path.join(subdirectory_path, filename)
                if os.path.isfile(file_path):
                    try:
                        with Image.open(file_path) as img:
                            img_resized = img.resize(size, Image.LANCZOS)
                            img_resized.save(file_path)
                    except IOError:
                        print(f"Cannot open or process the file: {file_path}")


def verify_jpg_images(directory):
    """Verify the integrity of JPG images in the specified directory."""
    for file_name in os.listdir(directory):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(directory, file_name)
            try:
                image = Image.open(file_path)
                image.verify()
            except (IOError, SyntaxError) as e:
                print(f"Fixing broken or corrupt image: {file_path}")
                os.remove(file_path)


def suffle_image_names(directory):
    """Shuffle the names of image files in the specified directory by adding a random triple-digit number prefix."""

    def generate_random_number():
        return str(random.randint(100000, 200000))

    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg")):
            random_number = "00" + generate_random_number()
            file_extension = os.path.splitext(filename)[1]
            new_filename = random_number + file_extension
            while os.path.exists(os.path.join(directory, new_filename)):
                random_number = "00" + generate_random_number()
                new_filename = random_number + file_extension
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            os.rename(old_file_path, new_file_path)


def convert_images_to_rgb(directory):
    """Ensure all images are in RGB format."""
    for subdir, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file_path.lower().endswith(".jpg"):
                try:
                    with Image.open(file_path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                            img.save(file_path)
                except Exception as e:
                    print(f"Error converting image {file_path}: {e}")


def display_random_images_from_subdirectories(root_dir):
    """Display a random sample of images from subdirectories."""
    subdirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    all_images = []
    for subdir in subdirs:
        images = [
            os.path.join(subdir, f)
            for f in os.listdir(subdir)
            if os.path.isfile(os.path.join(subdir, f)) and f.lower().endswith(".jpg")
        ]
        all_images.extend([(subdir, img) for img in images])
    num_images = len(all_images)
    if num_images < 9:
        print("Not enough images to display.")
        return
    random_images = random.sample(all_images, 9)
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(9):
        subdir, img_path = random_images[i]
        img = Image.open(img_path)
        ax = axs[i // 3, i % 3]
        if img.mode in ["L", "P"]:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(os.path.basename(subdir))
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def split_dataset(base_dir, train_dir, val_dir, val_ratio=0.2):
    """
    Splits the dataset into training and validation sets and moves original data.

    Parameters:
    - base_dir: Directory containing the class subdirectories.
    - train_dir: Directory where the training set will be created.
    - val_dir: Directory where the validation set will be created.
    - val_ratio: Ratio of files to be used for validation.
    """
    # Ensure the output directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Create the original_data directory
    original_data_dir = os.path.join(base_dir, "prepped_data")
    os.makedirs(original_data_dir, exist_ok=True)

    # Get the list of class subdirectories, excluding 'train', 'validation', and 'prepped_data'
    classes = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and d not in ["train", "validation", "prepped_data"]
    ]

    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        train_class_dir = os.path.join(train_dir, cls)
        val_class_dir = os.path.join(val_dir, cls)

        # Ensure the class subdirectories exist in train and val directories
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Get the list of files in the class directory
        files = [
            f
            for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
        ]

        # Shuffle the files
        random.shuffle(files)

        # Split the files
        val_count = int(len(files) * val_ratio)
        val_files = files[:val_count]
        train_files = files[val_count:]

        # Move files to the train and val directories
        for f in train_files:
            shutil.move(os.path.join(class_dir, f), os.path.join(train_class_dir, f))

        for f in val_files:
            shutil.move(os.path.join(class_dir, f), os.path.join(val_class_dir, f))

        print(
            f"Class {cls}: {len(train_files)} training files, {len(val_files)} validation files"
        )

    # Move the original class directories to the original_data directory
    for cls in classes:
        original_class_dir = os.path.join(base_dir, cls)
        new_location = os.path.join(original_data_dir, cls)
        shutil.move(original_class_dir, new_location)

    print(f"Moved original class directories to {original_data_dir}")


def remove_duplicates(directory):
    """Remove duplicate images in the specified directory."""
    unique_images = {}
    for subdir in os.listdir(directory):
        subdirectory_path = os.path.join(directory, subdir)
        if os.path.isdir(subdirectory_path):
            for filename in os.listdir(subdirectory_path):
                file_path = os.path.join(subdirectory_path, filename)
                if os.path.isfile(file_path):
                    with Image.open(file_path) as img:
                        img_hash = hash(img.tobytes())
                        if img_hash in unique_images:
                            os.remove(file_path)
                        else:
                            unique_images[img_hash] = file_path


def equalize_image_counts(main_dir):
    """Equalize the number of images across all subdirectories in the specified directory."""
    counts = {
        subdir: len(
            [
                file
                for file in os.listdir(os.path.join(main_dir, subdir))
                if file.endswith(".jpg")
            ]
        )
        for subdir in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, subdir))
    }
    min_count = min(counts.values())
    for subdir, count in counts.items():
        if count > min_count:
            subdir_path = os.path.join(main_dir, subdir)
            files = [file for file in os.listdir(subdir_path) if file.endswith(".jpg")]
            files_to_remove = random.sample(files, count - min_count)
            for file in files_to_remove:
                os.remove(os.path.join(subdir_path, file))
