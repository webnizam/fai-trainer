import os
import shutil
from PIL import Image, ImageOps, ImageEnhance
from .utils import (
    convert_png_to_jpg,
    convert_webp_to_jpg,
    convert_jpeg_to_jpg,
    resize_images_in_subdirectories,
    verify_jpg_images,
    shuffle_image_names,
    convert_images_to_rgb,
    find_subdirectories,
    split_dataset,
    remove_duplicates,
    equalize_image_counts,
)
import warnings
import random
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


def rotate_image_fixed(image):
    angles = [90, 180, 360]
    return [image.rotate(angle, expand=True) for angle in angles]


def rotate_image_random(image):
    angle = random.randint(0, 360)
    return image.rotate(angle, expand=True)


def translate_image(image):
    max_trans = min(image.size) // 10  # Translate up to 10% of the image size
    trans_x = random.randint(-max_trans, max_trans)
    trans_y = random.randint(-max_trans, max_trans)
    return image.transform(image.size, Image.AFFINE, (1, 0, trans_x, 0, 1, trans_y))


def scale_image(image):
    scale_factor = random.uniform(0.5, 1.5)
    width, height = image.size
    new_width, new_height = int(width * scale_factor), int(height * scale_factor)
    return image.resize((new_width, new_height), Image.ANTIALIAS)


def adjust_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(0.5, 1.5)  # Brightness adjustment factor
    return enhancer.enhance(factor)


def add_noise(image):
    np_image = np.array(image)
    noise = np.random.randint(0, 50, (np_image.shape), dtype="uint8")
    np_image = np_image + noise
    np_image = np.clip(np_image, 0, 255)
    return Image.fromarray(np_image)


def augment_image(image_path):
    base_image = Image.open(image_path)
    augmented_images = [
        base_image,
        ImageOps.mirror(base_image),
        ImageOps.flip(base_image),
        *rotate_image_fixed(base_image),
    ]
    return augmented_images


def augment_image_oversample(image_path):
    base_image = Image.open(image_path)
    augmented_images = [
        base_image,
        rotate_image_random(base_image),
        translate_image(base_image),
        scale_image(base_image),
        adjust_brightness(base_image),
        add_noise(base_image),
    ]
    return augmented_images


def convert_to_grayscale(images):
    grayscale_images = [ImageOps.grayscale(img) for img in images]
    return grayscale_images


def save_augmented_images(images, base_path, base_name):
    for i, img in enumerate(images):
        img_path = os.path.join(base_path, f"{base_name}_aug_{i}.jpg")
        img.save(img_path)


def augment_images_in_directory(directory):
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".jpg"):
                file_path = os.path.join(subdir, file)
                base_name = os.path.splitext(file)[0]
                augmented_images = augment_image(file_path)
                grayscale_images = convert_to_grayscale(augmented_images)
                save_augmented_images(
                    augmented_images + grayscale_images, subdir, base_name
                )
                os.remove(file_path)


def undersample_directory(directory):
    for subdir, _, files in os.walk(directory):
        jpg_files = [file for file in files if file.lower().endswith(".jpg")]
        if len(jpg_files) > 0:
            sampled_files = random.sample(jpg_files, len(jpg_files) // 2)
            for file in jpg_files:
                if file not in sampled_files:
                    os.remove(os.path.join(subdir, file))


def oversample_directory(directory, max_images):
    for subdir, _, files in os.walk(directory):
        jpg_files = [file for file in files if file.lower().endswith(".jpg")]
        if len(jpg_files) > 0:
            num_existing_images = len(jpg_files)
            sample_size = max_images - num_existing_images
            for i in range(sample_size):
                file = random.choice(jpg_files)
                img = Image.open(os.path.join(subdir, file))
                augmented_images = augment_image_oversample(os.path.join(subdir, file))
                grayscale_images = convert_to_grayscale(augmented_images)
                save_augmented_images(
                    augmented_images + grayscale_images, subdir, f"oversampled_{i}"
                )


def prepare_data(
    main_directory,
    image_size=(224, 224),
    processed_directory="processed_data",
    sampling_method="undersample",
):
    if not os.path.exists(processed_directory):
        shutil.copytree(main_directory, processed_directory)
    print("Processing directory created.")

    subdirectories = find_subdirectories(processed_directory)
    max_images = 0

    # Determine the maximum number of images in any class
    for subdir in subdirectories:
        num_images = len(
            [file for file in os.listdir(subdir) if file.lower().endswith(".jpg")]
        )
        if num_images > max_images:
            max_images = num_images

    for subdir in subdirectories:
        print(f"Preparing data in {subdir}")
        convert_png_to_jpg(subdir)
        convert_webp_to_jpg(subdir)
        convert_jpeg_to_jpg(subdir)
        resize_images_in_subdirectories(subdir, size=image_size)
        verify_jpg_images(subdir)
        shuffle_image_names(subdir)
        convert_images_to_rgb(subdir)
        remove_duplicates(subdir)
        augment_images_in_directory(subdir)

    if sampling_method == "undersample":
        for subdir in subdirectories:
            undersample_directory(subdir)
    elif sampling_method == "oversample":
        for subdir in subdirectories:
            oversample_directory(subdir, max_images)

    print("Data preparation completed.")

    equalize_image_counts(processed_directory)
    print("Image counts equalized across classes.")

    train_path = os.path.join(processed_directory, "train")
    val_path = os.path.join(processed_directory, "validation")
    split_dataset(processed_directory, train_path, val_path, val_ratio=0.2)
    print("Data split into training and validation sets.")


if __name__ == "__main__":
    main_directory = "./data_directory"
    prepare_data(main_directory, sampling_method="oversample")
