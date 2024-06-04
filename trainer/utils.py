import os
import shutil
import random
import numpy as np
from PIL import Image, ImageChops, UnidentifiedImageError
from tqdm import tqdm
import matplotlib.pyplot as plt


def convert_image_format(directory, from_ext, to_ext, convert_func):
    format_mapping = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
    target_format = format_mapping.get(to_ext.lower(), to_ext.upper().replace(".", ""))

    for file in os.listdir(directory):
        if file.endswith(from_ext):
            from_path = os.path.join(directory, file)
            to_path = os.path.join(directory, file.replace(from_ext, to_ext))
            try:
                img = Image.open(from_path)
                convert_func(img).save(to_path, target_format)
                os.remove(from_path)
            except (UnidentifiedImageError, IOError) as e:
                print(f"Error converting {from_path}: {e}")


def convert_png_to_jpg(directory):
    convert_image_format(directory, ".png", ".jpg", lambda img: img.convert("RGB"))


def convert_webp_to_jpg(directory):
    convert_image_format(directory, ".webp", ".jpg", lambda img: img.convert("RGB"))


def convert_jpeg_to_jpg(directory):
    convert_image_format(directory, ".jpeg", ".jpg", lambda img: img.convert("RGB"))


def count_images_in_directory(directory, extensions=[".jpg"]):
    return sum(
        1
        for file in os.listdir(directory)
        if any(file.lower().endswith(ext) for ext in extensions)
    )


def find_subdirectories(directory):
    return [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]


def find_truncated_images(directory):
    truncated_images = []
    for file in tqdm(os.listdir(directory)):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            try:
                Image.open(file_path).load()
            except Exception:
                truncated_images.append(file)
    return truncated_images


def resize_images_in_subdirectories(directory, size=(48, 48)):
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
    for file_name in os.listdir(directory):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(directory, file_name)
            try:
                image = Image.open(file_path)
                image.verify()
            except (IOError, SyntaxError):
                print(f"Fixing broken or corrupt image: {file_path}")
                os.remove(file_path)


def suffle_image_names(directory):
    def generate_random_number():
        return str(random.randint(100000, 200000))

    for filename in os.listdir(directory):
        if filename.lower().endswith(".jpg"):
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
    if len(all_images) < 9:
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
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    classes = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d not in ["train", "validation"]
    ]

    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        train_class_dir = os.path.join(train_dir, cls)
        val_class_dir = os.path.join(val_dir, cls)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        files = [
            f
            for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
        ]
        random.shuffle(files)

        val_count = int(len(files) * val_ratio)
        val_files = files[:val_count]
        train_files = files[val_count:]

        for f in train_files:
            shutil.move(os.path.join(class_dir, f), os.path.join(train_class_dir, f))

        for f in val_files:
            shutil.move(os.path.join(class_dir, f), os.path.join(val_class_dir, f))

        print(
            f"Class {cls}: {len(train_files)} training files, {len(val_files)} validation files"
        )

    # Remove empty class directories
    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        if not os.listdir(class_dir):
            os.rmdir(class_dir)

    for split_dir in [train_dir, val_dir]:
        for cls in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, cls)
            if os.path.isdir(class_dir) and not os.listdir(class_dir):
                os.rmdir(class_dir)


def remove_duplicates(directory):
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
