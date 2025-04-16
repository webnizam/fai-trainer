import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from prettytable import PrettyTable
from .confusion_matrix import get_predictions, plot_confusion_matrix
import warnings
import glob
from natsort import natsorted
from fpdf import FPDF
import random

warnings.filterwarnings("ignore", category=UserWarning)


def train_one_epoch(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def validate_one_epoch(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def print_summary_table(
    epochs, train_losses, val_losses, train_accuracies, val_accuracies, epoch_times
):
    table = PrettyTable()
    table.field_names = [
        "Epoch",
        "Train Loss",
        "Val Loss",
        "Train Acc",
        "Val Acc",
        "Elapsed Time",
    ]

    for i in range(epochs):
        table.add_row(
            [
                i + 1,
                f"{train_losses[i]:.4f}",
                f"{val_losses[i]:.4f}",
                f"{train_accuracies[i] * 100:.2f}%",
                f"{val_accuracies[i] * 100:.2f}%",
                f"{epoch_times[i]:.2f} sec",
            ]
        )

    print(table)
    with open(os.path.join("results", "epoch_summary.txt"), "w") as f:
        f.write(str(table))
    return table


def count_images_in_subdirs(main_dir):
    labels = []
    counts = []

    for subdir in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir)
        if os.path.isdir(subdir_path):
            image_files = natsorted(glob.glob(f"{subdir_path}/*.jpg"))
            num_images = len(image_files)
            labels.append(subdir)
            counts.append(num_images)

    return labels, counts


def generate_pie_chart(labels, counts, results_dir, locf="Train"):
    print("Generating chart for:" + locf)
    myexplode = [0.1] * len(labels)

    legend_labels = [f"{label} ({count})" for label, count in zip(labels, counts)]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        colors=plt.cm.tab20.colors,
        explode=myexplode,
        shadow=True,
        startangle=90,
    )
    plt.legend(
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title=locf + " Image Count",
    )
    plt.title(f"{locf} - Image Distribution")
    plt.tight_layout()
    new_filepath = f"{results_dir}/torch_{locf}_dataset_pie.png"
    plt.savefig(new_filepath)
    plt.close()
    return new_filepath


class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Training Report", 0, 1, "C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(2)

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_image(self, image_path, title, x=None, y=None, w=0, h=0):
        self.add_page()
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "C")
        self.image(image_path, x, y, w, h)
        self.ln(5)

    def add_inference_results(self, results):
        self.add_page()
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Inference Results", 0, 1, "L")
        self.ln(2)

        for index, result in enumerate(results):
            if index > 0:
                self.add_page()

            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "Inference Image", 0, 1, "C")
            self.image(result["image_path"], h=50)
            self.ln(5)

            self.set_font("Arial", "", 12)
            self.cell(0, 10, f"Predicted Class: {result['predicted_class']}", 0, 1, "L")
            self.cell(0, 10, f"Confidence: {result['confidence']:.2f}%", 0, 1, "L")
            self.ln(2)

            self.set_font("Arial", "", 10)
            self.cell(0, 10, "Class Probabilities", 0, 1, "L")
            self.ln(1)

            col_width = self.w / 2.5
            self.set_font("Arial", "", 10)
            self.cell(col_width, 10, "Class", 1)
            self.cell(col_width, 10, "Probability", 1)
            self.ln()

            for class_name, prob in result["probabilities"]:
                self.cell(col_width, 10, class_name, 1)
                self.cell(col_width, 10, f"{prob:.2f}%", 1)
                self.ln()
            self.ln(5)


def get_available_device():
    # Try Intel XPU first
    try:
        import intel_extension_for_pytorch  # noqa: F401
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device("xpu")
    except ImportError:
        print("Intel Extension for PyTorch not found. XPU support disabled.")
    except Exception as e:
        print(f"Error initializing XPU support: {str(e)}. XPU support disabled.")
    
    # Try CUDA next
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Try MPS (Apple Silicon)
    try:
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception as e:
        print(f"Error initializing MPS support: {str(e)}. MPS support disabled.")
    
    # Fall back to CPU
    print("No GPU acceleration available. Using CPU.")
    return torch.device("cpu")

def get_model(model_type="vit_b_16", num_classes=2, pretrained=True):
    """Get a model with specified architecture
    Args:
        model_type: str, one of ["resnet50", "vit_b_16", "vit_l_16", "efficientnet_v2_s", "convnext_tiny"]
        num_classes: int, number of output classes
        pretrained: bool, whether to use pretrained weights
    """
    if model_type == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_type == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    
    elif model_type == "vit_l_16":
        model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    
    elif model_type == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    
    elif model_type == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types are: resnet50, vit_b_16, vit_l_16, efficientnet_v2_s, convnext_tiny")
    
    return model

def train_model(
    batch_size=32,
    epochs=10,
    image_size=(224, 224),
    dataset_dir="datasets",
    results_dir="results",
    learning_rate=0.001,
    model_type="vit_b_16",
    pretrained=True,
):
    device = get_available_device()
    print(f"Using device: {device}")

    os.makedirs(results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(results_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "validation": transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x])
        for x in ["train", "validation"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "validation"]
    }

    class_names = image_datasets["train"].classes
    num_classes = len(class_names)

    model = get_model(model_type=model_type, num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    # Get trainable parameters based on model type
    if model_type == "resnet50":
        trainable_params = model.fc.parameters()
    elif model_type.startswith("vit"):
        trainable_params = model.heads.head.parameters()
    elif model_type == "efficientnet_v2_s":
        trainable_params = model.classifier[-1].parameters()
    elif model_type == "convnext_tiny":
        trainable_params = model.classifier[-1].parameters()
    else:
        trainable_params = model.parameters()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    epoch_times = []

    total_start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            dataloaders["train"], model, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(
            dataloaders["validation"], model, criterion, device
        )

        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        print(f"Elapsed Time: {epoch_time:.2f} seconds")
        print()

        # Save the latest model in the results directory
        torch.save(model.state_dict(), os.path.join(results_dir, "model.pth"))
        torch.save(model, os.path.join(results_dir, "model-full.pth"))

        # Save the model state and full model for each epoch in the checkpoint directory
        epoch_model_state_path = os.path.join(
            checkpoint_dir, f"model_epoch_{epoch + 1}.pth"
        )
        epoch_model_full_path = os.path.join(
            checkpoint_dir, f"model-full_epoch_{epoch + 1}.pth"
        )

        torch.save(model.state_dict(), epoch_model_state_path)
        torch.save(model, epoch_model_full_path)

        # Keep only the last 3 checkpoints
        checkpoints = natsorted(
            glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pth"))
        )
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                os.remove(old_checkpoint)

        full_checkpoints = natsorted(
            glob.glob(os.path.join(checkpoint_dir, "model-full_epoch_*.pth"))
        )
        if len(full_checkpoints) > 3:
            for old_full_checkpoint in full_checkpoints[:-3]:
                os.remove(old_full_checkpoint)

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    # Define the size for both figures
    figsize = (10, 5)

    # Plot Loss
    plt.figure(figsize=figsize)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over epochs")
    loss_plot_path = os.path.join(results_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=figsize)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over epochs")
    accuracy_plot_path = os.path.join(results_dir, "accuracy_plot.png")
    plt.savefig(accuracy_plot_path)
    plt.close()

    print("Training completed!")
    summary_table = print_summary_table(
        epochs, train_losses, val_losses, train_accuracies, val_accuracies, epoch_times
    )

    actual_labels, predicted_labels = get_predictions(
        model, dataloaders["validation"], device
    )
    confusion_matrix_path = plot_confusion_matrix(
        actual_labels, predicted_labels, class_names, results_dir
    )

    # Generate pie charts for image distributions
    train_labels, train_counts = count_images_in_subdirs(
        os.path.join(dataset_dir, "train")
    )
    train_pie_chart_path = generate_pie_chart(
        train_labels, train_counts, results_dir, locf="Train"
    )

    val_labels, val_counts = count_images_in_subdirs(
        os.path.join(dataset_dir, "validation")
    )
    val_pie_chart_path = generate_pie_chart(
        val_labels, val_counts, results_dir, locf="Validation"
    )

    # Randomly select 3 images from validation set for inference
    val_image_paths = glob.glob(os.path.join(dataset_dir, "validation", "*/*.jpg"))
    random.shuffle(val_image_paths)
    selected_images = val_image_paths[:3]

    inference_results = []

    for image_path in selected_images:
        image = Image.open(image_path).convert("RGB")
        image_tensor = data_transforms["validation"](image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1).squeeze()
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
            confidence = probabilities[predicted.item()] * 100

            sorted_probs = sorted(
                zip(class_names, probabilities.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )

            inference_results.append(
                {
                    "image_path": image_path,
                    "predicted_class": predicted_class,
                    "confidence": confidence.item(),
                    "probabilities": [
                        (class_name, prob * 100) for class_name, prob in sorted_probs
                    ],
                }
            )

    # Generate PDF report
    pdf = PDFReport()
    pdf.add_page()
    pdf.chapter_title("Training Summary")
    pdf.chapter_body(
        f"Epochs: {epochs}\nBatch Size: {batch_size}\nLearning Rate: {learning_rate}\nTotal Training Time: {total_training_time:.2f} seconds"
    )
    pdf.add_image(loss_plot_path, "Loss over Epochs", w=190)
    pdf.add_image(accuracy_plot_path, "Accuracy over Epochs", w=190)
    pdf.add_image(train_pie_chart_path, "Train Image Distribution", w=190)
    pdf.add_image(val_pie_chart_path, "Validation Image Distribution", w=190)
    pdf.add_image(confusion_matrix_path, "Confusion Matrix", w=190)
    pdf.add_inference_results(inference_results)
    pdf.output(os.path.join(results_dir, "training_report.pdf"))

    print("PDF report generated successfully!")


def test_model(
    image_path=None,
    image_size=(224, 224),
    results_dir="results",
    load_full_model=False,
    model_type="vit_b_16",
    pretrained=True,
):
    model_file = "model-full.pth" if load_full_model else "model.pth"

    if not os.path.exists(os.path.join(results_dir, model_file)):
        print("Model not found. Train the model first.")
        return

    device = get_available_device()
    print(f"Using device: {device}")

    data_transforms = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class_names = datasets.ImageFolder("./processed_data/train").classes

    if load_full_model:
        model = torch.load(os.path.join(results_dir, model_file), map_location=device)
    else:
        model = get_model(model_type=model_type, num_classes=len(class_names), pretrained=pretrained)
        model.load_state_dict(
            torch.load(os.path.join(results_dir, model_file), map_location=device)
        )
    model = model.to(device)
    model.eval()

    if image_path:
        image = Image.open(image_path).convert("RGB")
        image_tensor = data_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1).squeeze()
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
            print(f"\nPredicted class: {predicted_class}\n")

            sorted_probs = sorted(
                zip(class_names, probabilities.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )

            table = PrettyTable()
            table.field_names = ["Class", "Probability"]
            for class_name, prob in sorted_probs:
                table.add_row([class_name, f"{prob * 100:.2f}%"])

            print("Class probabilities:")
            print(table)

            result_image_path = os.path.join(results_dir, "result_image.png")
            plt.imshow(image)
            plt.title(f"Predicted: {predicted_class}")
            plt.savefig(result_image_path)
            plt.close()
    else:
        data_dir = "./processed_data/validation"
        test_dataset = datasets.ImageFolder(data_dir, data_transforms)
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4
        )

        actual_labels, predicted_labels = get_predictions(model, test_loader, device)
        confusion_matrix_path = plot_confusion_matrix(
            actual_labels, predicted_labels, class_names, results_dir
        )

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        with open(os.path.join(results_dir, "test_results.txt"), "w") as f:
            f.write(f"Test Accuracy: {accuracy:.4f}\n")

        return accuracy
