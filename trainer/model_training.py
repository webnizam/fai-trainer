import os
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
    epochs, train_losses, val_losses, train_accuracies, val_accuracies
):
    table = PrettyTable()
    table.field_names = ["Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc"]

    for i in range(epochs):
        table.add_row(
            [
                i + 1,
                f"{train_losses[i]:.4f}",
                f"{val_losses[i]::.4f}",
                f"{train_accuracies[i] * 100:.2f}%",
                f"{val_accuracies[i] * 100:.2f}%",
            ]
        )

    print(table)
    return table


def count_images_in_subdirs(main_dir):
    labels = []
    counts = []

    # Loop through each subdirectory in the main directory
    for subdir in os.listdir(main_dir):
        subdir_path = os.path.join(main_dir, subdir)
        if os.path.isdir(subdir_path):
            # Count the number of images in the subdirectory
            image_files = natsorted(glob.glob(f"{subdir_path}/*.jpg"))
            num_images = len(image_files)
            labels.append(subdir)
            counts.append(num_images)

    return labels, counts


def generate_pie_chart(labels, counts, results_dir, locf="Train"):
    print("Generating chart for:" + locf)
    myexplode = [0.1] * len(
        labels
    )  # Adjust this list if you want specific slices to be exploded

    # Combine labels and counts for legend
    legend_labels = [f"{label} ({count})" for label, count in zip(labels, counts)]

    fig, ax = plt.subplots(figsize=(10, 7.5))  # Adjust the figure size
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
        self.multi_cell(0, 10, body)
        self.ln()

    def add_image(self, image_path, title, x=None, y=None, w=0, h=0):
        self.add_page()
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "C")
        self.image(image_path, x, y, w, h)
        self.ln(10)

    def add_table(self, table, title):
        self.add_page()
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.set_font("Arial", "", 12)
        # Add the table rows to the PDF
        col_width = self.w / len(table.field_names)
        th = self.font_size
        for header in table.field_names:
            self.cell(col_width, th, str(header), border=1)
        self.ln(th)
        for row in table:
            for datum in row:
                self.cell(col_width, th, str(datum), border=1)
            self.ln(th)


def train_model(
    batch_size=32,
    epochs=10,
    image_size=(224, 224),
    dataset_dir="datasets",
    results_dir="results",
    learning_rate=0.001,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(results_dir, exist_ok=True)

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

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        train_loss, train_acc = train_one_epoch(
            dataloaders["train"], model, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(
            dataloaders["validation"], model, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        print()

    torch.save(model.state_dict(), os.path.join(results_dir, "model.pth"))
    torch.save(model, os.path.join(results_dir, "model-full.pth"))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over epochs")
    loss_plot_path = os.path.join(results_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()

    plt.figure(figsize=(10, 5))
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
        epochs, train_losses, val_losses, train_accuracies, val_accuracies
    )

    actual_labels, predicted_labels = get_predictions(
        model, dataloaders["validation"], device
    )
    plot_confusion_matrix(actual_labels, predicted_labels, class_names, results_dir)

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

    # Generate PDF report
    pdf = PDFReport()
    pdf.add_page()
    pdf.chapter_title("Training Summary")
    pdf.chapter_body(
        f"Epochs: {epochs}\nBatch Size: {batch_size}\nLearning Rate: {learning_rate}"
    )
    pdf.add_table(summary_table, "Training Summary Table")
    pdf.add_image(loss_plot_path, "Loss over Epochs", w=190)
    pdf.add_image(accuracy_plot_path, "Accuracy over Epochs", w=190)
    pdf.add_image(train_pie_chart_path, "Train Image Distribution", w=190)
    pdf.add_image(val_pie_chart_path, "Validation Image Distribution", w=190)
    pdf.output(os.path.join(results_dir, "training_report.pdf"))

    print("PDF report generated successfully!")


def test_model(
    image_path=None, image_size=(224, 224), results_dir="results", load_full_model=False
):
    model_file = "model-full.pth" if load_full_model else "model.pth"

    if not os.path.exists(os.path.join(results_dir, model_file)):
        print("Model not found. Train the model first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
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
        plot_confusion_matrix(actual_labels, predicted_labels, class_names, results_dir)

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
