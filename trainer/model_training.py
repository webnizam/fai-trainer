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
from .confusion_matrix import get_predictions, plot_confusion_matrix


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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    batch_size=32,
    epochs=10,
    image_size=(224, 224),
    dataset_dir="datasets",
    results_dir="results",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Data transformations
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

    # Load datasets
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
    num_classes = len(class_names)  # Number of classes

    # Model
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(
        num_ftrs, num_classes
    )  # Adjust the final layer to match the number of classes
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
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

        if epoch == 0 or epoch == epochs - 1:
            save_model_diagrams(
                results_dir,
                epoch,
                model,
                val_loader=dataloaders["validation"],
                device=device,
            )

    torch.save(model.state_dict(), os.path.join(results_dir, "model.pth"))
    torch.save(model, os.path.join(results_dir, "model-full.pth"))

    # Plot and save loss and accuracy graphs
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over epochs")
    plt.savefig(os.path.join(results_dir, "loss_plot.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over epochs")
    plt.savefig(os.path.join(results_dir, "accuracy_plot.png"))
    plt.close()

    print("Training completed!")
    print(f"Final Training Loss: {train_loss:.4f} Accuracy: {train_acc:.4f}")
    print(f"Final Validation Loss: {val_loss:.4f} Accuracy: {val_acc:.4f}")


def save_model_diagrams(results_dir, epoch, model, val_loader, device):
    # Save confusion matrix and other diagrams if needed
    actual_labels, predicted_labels = get_predictions(model, val_loader, device)
    plot_confusion_matrix(actual_labels, predicted_labels, val_loader.dataset.classes)

    plt.savefig(os.path.join(results_dir, f"confusion_matrix_epoch_{epoch + 1}.png"))
    plt.close()


def test_model(image_path=None, image_size=(224, 224), results_dir="results"):
    if not os.path.exists(os.path.join(results_dir, "model.pth")):
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

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(os.path.join(results_dir, "model.pth")))
    model = model.to(device)
    model.eval()

    if image_path:
        image = Image.open(image_path).convert("RGB")  # Convert image to RGB
        image_tensor = data_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1).squeeze()
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
            print(f"Predicted class: {predicted_class}")
            print("Class probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"{class_names[i]}: {prob * 100:.2f}%")

            # Save the result image with prediction
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
        plot_confusion_matrix(actual_labels, predicted_labels, class_names)

        plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
        plt.close()

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
