import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_predictions(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds


def calculate_scores(cm):
    correct_scores = [cm[i, i] / np.sum(cm[i]) for i in range(len(cm))]
    return correct_scores


def plot_confusion_matrix(actual, preds, class_labels):
    # Generate confusion matrix
    cm = confusion_matrix(actual, preds)
    num_classes = len(class_labels)

    # Calculate scores for each label
    scores = calculate_scores(cm)

    # Create a custom figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Use a gradient color map for intensity
    cmap = plt.cm.RdYlGn

    # Plot the confusion matrix with intensity proportional to scores
    cax = ax.matshow(cm, cmap=cmap)

    # Set text color based on correct or incorrect predictions
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if i == j else "black"
            ax.text(j, i, str(cm[i, j]), va="center", ha="center", color=color)

    # Set labels for the axes
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Calculate accuracy
    def calculate_accuracy(cm):
        return np.trace(cm) / np.sum(cm)

    accuracy = calculate_accuracy(cm)
    str_title = f"Confusion Matrix\n{accuracy * 100:.2f}% accuracy."
    plt.title(str_title)

    # Create custom legend handles and labels with scores
    legend_handles = [
        plt.Line2D(
            [0], [0], color=cmap(score), lw=4, label=f"{class_labels[i]}: {score:.2f}"
        )
        for i, score in enumerate(scores)
    ]

    # Sort the legend handles based on scores
    legend_handles = sorted(
        legend_handles, key=lambda x: float(x.get_label().split(": ")[1]), reverse=True
    )

    # Add a legend
    legend = plt.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(1, 1), title="Legend"
    )
    plt.setp(legend.get_texts(), color="black")  # Set legend text color to black

    # Save CM without cropping the legend
    plt.savefig("results/torch_confusion_matrix.png", bbox_inches="tight")

    # Display the plot
    # plt.show()
