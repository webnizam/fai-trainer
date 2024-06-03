import argparse
from .data_preparation import prepare_data
from .model_training import train_model, test_model


def main():
    parser = argparse.ArgumentParser(description="Prepare data and train the model.")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the trained model")
    parser.add_argument("--image-path", type=str, help="Path to an image for testing")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training and testing"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs for training"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(224, 224),
        help="Image dimensions (height, width)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results and diagrams",
    )

    args = parser.parse_args()

    if args.prepare_data:
        prepare_data(
            args.dataset_dir,
            image_size=tuple(args.image_size),
            processed_directory="./processed_data",
        )

    if args.train:
        train_model(
            batch_size=args.batch_size,
            epochs=args.epochs,
            image_size=tuple(args.image_size),
            dataset_dir="./processed_data",
            results_dir=args.results_dir,
        )

    if args.test:
        test_model(
            image_path=args.image_path,
            image_size=tuple(args.image_size),
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
