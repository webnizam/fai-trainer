import argparse
from .data_preparation import prepare_data
from .model_training import train_model


def main():
    parser = argparse.ArgumentParser(description="Prepare data and train the model.")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(48, 48),
        help="Image dimensions (height, width)",
    )

    args = parser.parse_args()

    if args.prepare_data:
        prepare_data("./data_directory", image_size=tuple(args.image_size))

    if args.train:
        train_model(
            batch_size=args.batch_size,
            epochs=args.epochs,
            image_size=tuple(args.image_size),
        )


if __name__ == "__main__":
    main()
