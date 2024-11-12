from argparse import ArgumentParser, Namespace
from pathlib import Path

from miditok.data_augmentation import augment_dataset


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Data Augmentation")

    # dataset setting
    parser.add_argument(
        "--data_folder",
        type=str,
        default="Pop1K7/midi_analyzed",
        help="folder of dataset"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    augment_dataset(
        data_path=Path(args.data_folder),
        out_path=f"{args.data_folder}_aug",
        pitch_offsets=[-12, 12],
        velocity_offsets=[-4, 4],
        duration_offsets=[-0.5, 0.5],
    )
