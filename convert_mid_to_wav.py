import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import soundfile as sf
from midi2audio import FluidSynth


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Symbolic Music Generation")

    # dataset setting
    parser.add_argument(
        "--input_folder",
        type=str,
        default="results/11-10-10-21-08/task1",
        help="folder of dataset"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results/11-10-10-21-08/task1_wav",
        help="folder of dataset"
    )
    return parser.parse_args()


def midi_to_wav(midi_path, wav_path, soundfont):
    fs = FluidSynth(soundfont)
    fs.midi_to_audio(midi_path, wav_path)


if __name__ == '__main__':
    args = parse_arguments()
    os.makedirs(args.output_folder, exist_ok=True)

    soundfont_file = "Dore Mark's NY S&S Model B-v5.2.sf2"
    midi_list = list(Path(args.input_folder).glob("*.mid"))

    for midi_path in midi_list:
        wav_path = Path(args.output_folder, midi_path.with_suffix(".wav").name)
        midi_to_wav(midi_path, wav_path, soundfont_file)
