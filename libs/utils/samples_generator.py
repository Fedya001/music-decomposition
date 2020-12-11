from argparse import ArgumentParser
from pathlib import Path
import os
import typing as tp
import random

import wav_utils


def generate_sample(input_path: str, output_path: str, rate: int, duration: float) -> None:
    """
    Performs pipeline of wav parsing to retrieve spectrogram ampliture matrix
    Pipiline is slice -> split_channels -> resample -> wtos.
    """
    slices_num: int = wav_utils.slice_wav(input_path, output_path, duration)
    slice_index: int = random.choice(list(range(slices_num - 1)))
    for i in range(slices_num):
        if i != slice_index:
            os.remove(output_path[:-4] + str(i) + ".wav")
    os.replace(output_path[:-4] + str(slice_index) + ".wav", output_path)

    # TODO: Remove file names: left.wav, right.wav, tmp.npy,
    # cuz we don't have any gurantee, that we won't have collisons
    wav_utils.split_channels(output_path, "left.wav", "right.wav")
    wav_utils.resample_wav("left.wav", output_path, rate)
    os.remove("left.wav")
    os.remove("right.wav")

    wav_utils.convert_wav_to_spectrogram(output_path, output_path, "tmp.npy")
    os.remove("tmp.npy")


def _generate_samples(input_path: str, output_path: str, size: int, rate: int, duration: float, set_name: str) -> None:
    dsd_dir = "Dev" if set_name == "train" else "Test"
    bounds: tp.Tuple[int, int] = (51, 100) if set_name == "train" else (1, 50)

    mixtures_path = Path(input_path) / "Mixtures" / dsd_dir
    sources_path = Path(input_path) / "Sources" / dsd_dir

    target_path: Path = Path(output_path) / set_name
    target_path.mkdir(exist_ok=True)
    for i in range(size):
        track_index: int = random.randint(bounds[0], bounds[1])
        target_subdir = str(i).zfill(3)
        (target_path / target_subdir).mkdir(exist_ok=True)

        track_dir_name: str = ""
        for filename in os.listdir(mixtures_path):
            if filename.startswith(str(track_index).zfill(3)):
                track_dir_name = filename
                break

        generate_sample(str(mixtures_path / track_dir_name / "mixture.wav"),
                        str(target_path / target_subdir / "mixture.npy"), rate, duration)
        generate_sample(str(sources_path / track_dir_name / "bass.wav"),
                        str(target_path / target_subdir / "bass.npy"), rate, duration)
        generate_sample(str(sources_path / track_dir_name / "drums.wav"),
                        str(target_path / target_subdir / "drums.npy"), rate, duration)
        generate_sample(str(sources_path / track_dir_name / "other.wav"),
                        str(target_path / target_subdir / "other.npy"), rate, duration)
        generate_sample(str(sources_path / track_dir_name / "vocals.wav"),
                        str(target_path / target_subdir / "vocals.npy"), rate, duration)


def generate_train_and_test(input_path: str, output_path: str, train_size: int, test_size: int, rate: int,
                            duration: float) -> None:
    _generate_samples(input_path, output_path, train_size, rate, duration, "train")
    _generate_samples(input_path, output_path, test_size, rate, duration, "test")


def main() -> None:
    arg_parser = ArgumentParser(prog="Samples generator",
                                description="Easy generate train and test samples")

    arg_parser.add_argument("-i", "--input", required=True, help="Input dir with dsd100 root")
    arg_parser.add_argument("-o", "--output", required=True, help="Ouput dir with generated samples")
    arg_parser.add_argument("--train_size", default=8, type=int, help="Train size")
    arg_parser.add_argument("--test_size", default=2, type=int, help="Test size")
    arg_parser.add_argument("-r", "--rate", default=8192, type=int, help="Rate of generated samples")
    arg_parser.add_argument("-d", "--duration", default=10, type=float, help="Sample duration")
    args = arg_parser.parse_args()

    generate_train_and_test(args.input, args.output, args.train_size, args.test_size, args.rate, args.duration)


if __name__ == "__main__":
    main()
