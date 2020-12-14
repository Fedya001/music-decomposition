from argparse import ArgumentParser
import cmath

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import typing as tp
import scipy.io.wavfile as wav


def convert_wav_to_spectrogram(wav_path: str, module_path: str, phase_path: str) -> None:
    rate, amplitudes = wav.read(wav_path)
    assert len(amplitudes.shape) == 1, "only mono can be converted to a spectrogram"
    spectrogram = librosa.stft(amplitudes.astype("float64"))

    v_abs = np.vectorize(lambda x: abs(x))
    np.save(module_path, v_abs(spectrogram.data))

    v_phase = np.vectorize(lambda x: cmath.phase(x))
    np.save(phase_path, v_phase(spectrogram.data))


def convert_spectrogram_to_wav(module_path: str, phase_path: str, rate: int, wav_path: str) -> None:
    modules = np.load(module_path)
    phases = np.load(phase_path)
    v_make_complex = np.vectorize(lambda module, phase: module * (cmath.cos(phase) + 1j * cmath.sin(phase)))

    spectrogram = v_make_complex(modules, phases)
    amplitudes = librosa.istft(spectrogram)
    wav.write(wav_path, rate, amplitudes.astype("int16"))


def slice_wav(wav_path: str, output_prefix_path: str, duration: float) -> int:
    rate, amplitudes = wav.read(wav_path)
    slice_len = int(rate * duration)
    for slice_index, sample_index in enumerate(range(0, len(amplitudes), slice_len)):
        wav_slice = amplitudes[sample_index:sample_index + slice_len]
        wav.write(output_prefix_path[:-4] + str(slice_index) + ".wav", rate, wav_slice)
    return len(range(0, len(amplitudes), slice_len))


def resample_wav(wav_path: str, result_path: str, rate: int, mono: bool = False) -> None:
    amplitudes, rate = librosa.load(wav_path, sr=rate, mono=mono)
    wav.write(result_path, rate, amplitudes)


def split_channels(wav_path: str, left_channel_path: str, right_channel_path: str) -> None:
    rate, amplitudes = wav.read(wav_path)
    wav.write(left_channel_path, rate, amplitudes[:, 0])
    wav.write(right_channel_path, rate, amplitudes[:, 1])


def join_channels(left_channel_path: str, right_channel_path: str, result_path: str) -> None:
    rate_left, amplitudes_left = wav.read(left_channel_path)
    rate_right, amplitudes_right = wav.read(right_channel_path)
    assert rate_left == rate_right, "joining channels must have same rate"
    assert len(amplitudes_left.shape) == 1, "left join channel must be mono"
    assert len(amplitudes_right.shape) == 1, "right join channel must be mono"
    assert amplitudes_left.shape == amplitudes_right.shape, "left and right joining channels must have same shape"
    wav.write(result_path, rate_left, np.column_stack((amplitudes_left, amplitudes_right)))


def display_spectrogram(module_path: str, png_path: str, rate: int) -> None:
    plt.figure(figsize=(20, 10))
    modules = np.load(module_path)
    db_scaled_modules = librosa.amplitude_to_db(modules, ref=np.max)
    librosa.display.specshow(db_scaled_modules, sr=rate, x_axis="time", y_axis="hz")
    plt.plot(format="%+2.f dB")
    plt.savefig(png_path)


def cut_wav(wav_path: str, output_path: str, from_timepoint: float, to_timepoint: float) -> None:
    rate, amplitudes = wav.read(wav_path)
    from_index = int(rate * from_timepoint)
    to_index = int(rate * to_timepoint)
    wav.write(output_path, rate, amplitudes[from_index:to_index])


def join_wavs(wav_paths: tp.List[str], output_path: str) -> None:
    joined_amplitudes: tp.Optional[tp.Sequence[int]] = None
    joined_rate: tp.Optional[int] = None

    for wav_path in wav_paths:
        rate, amplitudes = wav.read(wav_path)
        if joined_rate is None:
            joined_rate = rate
        assert joined_rate == rate, "All joining track must have the same rate"

        if joined_amplitudes is None:
            joined_amplitudes = amplitudes
        else:
            joined_amplitudes = np.vstack((joined_amplitudes, amplitudes))
    wav.write(output_path, joined_rate, joined_amplitudes)


def main() -> None:
    arg_parser = ArgumentParser(prog="wav/spectro utility",
                                description="Convert wav to spectrogram and vice versa, slice & resample wavs,"
                                "split & join stereo channels")

    subparsers = arg_parser.add_subparsers(help="actions")
    wtos_mode = subparsers.add_parser("wtos", help="Covert wav to spectrogram")
    wtos_mode.add_argument("-i", "--input", required=True, help="Input wav path")
    wtos_mode.add_argument("-m", "--module", required=True, help="Output spectrogram module path")
    wtos_mode.add_argument("-p", "--phase", required=True, help="Output spectrogram phase path")
    wtos_mode.set_defaults(which="wtos")

    stow_mode = subparsers.add_parser("stow", help="Convert spectrogram to wav")
    stow_mode.add_argument("-m", "--module", required=True, help="Input spectrogram  module path")
    stow_mode.add_argument("-p", "--phase", required=True, help="Input spectrogram  phase path")
    stow_mode.add_argument("-r", "--rate", required=True, type=int, help="Rate of initial wav")
    stow_mode.add_argument("-o", "--output", required=True, help="Output wav path")
    stow_mode.set_defaults(which="stow")

    slice_mode = subparsers.add_parser("slice", help="Slice wav into pieces with fixed duration")
    slice_mode.add_argument("-i", "--input", required=True, help="Input wav path")
    slice_mode.add_argument("-o", "--output", required=True, help="Output pieces prefix path")
    slice_mode.add_argument("-d", "--duration", required=True, type=float,
                            help="Duration of each piece in seconds (except the last one)")
    slice_mode.set_defaults(which="slice")

    resample_mode = subparsers.add_parser("resample")
    resample_mode.add_argument("-i", "--input", required=True, help="Input wav path")
    resample_mode.add_argument("-r", "--rate", required=True, type=int, help="New rate")
    resample_mode.add_argument("-o", "--output", required=True, help="Output resampled wav path")
    resample_mode.add_argument("--mono", action="store_true", help="Output resampled wav path")
    resample_mode.set_defaults(which="resample")

    split_channels_mode = subparsers.add_parser("split_channels", help="Split stereo into left ans right monos")
    split_channels_mode.add_argument("-i", "--input", required=True, help="Input stereo wav path")
    split_channels_mode.add_argument("-l", "--left", required=True, help="Output left mono wav path")
    split_channels_mode.add_argument("-r", "--right", required=True, help="Output right mono wav path")
    split_channels_mode.set_defaults(which="split_channels")

    join_channels_mode = subparsers.add_parser("join_channels", help="Join mono wavs into one stereo")
    join_channels_mode.add_argument("-l", "--left", required=True, help="Input left mono wav path")
    join_channels_mode.add_argument("-r", "--right", required=True, help="Input right mono wav path")
    join_channels_mode.add_argument("-o", "--output", required=True, help="Output stereo wav path")
    join_channels_mode.set_defaults(which="join_channels")

    visualize_mod = subparsers.add_parser("visualize", help="Visualize serialized spectrogram")
    visualize_mod.add_argument("-m", "--module", required=True, help="Spectrogram module path")
    visualize_mod.add_argument("-p", "--png", required=True, help="Result png path")
    visualize_mod.add_argument("-r", "--rate", required=True, type=int, help="wav rate of this spectrogram")
    visualize_mod.set_defaults(which="visualize")

    cut_mod = subparsers.add_parser("cut", help="Cut track")
    cut_mod.add_argument("-i", "--input", required=True, help="Input wav to cut")
    cut_mod.add_argument("-o", "--output", required=True, help="Output cut wav")
    cut_mod.add_argument("--from_time", required=True, type=int, help="Start timepoint (sec)")
    cut_mod.add_argument("--to_time", required=True, type=int, help="End timepoint (sec)")
    cut_mod.set_defaults(which="cut")

    join_mod = subparsers.add_parser("join", help="Join wavs")
    join_mod.add_argument("-i", "--input", action="append", required=True, help="List of wavs to join")
    join_mod.add_argument("-o", "--output", required=True, help="Output joined wav")
    join_mod.set_defaults(which="join")

    args = arg_parser.parse_args()

    if args.which == "wtos":
        convert_wav_to_spectrogram(args.input, args.module, args.phase)
    elif args.which == "stow":
        convert_spectrogram_to_wav(args.module, args.phase, args.rate, args.output)
    elif args.which == "slice":
        slice_wav(args.input, args.output, args.duration)
    elif args.which == "resample":
        resample_wav(args.input, args.output, args.rate, args.mono)
    elif args.which == "split_channels":
        split_channels(args.input, args.left, args.right)
    elif args.which == "join_channels":
        join_channels(args.left, args.right, args.output)
    elif args.which == "visualize":
        display_spectrogram(args.module, args.png, args.rate)
    elif args.which == "cut":
        cut_wav(args.input, args.output, args.from_time, args.to_time)
    elif args.which == "join":
        join_wavs(args.input, args.output)
    else:
        raise ValueError("Unsupported action")


if __name__ == "__main__":
    main()
