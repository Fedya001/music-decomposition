from argparse import ArgumentParser
import os
import typing as tp

from tensorflow import keras
import keras.backend as K
import numpy as np
import tensorflow as tf

from common import fit_matrix_size
import wav_utils


def sum_offset(y_true: tp.Any, y_pred: tp.Any) -> tp.Any:
    return K.sum(K.abs(y_pred - y_true))


def mean_offset(y_true: tp.Any, y_pred: tp.Any) -> tp.Any:
    return K.mean(K.abs(y_pred - y_true))


def max_offset(y_true: tp.Any, y_pred: tp.Any) -> tp.Any:
    return K.max(K.abs(y_pred - y_true))


def process_slice(path: str, model: tp.Any) -> None:
    RATE = 8192
    HEIGHT, WIDTH = 1024, 256
    # TODO: Rewrite cleaner without duplication.

    wav_utils.split_channels(path, "left.wav", "right.wav")

    wav_utils.resample_wav("left.wav", "left.wav", RATE)
    wav_utils.resample_wav("right.wav", "right.wav", RATE)

    wav_utils.convert_wav_to_spectrogram("left.wav", "left_module.npy", "left_phase.npy")
    wav_utils.convert_wav_to_spectrogram("right.wav", "right_module.npy", "right_phase.npy")

    left_module = np.load("left_module.npy")
    left_module_before_shape = left_module.shape

    tensor = tf.convert_to_tensor(fit_matrix_size(left_module, HEIGHT, WIDTH))
    tensor = tf.reshape(tensor, (-1, HEIGHT, WIDTH))
    left_module = model.predict(tensor, batch_size=1)[0][:left_module_before_shape[0], :left_module_before_shape[1]]
    np.save("left_module.npy", left_module)

    right_module = np.load("right_module.npy")
    right_module_before_shape = right_module.shape

    tensor = tf.convert_to_tensor(fit_matrix_size(right_module, HEIGHT, WIDTH))
    tensor = tf.reshape(tensor, (-1, HEIGHT, WIDTH))
    right_module = model.predict(tensor, batch_size=1)[0][:right_module_before_shape[0], :right_module_before_shape[1]]
    np.save("right_module.npy", right_module)

    modules, phases = np.load("left_module.npy"), np.load("left_phase.npy")
    modules, phases = fit_matrix_size(modules, HEIGHT, WIDTH), fit_matrix_size(phases, HEIGHT, WIDTH)
    np.save("left_module.npy", modules)
    np.save("left_phase.npy", phases)
    wav_utils.convert_spectrogram_to_wav("left_module.npy", "left_phase.npy", RATE, "left.wav")

    modules, phases = np.load("right_module.npy"), np.load("right_phase.npy")
    modules, phases = fit_matrix_size(modules, HEIGHT, WIDTH), fit_matrix_size(phases, HEIGHT, WIDTH)
    np.save("right_module.npy", modules)
    np.save("right_phase.npy", phases)
    wav_utils.convert_spectrogram_to_wav("right_module.npy", "right_phase.npy", RATE, "right.wav")

    os.remove("left_module.npy")
    os.remove("left_phase.npy")
    os.remove("right_module.npy")
    os.remove("right_phase.npy")

    wav_utils.join_channels("left.wav", "right.wav", path)
    os.remove("left.wav")
    os.remove("right.wav")


def separate(input_path: str, output_path: str, model_path: str) -> None:
    slices_num: int = wav_utils.slice_wav(input_path, output_path, 16)

    model = keras.models.load_model(
        model_path, {"sum_offset": sum_offset, "mean_offset": mean_offset, "max_offset": max_offset})
    for slice_index in range(slices_num):
        path = output_path[:-4] + str(slice_index) + ".wav"
        process_slice(path, model)

    slice_names = [output_path[:-4] + str(slice_index) + ".wav" for slice_index in range(slices_num)]
    wav_utils.join_wavs(slice_names, output_path)

    for name in slice_names:
        os.remove(name)


def main() -> None:
    arg_parser = ArgumentParser(prog="Source separation",
                                description="Utility for separating tracks using pretrained models")
    arg_parser.add_argument("-m", "--model", required=True, help="Model to apply")
    arg_parser.add_argument("-i", "--input", required=True, help="Input track")
    arg_parser.add_argument("-o", "--output", required=True, help="Generated track")

    args = arg_parser.parse_args()
    separate(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
