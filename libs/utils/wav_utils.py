from argparse import ArgumentParser


def main() -> None:
    arg_parser = ArgumentParser(prog="wav/spectro utility",
                                description="Convert wav to spectrogram and vice versa, slice & resample wavs,"
                                "split & join stereo channels")

    subparsers = arg_parser.add_subparsers(help="actions")
    wtos_mode = subparsers.add_parser("wtos", help="Covert wav to spectrogram")
    wtos_mode.add_argument("-i", "--input", required=True, help="Input wav path")
    wtos_mode.add_argument("-o", "--output", required=True, help="Output spectrogram path")
    wtos_mode.set_defaults(which="wtos")

    stow_mode = subparsers.add_parser("stow", help="Convert spectrogram to wav")
    stow_mode.add_argument("-i", "--input", required=True, help="Input spectrogram  path")
    stow_mode.add_argument("-o", "--output", required=True, help="Output wav path")
    stow_mode.set_defaults(which="stow")

    slice_mode = subparsers.add_parser("slice", help="Slice wav into pieces with fixed duration")
    slice_mode.add_argument("-i", "--input", required=True, help="Input wav path")
    slice_mode.add_argument("-o", "--output", required=True, help="Output pieces prefix path")
    slice_mode.add_argument("-d", "--duration", required=True, help="Duration of each piece (except the last one)")
    slice_mode.set_defaults(which="slice")

    resample_mode = subparsers.add_parser("resample")
    resample_mode.add_argument("-i", "--input", required=True, help="Input wav path")
    resample_mode.add_argument("-o", "--output", required=True, help="Output resampled wav path")
    resample_mode.add_argument("-r", "--rate", required=True, help="New rate")
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

    args = arg_parser.parse_args()

    if args.which == "wtos":
        raise NotImplementedError
    elif args.which == "stow":
        raise NotImplementedError
    elif args.which == "slice":
        raise NotImplementedError
    elif args.which == "resample":
        raise NotImplementedError
    elif args.which == "split_channels":
        raise NotImplementedError
    elif args.which == "join_channels":
        raise NotImplementedError
    else:
        raise ValueError("Unsupported action")


if __name__ == "__main__":
    main()
