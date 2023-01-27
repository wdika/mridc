# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import os


def register_parser(parser: argparse._SubParsersAction):
    """Register parser for the launch command."""
    parser_launch = parser.add_parser(
        "app",
        help="Launch MRIDC app with graphical interface (on default browser).",
    )
    # optionally select gpu to use
    parser_launch.add_argument(
        "-gid",
        "--gpu-id",
        type=int,
        default=None,
        help="Select GPU to use (in case of multiple GPUs). If not specified, the first GPU is used.",
    )
    parser_launch.set_defaults(func=main)


def main(args):
    """Run the app as process."""
    if args.gpu_id is None:
        os.system("streamlit run mridc/app/run.py --server.fileWatcherType none")
    else:
        os.system(f"CUDA_VISIBLE_DEVICES={args.gpu_id} streamlit run mridc/app/run.py --server.fileWatcherType none")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers())
    args = parser.parse_args()
    main(args)
