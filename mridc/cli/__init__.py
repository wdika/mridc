# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import argparse


def main():
    """Run the CLI."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparser = parser.add_subparsers(help="MRIDC commands.")
    subparser.required = True
    subparser.dest = "subcommand"

    from mridc.app.launch import register_parser as register_app_subcommand
    from mridc.cli.launch import register_parser as register_launch_subcommand

    register_app_subcommand(subparser)
    register_launch_subcommand(subparser)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
