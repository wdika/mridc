# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/arguments.py

from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Union


def add_optimizer_args(
    parent_parser: ArgumentParser,
    optimizer: str = "adam",
    default_lr: float = None,
    default_opt_args: Optional[Union[Dict[str, Any], List[str]]] = None,
) -> ArgumentParser:
    """Extends existing argparse with support for optimizers.
    # Example of adding optimizer args to command line :
    python train_script.py ... --optimizer "novograd" --lr 0.01 \
        --opt_args betas=0.95,0.5 weight_decay=0.001
    Args:
        parent_parser (ArgumentParser): Custom CLI parser that will be extended.
        optimizer (str): Default optimizer required.
        default_lr (float): Default learning rate that should be overridden during training.
        default_opt_args (list(str)): List of overriding arguments for the instantiated optimizer.
    Returns:
        ArgumentParser: Parser extended by Optimizers arguments.
    """
    if default_opt_args is None:
        default_opt_args = []

    parser = ArgumentParser(parents=[parent_parser], add_help=True, conflict_handler="resolve")

    parser.add_argument("--optimizer", type=str, default=optimizer, help="Name of the optimizer. Defaults to Adam.")
    parser.add_argument("--lr", type=float, default=default_lr, help="Learning rate of the optimizer.")
    parser.add_argument(
        "--opt_args",
        default=default_opt_args,
        nargs="+",
        type=str,
        help="Overriding arguments for the optimizer. \n Must follow the pattern : \n name=value separated by spaces."
        "Example: --opt_args weight_decay=0.001 eps=1e-8 betas=0.9,0.999",
    )

    return parser


def add_scheduler_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """Extends existing argparse with default LR scheduler args.
    Args:
        parent_parser (ArgumentParser): Custom CLI parser that will be extended.
    Returns:
        ArgumentParser: Parser extended by LR Scheduler arguments.
    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
    parser.add_argument("--warmup_steps", type=int, required=False, default=None, help="Number of warmup steps")
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        required=False,
        default=None,
        help="Number of warmup steps as a percentage of total training steps",
    )
    parser.add_argument("--hold_steps", type=int, required=False, default=None, help="Number of hold LR steps")
    parser.add_argument(
        "--hold_ratio",
        type=float,
        required=False,
        default=None,
        help="Number of hold LR steps as a percentage of total training steps",
    )
    parser.add_argument("--min_lr", type=float, required=False, default=0.0, help="Minimum learning rate")
    parser.add_argument(
        "--last_epoch", type=int, required=False, default=-1, help="Last epoch id. -1 indicates training from scratch"
    )
    return parser


def add_recon_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """Extends existing argparse with default recon collection args.
    Args:
        parent_parser (ArgumentParser): Custom CLI parser that will be extended.
    Returns:
        ArgumentParser: Parser extended by MRIDC recon Collection arguments.
    """
    parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
    parser.add_argument(
        "--data_dir", type=str, required=False, help="data directory to training or/and evaluation dataset"
    )
    parser.add_argument("--config_file", type=str, required=False, default=None, help="Recon model configuration file")
    parser.add_argument(
        "--pretrained_model_name", default="recon-base-uncased", type=str, required=False, help="pretrained model name"
    )
    parser.add_argument("--do_lower_case", action="store_true", required=False, help="lower case data")
    return parser
