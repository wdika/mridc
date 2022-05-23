# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/distributed.py

import os

import torch

from mridc.utils import logging


def initialize_distributed(args, backend="nccl"):
    """
    Initialize distributed training.

    Parameters
    ----------
    args: The arguments object.
    backend: The backend to use.
        default: "nccl"

    Returns
    -------
    local_rank: The local rank of the process.
    rank: The rank of the process.
    world_size: The number of processes.
    """
    # Get local rank in case it is provided.
    local_rank = args.local_rank

    # Get rank and world size.
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    logging.info(
        f"Initializing torch.distributed with local_rank: {local_rank}, rank: {rank}, world_size: {world_size}"
    )

    # Set the device id.
    device = rank % torch.cuda.device_count()
    if local_rank is not None:
        device = local_rank
    torch.cuda.set_device(device)

    # Call the init process.
    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6000")
    init_method += f"{master_ip}:{master_port}"
    torch.distributed.init_process_group(backend=backend, world_size=world_size, rank=rank, init_method=init_method)
    return local_rank, rank, world_size
