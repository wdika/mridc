# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/get_rank.py
from mridc.utils.env_var_parsing import get_envint


def is_global_rank_zero():
    """Helper function to determine if the current process is global_rank 0 (the main process)."""
    # Try to get the pytorch RANK env var RANK is set by torch.distributed.launch
    rank = get_envint("RANK", None)
    if rank is not None:
        return rank == 0

    # Try to get the SLURM global rank env var SLURM_PROCID is set by SLURM
    slurm_rank = get_envint("SLURM_PROCID", None)
    if slurm_rank is not None:
        return slurm_rank == 0

    # if neither pytorch and SLURM env vars are set check NODE_RANK/GROUP_RANK and LOCAL_RANK env vars assume
    # global_rank is zero if undefined
    node_rank = get_envint("NODE_RANK", get_envint("GROUP_RANK", 0))
    local_rank = get_envint("LOCAL_RANK", 0)
    return node_rank == 0 and local_rank == 0
