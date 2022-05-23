# coding=utf-8
__author__ = "Dimitrios Karkalousos"

import torch.nn as nn

__all__ = ["rnn_weights_init"]


def rnn_weights_init(module, std_init_range=0.02, xavier=True):
    """
    # TODO: check if this is the correct way to initialize RNN weights
    Initialize different weights in Transformer model.

    Parameters
    ----------
    module: torch.nn.Module to be initialized
    std_init_range: standard deviation of normal initializer
    xavier: if True, xavier initializer will be used in Linear layers as was proposed in AIAYN paper, otherwise normal
    initializer will be used (like in BERT paper)
    """
    if isinstance(module, nn.Linear):
        if xavier:
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std_init_range)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std_init_range)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)
