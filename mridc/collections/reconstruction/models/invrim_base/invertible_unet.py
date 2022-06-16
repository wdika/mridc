import torch.nn as nn

from mridc.collections.reconstruction.models.invrim_base.invert_to_learn import InvertibleModule
from mridc.collections.reconstruction.models.invrim_base.invertible_layers import Housholder1x1, RevNetLayer


class InvertibleUnet(InvertibleModule):
    def __init__(self, n_channels, n_hidden, dilations, reversible_block=RevNetLayer, conv_nd=2, n_householder=3):
        super(InvertibleUnet, self).__init__()
        self.in_ch = n_channels
        self.n_hidden = n_hidden
        self.dilations = dilations
        self.conv_nd = conv_nd
        self.n_householder = n_householder
        self.layers, self.embeddings = self.make_layers(reversible_block)

    def make_layers(self, reversible_block):
        block_list = nn.ModuleList()
        embeddings_list = nn.ModuleList()
        for in_ch, n_hidden, dilation in zip(self.in_ch, self.n_hidden, self.dilations):
            layer = reversible_block(n_channels=in_ch, n_hidden=n_hidden, dilation=dilation, conv_nd=self.conv_nd)
            block_list.append(layer)
            embedding = Housholder1x1(self.in_ch[0], conv_nd=self.conv_nd, n_projections=self.n_householder)
            embeddings_list.append(embedding)
        return block_list, embeddings_list

    def forward(self, x):
        for layer, emb in zip(self.layers, self.embeddings):
            x = emb.forward(x)
            x = layer.forward(x)
            x = emb.reverse(x)
        return x

    def reverse(self, x):
        for layer, emb in zip(reversed(self.layers), reversed(self.embeddings)):
            x = emb.forward(x)
            x = layer.reverse(x)
            x = emb.reverse(x)
        return x
