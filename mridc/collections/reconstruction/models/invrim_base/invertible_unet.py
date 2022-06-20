import torch.nn as nn

from mridc.collections.reconstruction.models.invrim_base.invert_to_learn import InvertibleModule
from mridc.collections.reconstruction.models.invrim_base.invertible_layers import Housholder1x1, RevNetLayer


class InvertibleUnet(InvertibleModule):
    def __init__(self, n_channels, n_hidden, dilations, reversible_block=RevNetLayer, conv_nd=2, n_householder=3):
        """
        Initialize the network.

        Parameters
        ----------
        n_channels : list
            The number of channels of the input and output.
        n_hidden : list
            The number of hidden channels of the network.
        dilations : list
            The dilation of the network.
        reversible_block : class
            The class of the reversible block.
        conv_nd : int
            The number of dimensions of the convolution.
        n_householder : int
            The number of householder projections.
        """
        super(InvertibleUnet, self).__init__()
        self.in_ch = n_channels
        self.n_hidden = n_hidden
        self.dilations = dilations
        self.conv_nd = conv_nd
        self.n_householder = n_householder
        self.layers, self.embeddings = self.make_layers(reversible_block)

    def make_layers(self, reversible_block):
        """
        Make the layers of the network.

        Parameters
        ----------
        reversible_block : class
            The class of the reversible block.

        Returns
        -------
        layers : list
            The layers of the network.
        embeddings : list
            The embeddings of the network.
        """
        block_list = nn.ModuleList()
        embeddings_list = nn.ModuleList()
        for in_ch, n_hidden, dilation in zip(self.in_ch, self.n_hidden, self.dilations):
            layer = reversible_block(n_channels=in_ch, n_hidden=n_hidden, dilation=dilation, conv_nd=self.conv_nd)
            block_list.append(layer)
            embedding = Housholder1x1(self.in_ch[0], conv_nd=self.conv_nd, n_projections=self.n_householder)
            embeddings_list.append(embedding)
        return block_list, embeddings_list

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input to the network.
        """
        for layer, emb in zip(self.layers, self.embeddings):
            x = emb.forward(x)
            x = layer.forward(x)
            x = emb.reverse(x)
        return x

    def reverse(self, x):
        """
        Reverse pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input to the network.
        """
        for layer, emb in zip(reversed(self.layers), reversed(self.embeddings)):
            x = emb.forward(x)
            x = layer.reverse(x)
            x = emb.reverse(x)
        return x
