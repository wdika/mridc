from torch import nn

from mridc.collections.reconstruction.models.invrim_base.utils import determine_conv_class


class ResidualBlockPixelshuffle(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, kernel_size=3, dilation=1, conv_nd=2, use_glu=True):
        """
        Initialize the layer.

        Parameters
        ----------
        n_input : int
            Number of input channels.
        n_output : int
            Number of output channels.
        n_hidden : int
            Number of hidden channels.
        kernel_size : int
            Kernel size.
        dilation : int
            Dilation factor.
        conv_nd : int
            Number of dimensions of the convolution.
        use_glu : bool
            Whether to use GLU.
        """
        super(ResidualBlockPixelshuffle, self).__init__()
        self.n_output = n_output
        self.conv_nd = conv_nd
        self.use_glu = use_glu
        conv_layer = determine_conv_class(conv_nd, transposed=False)
        transposed_conv_layer = determine_conv_class(conv_nd, transposed=True)

        if use_glu:
            n_output = n_output * 2

        self.l1 = nn.utils.weight_norm(
            conv_layer(n_input, n_hidden, kernel_size=dilation, stride=dilation, padding=0, bias=True)
        )
        self.l2 = nn.utils.weight_norm(
            conv_layer(n_hidden, n_hidden, kernel_size=kernel_size, padding=kernel_size // 2, dilation=1, bias=True)
        )
        self.l3 = nn.utils.weight_norm(
            transposed_conv_layer(n_hidden, n_output, kernel_size=dilation, stride=dilation, padding=0, bias=False)
        )

    def forward(self, x):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            The input to the layer.
        """
        x_size = list(x.size())
        x_size[1] = self.n_output
        x = nn.functional.relu_(self.l1.to(x)(x))
        x = nn.functional.relu_(self.l2.to(x)(x))
        x = self.l3.to(x)(x, output_size=tuple(x_size))
        if self.use_glu:
            x = nn.functional.glu(x, 1)
        return x
