"""
This module contains the helper classes for
the ProtCNN class
"""
import torch
import torch.nn.functional as F

class Lambda(torch.nn.Module):
    """
    The Lambda class creates a PyTorch module that applies a given function to its input
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class ResidualBlock(torch.nn.Module):
    """
    The residual block used by ProtCNN (https://www.biorxiv.org/content/10.1101/626507v3.full).

    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        dilation: Dilation rate of the first convolution
    """

    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()

        # Initialize the required layers
        # Skip connection
        self.skip = torch.nn.Sequential()
        # 1D Batch Normalization Layer
        self.bn1 = torch.nn.BatchNorm1d(in_channels)
        # 1D Convolution Layer
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, bias=False, dilation=dilation, padding=dilation)
        # 1D Batch Normalization Layer
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        # 1D Convolution Layer
        self.conv2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, bias=False, padding=1)

    def forward(self, x)->torch.Tensor:
        """
        Defines the forward pass of the neural network.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (batch_size, num_channels, height, width)

        Returns:
        - torch.Tensor : The output tensor after passing through several layers and functions.
        """
        # Execute the required layers and functions
        activation = F.relu(self.bn1(x))
        x1 = self.conv1(activation)
        x2 = self.conv2(F.relu(self.bn2(x1)))

        return x2 + self.skip(x)
