import unittest
import torch
import torch.nn.functional as F

from models import Lambda, ResidualBlock
from models.protcnn import ProtCNN

class TestModels(unittest.TestCase):
    """
    A class of unit and integration tests to test the functionality of the functions related to modelling
    """
    def testLambda(self):
        """
        A unit test for the Lambda Module Class
        """
        input_tensor = torch.randn(3, 4)
        lambda_module = Lambda(lambda x: x * 2)
        output_tensor = lambda_module(input_tensor)

        self.assertTrue(torch.allclose(output_tensor, input_tensor * 2))
    
    def testRBForward(self):
        """
        A unit test for the Residual Block forward function
        """
        in_channels = 8
        out_channels = 8
        dilation = 2
        x = torch.randn(2, in_channels, 10)
        residual_block = ResidualBlock(in_channels, out_channels, dilation)

        # Check if the output has the expected shape
        output = residual_block(x)
        self.assertEqual(output.shape, torch.Size([2, out_channels, 10]))

        # Check if the output is a sum of the input and the result of the convolutional layers
        self.assertTrue(torch.allclose(output, x + residual_block.conv2(F.relu(residual_block.bn2(residual_block.conv1(F.relu(residual_block.bn1(x))))))))
    
    def testProteinCNNForward(self):
        """
        A unit test for the ProteinCNN forward function
        """
        num_classes = 4
        lr = 0.1
        momentum = 0.9
        weight_decay = 0.01
        milestones = [5,10]
        gamma = 0.1
        x = torch.randn(2, 22, 10)
        protcnn = ProtCNN(num_classes, lr, momentum, weight_decay, milestones, gamma)

        # Check if the output has the expected shape
        output = protcnn(x)
        self.assertEqual(output.shape, torch.Size([2, num_classes]))

        # Check if the output is the result of the forward pass through the defined model
        self.assertTrue(torch.allclose(output, protcnn.model(x.float())))

    def testResidualBlockLambdaProtCNN(self):
        """
        An integration test for the ResidualBlock, Lambda and ProteinCNN functions
        """
        # Define input data
        x = torch.randn(22, 128, 10)
        
        # Instantiate ResidualBlock, Lambda and ProtCNN
        residual_block = ResidualBlock(128, 128, dilation=2)
        lambda_layer = Lambda(lambda x: x.flatten(start_dim=1))
        protcnn = ProtCNN(num_classes=10, lr=0.001, momentum=0.9, weight_decay=0.0001, milestones=[10,20,30], gamma=0.1)

        # Pass input through ResidualBlock
        x = residual_block(x)
        
        # Pass output of ResidualBlock through Lambda layer
        x = lambda_layer(x)
        
        # Pass output of Lambda layer through ProtCNN
        output = protcnn(x)
        
        # Assert shape of output
        self.assertEqual(output.shape, (1, 10))

