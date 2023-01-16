"""
This module contains the ProtCNN class
and associated methods. This class is used
to run the ProtCNN model as described in
https://www.biorxiv.org/content/10.1101/626507v3.full
"""
import torch
import lightning as pl
import torchmetrics

from models import ResidualBlock, Lambda

class ProtCNN(pl.LightningModule):
    '''
    A PyTorch Lightning LightningModule subclass which represents a 1D convolutional neural network
    for protein sequence classification
    '''
    def __init__(self, num_classes:int, lr:float, momentum:float, weight_decay:float,
                 milestones:list, gamma:float, num_aa=22):
        super().__init__()
        # Define architecture of the model
        self.model = torch.nn.Sequential(
            # 1D convolution layer with 22 input channels, 128 output channels,
            # kernel size of 1 and no padding
            torch.nn.Conv1d(num_aa, 128, kernel_size=1, padding=0, bias=False),
            # Residual block with 128 input and output channels, and a dilation of 2
            ResidualBlock(128, 128, dilation=2),
            # Residual block with 128 input and output channels, and a dilation of 3
            ResidualBlock(128, 128, dilation=3),
            # Maxpooling layer with kernel size of 3 and stride 2
            torch.nn.MaxPool1d(3, stride=2, padding=1),
            # Flatten the output of maxpool layer
            Lambda(lambda x: x.flatten(start_dim=1)),
            # Linear Layer with 7680 inputs and num_classes outputs
            torch.nn.Linear(7680, num_classes)
        )
        # Initialize accuracy metrics for training and validation
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

        # Define optimization parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma

    def forward(self, x:torch.Tensor):
        # Performs the forward pass through the model
        return self.model(x.float())

    def training_step(self, batch:dict)->torch.Tensor:
        """
        This function performs a single training step on a batch of data.

        Parameters:
            batch (Dict): A dictionary containing the input and target data for the model.

        Returns:
            torch.Tensor: The value of the loss function computed on the input data.
        """
        # Get input and target from the batch
        x, y = batch['sequence'], batch['target']
        # Perform the forward pass and compute the loss
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # Log the loss for training
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        # Compute the predictions and accuracy
        pred = torch.argmax(y_hat, dim=1)
        self.train_acc(pred, y)
        # Log the training accuracy
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch:dict, batch_idx:int, logging = True)->torch.Tensor:
        """
        Perform a forward pass on the validation input and compute the accuracy of the
        model's predictions.

        Parameters:
        - batch (dict): a dictionary containing the validation data and target
        - batch_idx (int): the index of the current batch in the validation data
        (not used by our function but required )

        Returns:
        - acc (torch.Tensor): the accuracy of the model's predictions on the current batch
        """
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)
        acc = self.valid_acc(pred, y)
        if logging:
            self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)

        return acc

    def configure_optimizers(self)->dict:
        """
        Configures the optimizer and learning rate scheduler for a PyTorch model.

        Parameters:
            lr (float, optional): The initial learning rate for the optimizer.
            momentum (float, optional): The momentum value to use for the optimizer.
            weight_decay (float, optional): weight decay for optimizer.
            milestones (List[int], optional): The milestones for learning rate decay.
            gamma (float, optional): The factor by which the learning rate will be reduced.

        Returns:
            Dict: A dictionary containing the optimizer and the learning rate scheduler.
        """
        # Define the optimizer and learning rate scheduler
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones,
                                                            gamma=self.gamma)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
