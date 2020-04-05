# torch imports
import torch.nn.functional as F
import torch.nn as nn


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.

    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    def __init__(
        self, input_features, hidden_dim, output_dim, momentum, dropout_rate, num_layers
    ):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        :param momentum:  the batch norm momentum
        :dropout_rate:  dropout rate to use in the dropout layers
        :num_layers:  number of hidden layers
        """
        super(BinaryClassifier, self).__init__()

        self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList()
        current_dim = input_features

        for i in range(1, num_layers + 1):
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dim, momentum=momentum))
            self.layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        
        # sigmoid layer
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """

        for layer in self.layers:
            x = layer(x)
            
        x = self.out_layer(x)
        out = self.sig(x)

        return out