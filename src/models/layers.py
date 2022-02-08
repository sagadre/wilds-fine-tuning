import torch.nn as nn

class Feedforward(nn.Module):

    def __init__(self, layer_sizes):
        super(Feedforward, self).__init__()
        layers = []
        for i in range(1, len(layer_sizes)-1):
            layers.append(nn.Linear(
                layer_sizes[i-1], layer_sizes[i])),
            layers.append(nn.LayerNorm(layer_sizes[i])),
            layers.append(nn.ReLU())
        layers.append(nn.Linear(
            layer_sizes[-2], layer_sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)