import torch.nn as nn


def mlp(
    num_inputs,
    num_outputs,
    hidden_layers,
    activation=nn.ReLU,
    output_activation=nn.Identity,
):
    layers = []
    layer_sizes = [num_inputs] + hidden_layers + [num_outputs]
    for i in range(len(layer_sizes) - 1):
        act = activation if i < len(layer_sizes) - 2 else output_activation
        layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act()]
    return nn.Sequential(*layers)
