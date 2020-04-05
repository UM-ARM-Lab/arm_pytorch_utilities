import torch
import torch.nn
from arm_pytorch_utilities.model.mdn import MixtureDensityNetwork
from arm_pytorch_utilities.load_data import DataConfig


def make_fully_connected_layers(input_dim=7, h_units=(32, 32, 32), bias=True, activation_factory=torch.nn.LeakyReLU):
    layers = []
    h_layers = len(h_units)
    for i in range(h_layers):
        in_dim = input_dim if i == 0 else h_units[i - 1]
        out_dim = h_units[i]
        layers.append(torch.nn.Linear(in_dim, out_dim, bias=bias))
        layers.append(activation_factory())
    return layers


def make_mdn_end_block(num_components=4):
    def make_block(fc_output_dim, output_dim):
        return MixtureDensityNetwork(fc_output_dim, output_dim, num_components)

    return make_block


def make_linear_end_block(bias=True, activation=None):
    def make_block(fc_output_dim, output_dim):
        layers = [torch.nn.Linear(fc_output_dim, output_dim, bias=bias)]
        if activation is not None:
            layers.append(activation)
        return torch.nn.Sequential(*layers).double()

    return make_block


def make_sequential_network(config: DataConfig, end_block_factory=make_linear_end_block(), h_units=(32, 32),
                            **kwargs):
    # must have end block (otherwise we would always leave with an activation)
    if end_block_factory is None:
        raise RuntimeError("Need an end block to network")
    # setup input and output sizes based on data
    input_dim = config.input_dim()
    output_dim = config.ny

    layers = make_fully_connected_layers(input_dim=input_dim, h_units=h_units, **kwargs)
    end_block_input = h_units[-1] if len(h_units) else input_dim
    layers.append(end_block_factory(end_block_input, output_dim))

    network = torch.nn.Sequential(*layers).double()
    return network
