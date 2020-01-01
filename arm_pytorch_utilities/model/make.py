import torch
import torch.nn
from arm_pytorch_utilities.model.mdn import MixtureDensityNetwork
from arm_pytorch_utilities.load_data import DataConfig


def make_fully_connected_layers(input_dim=7, output_dim=3, H_units=32, H_layers=3, bias=True):
    layers = []
    for i in range(H_layers):
        in_dim = input_dim if i == 0 else H_units
        out_dim = output_dim if i == H_layers - 1 else H_units
        layers.append(torch.nn.Linear(in_dim, out_dim, bias=bias))
        layers.append(torch.nn.LeakyReLU())
    return layers


def make_mdn_end_block(num_components=4):
    def make_block(fc_output_dim, output_dim):
        return MixtureDensityNetwork(fc_output_dim, output_dim, num_components)

    return make_block


def make_linear_end_block(bias=True, activation=None):
    def make_block(fc_output_dim, output_dim):
        layers = [torch.nn.Linear(fc_output_dim, output_dim, bias=bias)]
        if activation:
            layers.append(activation())
        return torch.nn.Sequential(*layers).double()

    return make_block


def make_sequential_network(config: DataConfig, end_block_factory=make_linear_end_block(), H_units=32, H_layers=3,
                            **kwargs):
    if config.nx is None:
        raise RuntimeError("Unsepcified input dimension in config; load data first")
    # must have end block (otherwise we would always leave with an activation)
    if end_block_factory is None:
        raise RuntimeError("Need an end block to network")
    # setup input and output sizes based on data
    input_dim = config.nx
    if config.nu:
        input_dim += config.nu
    output_dim = config.ny

    # fully connected output size depends on if there
    fc_output_dim = H_units

    layers = make_fully_connected_layers(input_dim=input_dim, output_dim=fc_output_dim, H_units=H_units,
                                         H_layers=H_layers, **kwargs)
    layers.append(end_block_factory(H_units, output_dim))

    network = torch.nn.Sequential(*layers).double()
    return network
