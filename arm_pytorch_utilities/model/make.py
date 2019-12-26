import torch
import torch.nn
from arm_pytorch_utilities.model.mdn import MixtureDensityNetwork


def make_fully_connected_layers(input_dim=7, output_dim=3, H_units=32, H_layers=3):
    layers = []
    for i in range(H_layers):
        in_dim = input_dim if i == 0 else H_units
        out_dim = output_dim if i == H_layers - 1 else H_units
        layers.append(torch.nn.Linear(in_dim, out_dim, bias=True))
        layers.append(torch.nn.LeakyReLU())
    return layers


def make_sequential_network(end_block=None, input_dim=7, output_dim=3, H_units=32, H_layers=3, fc_output_dim=None):
    fc_output_dim = fc_output_dim or output_dim
    layers = make_fully_connected_layers(input_dim=input_dim, output_dim=fc_output_dim, H_units=H_units,
                                         H_layers=H_layers)
    if end_block:
        layers.append(end_block)

    network = torch.nn.Sequential(*layers).double()
    return network


def make_deterministic_model(**kwargs):
    return make_sequential_network(end_block=None, **kwargs)


def make_mdn_model(output_dim=3, num_components=4, H_units=32, **kwargs):
    mdn_block = MixtureDensityNetwork(H_units, output_dim, num_components)
    return make_sequential_network(end_block=mdn_block, **kwargs)
