from src.third_party.ConvONets.config import get_model as ConvONets
from pdb import set_trace


def create_network(mode_opt):
    # eval(mode_opt.network_type): ConvONets中的get_model
    network = eval(mode_opt.network_type)(mode_opt)
    return network
