import omegaconf
from torch import nn

from src.third_party.ConvONets.conv_onet import generation_two_stage as generation
from src.third_party.ConvONets.conv_onet import models
from src.third_party.ConvONets.conv_onet.models import (
    OnlyPointNet_Implicit,
    OnlyPointNet_New,
    ConvolutionalOccupancyNetworkGeoArt,
    ConvolutionalOccupancyNetworkGeoMapping,
    ConvolutionalOccupancyNetworkGeoMapping_New,
    ConvolutionalOccupancyNetworkGeoMapping_3frames,
    ConvolutionalOccupancyNetworkGeoMapping_3frames_interpolation,
    ConvolutionalOccupancyNetworkGeoMapping_3frames_extrapolation,
    ConvolutionalOccupancyNetworkGeoMapping_3frames_grid_visualization,
    ConvolutionalOccupancyNetworkGeoMapping_New_Implicit,
)
from src.third_party.ConvONets.encoder import encoder_dict

from ipdb import set_trace

def get_model(cfg, dataset=None, **kwargs):
    """Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config
        dataset (dataset): dataset
    """
    convonet_type = cfg["convonet_type"]
    decoder = cfg["decoder"]
    encoder = cfg["encoder"]
    c_dim = cfg["c_dim"]
    decoder_kwargs = cfg["decoder_kwargs"]
    encoder_kwargs = cfg["encoder_kwargs"]
    padding = cfg["padding"]

    if padding is None:
        padding = 0.1

    # for pointcloud_crop
    try:
        encoder_kwargs["unit_size"] = cfg["data"]["unit_size"]
        decoder_kwargs["unit_size"] = cfg["data"]["unit_size"]
    except:
        pass
    # local positional encoding
    if "local_coord" in cfg.keys():
        encoder_kwargs["local_coord"] = cfg["local_coord"]
        decoder_kwargs["local_coord"] = cfg["local_coord"]
    if "pos_encoding" in cfg:
        encoder_kwargs["pos_encoding"] = cfg["pos_encoding"]
        decoder_kwargs["pos_encoding"] = cfg["pos_encoding"]

    decoders = []
    # 如果有多个decoders该怎么处理
    if isinstance(cfg["decoder"], list) or isinstance(
        cfg["decoder"], omegaconf.listconfig.ListConfig
    ):
        for i, d_name in enumerate(cfg["decoder"]):
            decoder = models.decoder_dict[d_name](padding=padding, **decoder_kwargs[i])
            decoders.append(decoder)
    else:
        # models.decoder_dict: 一个将decoder中字符串映射到对应模型的字典：
        # {'simple_fc': <class 'src.third_party.ConvONets.conv_onet.models.decoder.FCDecoder'>,
        # 'simple_local': <class 'src.third_party.ConvONets.conv_onet.models.decoder.LocalDecoder'>,
        # 'simple_local_v1': <class 'src.third_party.ConvONets.conv_onet.models.decoder.LocalDecoderV1'>}

        decoder = models.decoder_dict[cfg["decoder"]](padding=padding, **decoder_kwargs)
        decoders.append(decoder)

    if encoder == "idx":
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        # encoder_dict 在 encoder 文件夹下的__init__.py文件中
        # encoder_dict[encoder]: <class 'src.third_party.ConvONets.encoder.encoder.LocalPoolPointnetPPFusion'>
        # encoder_kwargs包含的内容恰恰是encoder/encoder.py下LocalPoolPointnetPPFusion所需要的参数
        encoder = encoder_dict[encoder](c_dim=c_dim, padding=padding, **encoder_kwargs)
    else:
        encoder = None

    # eval(convonet_type): <class 'src.third_party.ConvONets.conv_onet.models.ConvolutionalOccupancyNetworkGeoArt'>
    # 当前的setting下: decoder分为5个，并且都是'simple_local_v1'
    if len(decoders) == 1:
        model = eval(convonet_type)(decoder, encoder)
    else:
        model = eval(convonet_type)(decoders, encoder)

    return model


def get_generator(model, cfg, **kwargs):
    """Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
    """
    vol_bound = None
    vol_info = None

    generator = generation.Generator3D(
        model,
        threshold=cfg["test"]["threshold"],
        resolution0=cfg["generation"]["resolution_0"],
        upsampling_steps=cfg["generation"]["upsampling_steps"],
        sample=cfg["generation"]["use_sampling"],
        refinement_step=cfg["generation"]["refinement_step"],
        simplify_nfaces=cfg["generation"]["simplify_nfaces"],
        input_type=cfg["data"]["input_type"],
        padding=cfg["data"]["padding"],
        vol_info=vol_info,
        vol_bound=vol_bound,
    )
    return generator
