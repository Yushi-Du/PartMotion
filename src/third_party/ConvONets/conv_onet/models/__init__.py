import torch
import torch.nn as nn
from torch import distributions as dist

from src.third_party.ConvONets.conv_onet.models import decoder

from ipdb import set_trace

# Decoder dictionary
decoder_dict = {
    "simple_fc": decoder.FCDecoder,
    "simple_local": decoder.LocalDecoder,
    "simple_local_v1": decoder.LocalDecoderV1,
    "simple_local_v2": decoder.LocalDecoderV2,
    "simple_local_v2_ablation": decoder.LocalDecoderV2_Ablation,
    "simple_local_v3": decoder.LocalDecoderV3,
    "ablation_new": decoder.AblationDecoder_New,
    "ablation_implicit": decoder.AblationDecoder_Implicit,
    "6dim_output": decoder.Six_dim_out,
}


class ConvolutionalOccupancyNetworkGeoArt(nn.Module):
    def __init__(self, decoders, encoder=None):
        # 从/home/wuruihai/Ditto-master/src/third_party/ConvONets/conv_onet/config.py处接受encoder和decoder
        super().__init__()

        (
            self.decoder_occ,
            self.decoder_seg,
            self.decoder_joint_type,
            self.decoder_revolute,
            self.decoder_prismatic,
        ) = decoders
        # 五个都是simple_local_v1

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None

    def forward(self, input_0, input_1, p_occ, p_seg, state_start, state_end, state_label, return_feature=False, **kwargs):
        """Performs a forward pass through the network.
        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            state_label: start和end之间的"开度"差
        """
        #############

        c, task_feature = self.encoder(input_0, input_1, state_start, state_end, state_label)
        # c: dict('geo_grid', 'corr_xy', 'corr_yz', 'corr_xz')
        # c['geo_grid'].shape: (batch_size, 64, 32, 32, 32)
        # c['corr_xy'].shape: (batch_size, 64, 64, 64)
        # self.decoder_occ, ..., self.decoder_prismatic:
        # 全都是src.third_party.ConvONets.conv_onet.models.decoder.LocalDecoderV1
        occ_logits = self.decoder_occ(p_occ, c, task_feature, **kwargs) # occ_logits: (batch_size, 100000)
        seg_logits = self.decoder_seg(p_seg, c, task_feature, **kwargs) # seg_logits: (batch_size, 4393)
        joint_type_logits = self.decoder_joint_type(p_seg, c, task_feature, **kwargs) # joint_type_logits: (batch_size, 4393)
        joint_param_r = self.decoder_revolute(p_seg, c, task_feature, **kwargs) # (batch_size, 4393, 8)
        joint_param_p = self.decoder_prismatic(p_seg, c, task_feature, **kwargs) # (batch_size, 4393, 4)

        if return_feature:
            return (
                occ_logits,
                seg_logits,
                joint_type_logits,
                joint_param_r,
                joint_param_p,
                c,
            )
        else:
            return (
                occ_logits,
                seg_logits,
                joint_type_logits,
                joint_param_r,
                joint_param_p,
            )

    # 以下这些都只在test时用到
    def decode_joints(self, p, c, task_feat):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """
        joint_type_logits = self.decoder_joint_type(p, c, task_feat)
        joint_param_r = self.decoder_revolute(p, c, task_feat)
        joint_param_p = self.decoder_prismatic(p, c, task_feat)
        return joint_type_logits, joint_param_r, joint_param_p

    def decode_occ(self, p, c, task_feat, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        # here
        logits = self.decoder_occ(p, c, task_feat, **kwargs)
        return logits

    def decode_seg(self, p, c, task_feat, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_seg(p, c, task_feat, **kwargs)
        return logits

    def encode_inputs(self, input_0, input_1, state_start, state_end, state_target):
        """Encodes the input.
        Args:
            input (tensor): the input
        """
        # type(self.encoder): <class 'src.third_party.ConvONets.encoder.encoder.LocalPoolPointnetPPFusion'>
        # print(f'from encode_inputs: self.encoder: {type(self.encoder)}')

        if self.encoder is not None:
            c, task_feat = self.encoder(input_0, input_1, state_start, state_end, state_target)
        else:
            # Return inputs?
            c = torch.empty(input_0.size(0), 0)

        return c, task_feat

# 我们的模型(已弃用)
class ConvolutionalOccupancyNetworkGeoMapping(nn.Module):
    def __init__(self, decoders, encoder=None):
        # 从/home/wuruihai/Ditto-master/src/third_party/ConvONets/conv_onet/config.py处接受encoder和decoder
        super().__init__()

        (
            self.decoder_occ,
            self.decoder_seg,
            self.decoder_joint_type,
            self.decoder_revolute,
            self.decoder_prismatic,
            self.decoder_trans_mat,
        ) = decoders
        # 前五个都是simple_local_v1, 最后一个是simple_local_v2

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None

    def forward(self, input_0, input_1, p_occ, p_seg, state_start, state_end, state_label, return_feature=False, **kwargs):
        """Performs a forward pass through the network.
        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            state_label: start和end之间的"开度"差
        """
        #############

        c, task_feature = self.encoder(input_0, input_1, state_start, state_end, state_label)
        # c: dict('geo_grid', 'corr_xy', 'corr_yz', 'corr_xz')
        # c['geo_grid'].shape: (batch_size, 64, 32, 32, 32)
        # c['corr_xy'].shape: (batch_size, 64, 64, 64)
        # self.decoder_occ, ..., self.decoder_prismatic:
        # 全都是src.third_party.ConvONets.conv_onet.models.decoder.LocalDecoderV1
        occ_logits = self.decoder_occ(p_occ, c, task_feature, **kwargs) # occ_logits: (batch_size, 100000)
        seg_logits = self.decoder_seg(p_seg, c, task_feature, **kwargs) # seg_logits: (batch_size, 4393)
        joint_type_logits = self.decoder_joint_type(p_seg, c, task_feature, **kwargs) # joint_type_logits: (batch_size, 4393)
        joint_param_r = self.decoder_revolute(p_seg, c, task_feature, **kwargs) # (batch_size, 4393, 8)
        joint_param_p = self.decoder_prismatic(p_seg, c, task_feature, **kwargs) # (batch_size, 4393, 4)
        trans_mat = self.decoder_trans_mat(input_0, c, task_feature, **kwargs) # (batch_size, 8192, 16)

        # 默认return_feature都是false
        if return_feature:
            return (
                occ_logits,
                seg_logits,
                joint_type_logits,
                joint_param_r,
                joint_param_p,
                c,
                trans_mat,
            )
        else:
            return (
                occ_logits,
                seg_logits,
                joint_type_logits,
                joint_param_r,
                joint_param_p,
                trans_mat,
            )

    # 以下这些都只在test时用到
    def decode_joints(self, p, c, task_feat):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """
        joint_type_logits = self.decoder_joint_type(p, c, task_feat)
        joint_param_r = self.decoder_revolute(p, c, task_feat)
        joint_param_p = self.decoder_prismatic(p, c, task_feat)
        return joint_type_logits, joint_param_r, joint_param_p

    def decode_occ(self, p, c, task_feat, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        # here
        logits = self.decoder_occ(p, c, task_feat, **kwargs)
        return logits

    def decode_seg(self, p, c, task_feat, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_seg(p, c, task_feat, **kwargs)
        return logits

    def decode_trans_mat(self, p, c, task_feat, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_trans_mat(p, c, task_feat, **kwargs)
        return logits

    def encode_inputs(self, input_0, input_1, state_start, state_end, state_target):
        """Encodes the input.
        Args:
            input (tensor): the input
        """
        # type(self.encoder): <class 'src.third_party.ConvONets.encoder.encoder.LocalPoolPointnetPPFusion'>
        # print(f'from encode_inputs: self.encoder: {type(self.encoder)}')

        if self.encoder is not None:
            c, task_feat = self.encoder(input_0, input_1, state_start, state_end, state_target)
        else:
            # Return inputs?
            c = torch.empty(input_0.size(0), 0)

        return c, task_feat

# given gt seg (现在使用的，生成trans_mat的版本)
class ConvolutionalOccupancyNetworkGeoMapping_New(nn.Module):
    def __init__(self, decoders, encoder=None):
        # 从/home/wuruihai/Ditto-master/src/third_party/ConvONets/conv_onet/config.py处接受encoder和decoder
        super().__init__()

        (self.decoder_trans_mat) = decoders
        # simple_local_v2

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None

    def forward(self, input_0, state_start, state_end, return_feature=False, **kwargs):
        """Performs a forward pass through the network.
        Args:
            input_0 (tensor): sampled points, B*N*D (B, N, 4)
            inputs (tensor): conditioning input, B*N*3
        """
        #############

        c, task_feature = self.encoder(input_0, state_start, state_end)
        # c: dict('geo_grid', 'corr_xy', 'corr_yz', 'corr_xz')
        # c['geo_grid'].shape: (batch_size, 64, 32, 32, 32)
        # c['corr_xy'].shape: (batch_size, 64, 64, 64)
        # self.decoder_occ, ..., self.decoder_prismatic:
        # 全都是src.third_party.ConvONets.conv_onet.models.decoder.LocalDecoderV1
        # decode时的输入为query point, 所以只需要输入前三维即可
        trans_mat = self.decoder_trans_mat(input_0[:, :, 0:3], c, task_feature, **kwargs) # (batch_size, 8192, 16)

        # 默认return_feature都是false
        return (
            trans_mat
        )

    # 以下这些都只在test时用到
    def decode_trans_mat(self, p, c, task_feat, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_trans_mat(p, c, task_feat, **kwargs)
        return logits

    def encode_inputs(self, input_0, state_start, state_end):
        """Encodes the input.
        Args:
            input (tensor): the input
        """
        # type(self.encoder): <class 'src.third_party.ConvONets.encoder.encoder.LocalPoolPointnetPPFusion'>
        # print(f'from encode_inputs: self.encoder: {type(self.encoder)}')

        if self.encoder is not None:
            c, task_feat = self.encoder(input_0, state_start, state_end)
        else:
            # Return inputs?
            c = torch.empty(input_0.size(0), 0)

        return c, task_feat

# given gt seg, 3 frames (现在使用的，生成trans_mat的版本)
class ConvolutionalOccupancyNetworkGeoMapping_3frames(nn.Module):
    def __init__(self, decoders, encoder=None, sample_grid=False):
        # 从/home/wuruihai/Ditto-master/src/third_party/ConvONets/conv_onet/config.py处接受encoder和decoder
        super().__init__()

        self.if_sample_grid = sample_grid

        (self.decoder_trans_mat) = decoders
        # simple_local_v2

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None

    def sample_grid(self):
        x = torch.arange(0, 1, 1 / 32)
        y = torch.arange(0, 1, 1 / 32)
        z = torch.arange(0, 1, 1 / 32)

        cor_x, cor_y, cor_z = torch.meshgrid(x, y, z)
        cor_x = cor_x.reshape(-1, 1)
        cor_y = cor_y.reshape(-1, 1)
        cor_z = cor_z.reshape(-1, 1)
        grid = torch.cat((cor_x, cor_y, cor_z), dim=1)
        grid = grid.unsqueeze(0)
        return grid

    def forward(self, input_start, input_end, state_start, state_end, state_target, return_feature=False, **kwargs):
        """Performs a forward pass through the network.
        Args:
            input_0 (tensor): sampled points, B*N*D (B, N, 4)
            inputs (tensor): conditioning input, B*N*3
        """
        #############

        c, task_feature = self.encoder(input_start, input_end, state_start, state_end, state_target)
        # c: dict('geo_grid', 'corr_xy', 'corr_yz', 'corr_xz')
        # c['geo_grid'].shape: (batch_size, 64, 32, 32, 32)
        # c['corr_xy'].shape: (batch_size, 64, 64, 64)
        # self.decoder_occ, ..., self.decoder_prismatic:
        # 全都是src.third_party.ConvONets.conv_onet.models.decoder.LocalDecoderV1
        # decode时的输入为query point, 所以只需要输入前三维即可
        trans_mat = self.decoder_trans_mat(input_start[:, :, 0:3], c, task_feature, **kwargs) # (batch_size, 8192, 16)
        if self.if_sample_grid is True:
            grids = self.sample_grid().to(task_feature.device)
            grid_mat = self.decoder_trans_mat(grids, c, task_feature, **kwargs)

        # 默认return_feature都是false
        if self.if_sample_grid is True:
            return (
                trans_mat, grids, grid_mat
            )
        return (
            trans_mat
        )

    # 以下这些都只在test时用到
    def decode_trans_mat(self, p, c, task_feat, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_trans_mat(p, c, task_feat, **kwargs)
        return logits

    def encode_inputs(self, input_start, input_end, state_start, state_end, state_target):
        """Encodes the input.
        Args:
            input (tensor): the input
        """
        # type(self.encoder): <class 'src.third_party.ConvONets.encoder.encoder.LocalPoolPointnetPPFusion'>
        # print(f'from encode_inputs: self.encoder: {type(self.encoder)}')

        if self.encoder is not None:
            c, task_feat = self.encoder(input_start, input_end, state_start, state_end, state_target)
        else:
            # Return inputs?
            c = torch.empty(input_0.size(0), 0)

        return c, task_feat

# 现在使用的用于插值的模型
class ConvolutionalOccupancyNetworkGeoMapping_3frames_interpolation(nn.Module):
    def __init__(self, decoders, encoder=None, sample_grid=False):
        # 从/home/wuruihai/Ditto-master/src/third_party/ConvONets/conv_onet/config.py处接受encoder和decoder
        super().__init__()

        self.if_sample_grid = sample_grid

        (self.decoder_trans_mat) = decoders
        # simple_local_v2

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None

    def sample_grid(self):
        x = torch.arange(0, 1, 1 / 32)
        y = torch.arange(0, 1, 1 / 32)
        z = torch.arange(0, 1, 1 / 32)

        cor_x, cor_y, cor_z = torch.meshgrid(x, y, z)
        cor_x = cor_x.reshape(-1, 1)
        cor_y = cor_y.reshape(-1, 1)
        cor_z = cor_z.reshape(-1, 1)
        grid = torch.cat((cor_x, cor_y, cor_z), dim=1)
        grid = grid.unsqueeze(0)
        return grid

    def forward(self, input_start, input_end, state_start, state_end, state_target, return_feature=False, **kwargs):
        """Performs a forward pass through the network.
        Args:
            input_0 (tensor): sampled points, B*N*D (B, N, 4)
            inputs (tensor): conditioning input, B*N*3
        """
        #############

        c, task_feature = self.encoder(input_start, input_end, state_start, state_end, state_target)
        # c: dict('geo_grid', 'corr_xy', 'corr_yz', 'corr_xz')
        # c['geo_grid'].shape: (batch_size, 64, 32, 32, 32)
        # c['corr_xy'].shape: (batch_size, 64, 64, 64)
        # self.decoder_occ, ..., self.decoder_prismatic:
        # 全都是src.third_party.ConvONets.conv_onet.models.decoder.LocalDecoderV1
        # decode时的输入为query point, 所以只需要输入前三维即可
        trans_mat = self.decoder_trans_mat(input_start[:, :, 0:3], c, task_feature, **kwargs) # (batch_size, 8192, 16)
        if self.if_sample_grid is True:
            grids = self.sample_grid().to(task_feature.device)
            grid_mat = self.decoder_trans_mat(grids, c, task_feature, **kwargs)

        # 默认return_feature都是false
        if self.if_sample_grid is True:
            return (
                trans_mat, grids, grid_mat
            )
        return (
            trans_mat
        )

    # 以下这些都只在test时用到
    def decode_trans_mat(self, p, c, task_feat, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_trans_mat(p, c, task_feat, **kwargs)
        return logits

    def encode_inputs(self, input_start, input_end, state_start, state_end, state_target):
        """Encodes the input.
        Args:
            input (tensor): the input
        """
        # type(self.encoder): <class 'src.third_party.ConvONets.encoder.encoder.LocalPoolPointnetPPFusion'>
        # print(f'from encode_inputs: self.encoder: {type(self.encoder)}')

        if self.encoder is not None:
            c, task_feat = self.encoder(input_start, input_end, state_start, state_end, state_target)
        else:
            # Return inputs?
            c = torch.empty(input_0.size(0), 0)

        return c, task_feat

class ConvolutionalOccupancyNetworkGeoMapping_3frames_extrapolation(nn.Module):
    def __init__(self, decoders, encoder=None, sample_grid=False):
        # 从/home/wuruihai/Ditto-master/src/third_party/ConvONets/conv_onet/config.py处接受encoder和decoder
        super().__init__()

        self.if_sample_grid = sample_grid

        (self.decoder_trans_mat) = decoders
        # simple_local_v2

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None

    def sample_grid(self):
        x = torch.arange(0, 1, 1 / 32)
        y = torch.arange(0, 1, 1 / 32)
        z = torch.arange(0, 1, 1 / 32)

        cor_x, cor_y, cor_z = torch.meshgrid(x, y, z)
        cor_x = cor_x.reshape(-1, 1)
        cor_y = cor_y.reshape(-1, 1)
        cor_z = cor_z.reshape(-1, 1)
        grid = torch.cat((cor_x, cor_y, cor_z), dim=1)
        grid = grid.unsqueeze(0)
        return grid

    def forward(self, input_start, input_end, state_start, state_end, state_target, return_feature=False, **kwargs):
        """Performs a forward pass through the network.
        Args:
            input_0 (tensor): sampled points, B*N*D (B, N, 4)
            inputs (tensor): conditioning input, B*N*3
        """
        #############

        c, task_feature = self.encoder(input_start, input_end, state_start, state_end, state_target)
        # c: dict('geo_grid', 'corr_xy', 'corr_yz', 'corr_xz')
        # c['geo_grid'].shape: (batch_size, 64, 32, 32, 32)
        # c['corr_xy'].shape: (batch_size, 64, 64, 64)
        # self.decoder_occ, ..., self.decoder_prismatic:
        # 全都是src.third_party.ConvONets.conv_onet.models.decoder.LocalDecoderV1
        # decode时的输入为query point, 所以只需要输入前三维即可
        trans_mat = self.decoder_trans_mat(input_start[:, :, 0:3], c, task_feature, **kwargs) # (batch_size, 8192, 16)
        if self.if_sample_grid is True:
            grids = self.sample_grid().to(task_feature.device)
            grid_mat = self.decoder_trans_mat(grids, c, task_feature, **kwargs)

        # 默认return_feature都是false
        if self.if_sample_grid is True:
            return (
                trans_mat, grids, grid_mat
            )
        return (
            trans_mat
        )

    # 以下这些都只在test时用到
    def decode_trans_mat(self, p, c, task_feat, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_trans_mat(p, c, task_feat, **kwargs)
        return logits

    def encode_inputs(self, input_start, input_end, state_start, state_end, state_target):
        """Encodes the input.
        Args:
            input (tensor): the input
        """
        # type(self.encoder): <class 'src.third_party.ConvONets.encoder.encoder.LocalPoolPointnetPPFusion'>
        # print(f'from encode_inputs: self.encoder: {type(self.encoder)}')

        if self.encoder is not None:
            c, task_feat = self.encoder(input_start, input_end, state_start, state_end, state_target)
        else:
            # Return inputs?
            c = torch.empty(input_0.size(0), 0)

        return c, task_feat

# 现在使用的用于可视化grid的模型
class ConvolutionalOccupancyNetworkGeoMapping_3frames_grid_visualization(nn.Module):
    def __init__(self, decoders, encoder=None, sample_grid=True):
        # 从/home/wuruihai/Ditto-master/src/third_party/ConvONets/conv_onet/config.py处接受encoder和decoder
        super().__init__()

        self.if_sample_grid = sample_grid

        (self.decoder_trans_mat) = decoders
        # simple_local_v2

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None

    # def sample_grid(self):
    #     x = torch.arange(-1, 1, 1 / 16)
    #     y = torch.arange(-1, 1, 1 / 16)
    #     z = torch.arange(-1, 1, 1 / 16)
    #
    #     cor_x, cor_y, cor_z = torch.meshgrid(x, y, z)
    #     cor_x = cor_x.reshape(-1, 1)
    #     cor_y = cor_y.reshape(-1, 1)
    #     cor_z = cor_z.reshape(-1, 1)
    #     grid = torch.cat((cor_x, cor_y, cor_z), dim=1)
    #     grid = grid.unsqueeze(0)
    #     return grid
    def sample_grid(self):
        x = torch.arange(-1, 1, 1 / 16)
        y = torch.arange(-1, 1, 1 / 16)
        z = torch.arange(-1, 1, 1 / 16)

        cor_x, cor_y, cor_z = torch.meshgrid(x, y, z)
        cor_x = cor_x.reshape(-1, 1)
        cor_y = cor_y.reshape(-1, 1)
        cor_z = cor_z.reshape(-1, 1)
        grid = torch.cat((cor_x, cor_y, cor_z), dim=1)
        grid = grid.unsqueeze(0)
        return grid

    def forward(self, input_start, input_end, state_start, state_end, state_target, return_feature=False, **kwargs):
        """Performs a forward pass through the network.
        Args:
            input_0 (tensor): sampled points, B*N*D (B, N, 4)
            inputs (tensor): conditioning input, B*N*3
        """
        #############

        c, task_feature = self.encoder(input_start, input_end, state_start, state_end, state_target)
        # c: dict('geo_grid', 'corr_xy', 'corr_yz', 'corr_xz')
        # c['geo_grid'].shape: (batch_size, 64, 32, 32, 32)
        # c['corr_xy'].shape: (batch_size, 64, 64, 64)
        # self.decoder_occ, ..., self.decoder_prismatic:
        # 全都是src.third_party.ConvONets.conv_onet.models.decoder.LocalDecoderV1
        # decode时的输入为query point, 所以只需要输入前三维即可
        trans_mat = self.decoder_trans_mat(input_start[:, :, 0:3], c, task_feature, **kwargs) # (batch_size, 8192, 16)
        if self.if_sample_grid is True:
            grids = self.sample_grid().to(task_feature.device)
            grid_mat = self.decoder_trans_mat(grids, c, task_feature, **kwargs)

        # 默认return_feature都是false
        if self.if_sample_grid is True:
            return (
                trans_mat, grids, grid_mat
            )
        return (
            trans_mat
        )

    # 以下这些都只在test时用到
    def decode_trans_mat(self, p, c, task_feat, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_trans_mat(p, c, task_feat, **kwargs)
        return logits

    def encode_inputs(self, input_start, input_end, state_start, state_end, state_target):
        """Encodes the input.
        Args:
            input (tensor): the input
        """
        # type(self.encoder): <class 'src.third_party.ConvONets.encoder.encoder.LocalPoolPointnetPPFusion'>
        # print(f'from encode_inputs: self.encoder: {type(self.encoder)}')

        if self.encoder is not None:
            c, task_feat = self.encoder(input_start, input_end, state_start, state_end, state_target)
        else:
            # Return inputs?
            c = torch.empty(input_0.size(0), 0)

        return c, task_feat

# given gt seg (baseline使用的, 直接生成的版本)
class ConvolutionalOccupancyNetworkGeoMapping_New_Implicit(nn.Module):
    def __init__(self, decoders, encoder=None):
        # 从/home/wuruihai/Ditto-master/src/third_party/ConvONets/conv_onet/config.py处接受encoder和decoder
        super().__init__()

        (self.decoder_implicit) = decoders
        # 前五个都是simple_local_v1, 最后一个是simple_local_v2

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None

    def forward(self, input_0, state_start, state_end, return_feature=False, **kwargs):
        """Performs a forward pass through the network.
        Args:
            input_0 (tensor): sampled points, B*N*D (B, N, 4)
            inputs (tensor): conditioning input, B*N*3
        """
        #############
        batch_size = input_0.shape[0]

        c, task_feature = self.encoder(input_0, state_start, state_end)
        # c: dict('geo_grid', 'corr_xy', 'corr_yz', 'corr_xz')
        # c['geo_grid'].shape: (batch_size, 64, 32, 32, 32)
        # c['corr_xy'].shape: (batch_size, 64, 64, 64)
        # self.decoder_occ, ..., self.decoder_prismatic:
        # 全都是src.third_party.ConvONets.conv_onet.models.decoder.LocalDecoderV1
        # decode时的输入为query point, 所以只需要输入前三维即可
        output_p, art_code = self.decoder_implicit(input_0[:, :, 0:3], c, task_feature, **kwargs) # (batch_size, 2048*3)

        output_p = output_p.reshape(batch_size, 8192, 3)

        # 默认return_feature都是false
        return (
            output_p, art_code
        )

    # 以下这些都只在test时用到
    def decode_implicit(self, p, c, task_feat, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_implicit(p, c, task_feat, **kwargs)
        return logits

    def encode_inputs(self, input_0, state_start, state_end):
        """Encodes the input.
        Args:
            input (tensor): the input
        """
        # type(self.encoder): <class 'src.third_party.ConvONets.encoder.encoder.LocalPoolPointnetPPFusion'>
        # print(f'from encode_inputs: self.encoder: {type(self.encoder)}')

        if self.encoder is not None:
            c, task_feat = self.encoder(input_0, state_start, state_end)
        else:
            # Return inputs?
            c = torch.empty(input_0.size(0), 0)

        return c, task_feat

# ablation我们的
class OnlyPointNet_New(nn.Module):
    def __init__(self, decoders, encoder=None):
        # 从/home/wuruihai/Ditto-master/src/third_party/ConvONets/conv_onet/config.py处接受encoder和decoder
        super().__init__()

        (self.decoder_trans_mat) = decoders
        # AblationDecoder_New

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None

    def forward(self, input_0, state_start, state_end, return_feature=False, **kwargs):
        """Performs a forward pass through the network.
        Args:
            input_0 (tensor): sampled points, B*N*D (B, N, 4)
            inputs (tensor): conditioning input, B*N*3
        """
        #############

        c, task_feature = self.encoder(input_0, state_start, state_end)
        # c: (B, N, 64)
        # self.decoder_occ, ..., self.decoder_prismatic:
        # 全都是src.third_party.ConvONets.conv_onet.models.decoder.LocalDecoderV1
        # decode时的输入为query point, 所以只需要输入前三维即可
        trans_mat = self.decoder_trans_mat(input_0[:, :, 0:3], c, task_feature, **kwargs) # (batch_size, 8192, 16)

        # 默认return_feature都是false
        return (
            trans_mat
        )

    # 以下这些都只在test时用到
    def decode_trans_mat(self, p, c, task_feat, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_trans_mat(p, c, task_feat, **kwargs)
        return logits

    def encode_inputs(self, input_0, state_start, state_end):
        """Encodes the input.
        Args:
            input (tensor): the input
        """
        # type(self.encoder): <class 'src.third_party.ConvONets.encoder.encoder.LocalPoolPointnetPPFusion'>
        # print(f'from encode_inputs: self.encoder: {type(self.encoder)}')

        if self.encoder is not None:
            c, task_feat = self.encoder(input_0, state_start, state_end)
        else:
            # Return inputs?
            c = torch.empty(input_0.size(0), 0)

        return c, task_feat

# ablation baseline
class OnlyPointNet_Implicit(nn.Module):
    def __init__(self, decoders, encoder=None):
        # 从/home/wuruihai/Ditto-master/src/third_party/ConvONets/conv_onet/config.py处接受encoder和decoder
        super().__init__()

        (self.decoder_implicit) = decoders

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None

    def forward(self, input_0, state_start, state_end, return_feature=False, **kwargs):
        """Performs a forward pass through the network.
        Args:
            input_0 (tensor): sampled points, B*N*D (B, N, 4)
            inputs (tensor): conditioning input, B*N*3
        """
        #############
        batch_size = input_0.shape[0]

        c, task_feature = self.encoder(input_0, state_start, state_end)
        # c: dict('geo_grid', 'corr_xy', 'corr_yz', 'corr_xz')
        # c['geo_grid'].shape: (batch_size, 64, 32, 32, 32)
        # c['corr_xy'].shape: (batch_size, 64, 64, 64)
        # self.decoder_occ, ..., self.decoder_prismatic:
        # 全都是src.third_party.ConvONets.conv_onet.models.decoder.LocalDecoderV1
        # decode时的输入为query point, 所以只需要输入前三维即可
        output_p, art_code = self.decoder_implicit(input_0[:, :, 0:3], c, task_feature, **kwargs) # (batch_size, 2048*3)

        # 默认return_feature都是false
        return (
            output_p, art_code
        )

    # 以下这些都只在test时用到
    def decode_implicit(self, p, c, task_feat, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_implicit(p, c, task_feat, **kwargs)
        return logits

    def encode_inputs(self, input_0, state_start, state_end):
        """Encodes the input.
        Args:
            input (tensor): the input
        """
        # type(self.encoder): <class 'src.third_party.ConvONets.encoder.encoder.LocalPoolPointnetPPFusion'>
        # print(f'from encode_inputs: self.encoder: {type(self.encoder)}')

        if self.encoder is not None:
            c, task_feat = self.encoder(input_0, state_start, state_end)
        else:
            # Return inputs?
            c = torch.empty(input_0.size(0), 0)

        return c, task_feat
