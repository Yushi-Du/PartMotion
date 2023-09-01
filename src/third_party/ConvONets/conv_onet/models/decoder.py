import torch
import torch.nn as nn
import torch.nn.functional as F

from src.third_party.ConvONets.common import (
    map2local,
    normalize_3d_coordinate,
    normalize_coordinate,
)
from src.third_party.ConvONets.layers import ResnetBlockFC
from pdb import set_trace


class FCDecoder(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.
    Args:
    dim (int): input dimension
    c_dim (int): dimension of latent conditioned code c
    out_dim (int): dimension of latent conditioned code c
    leaky (bool): whether to use leaky ReLUs
    sample_mode (str): sampling feature strategy, bilinear|nearest
    padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
    ):
        super().__init__()
        self.c_dim = c_dim

        self.fc = nn.Linear(dim + c_dim, out_dim)
        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(
            c,
            vgrid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(
            p.clone(), padding=self.padding
        )  # normalize to the range of (0, 1)

        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)

        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(
                c,
                vgrid,
                padding_mode="border",
                align_corners=True,
                mode=self.sample_mode,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)

        net = self.fc(torch.cat((c, p), dim=2)).squeeze(-1)

        return net


class LocalDecoder(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
        concat_feat=False,
        concat_feat_4=False,
        no_xyz=False,
    ):
        super().__init__()
        self.concat_feat = concat_feat or concat_feat_4
        if concat_feat:
            c_dim *= 3
        elif concat_feat_4:
            c_dim *= 4
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim, hidden_size) for i in range(n_blocks)]
            )

        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for i in range(n_blocks)]
        )

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(
            c,
            vgrid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(
            p.clone(), padding=self.padding
        )  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'

        c = (
            F.grid_sample(
                c,
                vgrid,
                padding_mode="border",
                align_corners=True,
                mode=self.sample_mode,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                if "grid" in plane_type:
                    c.append(self.sample_grid_feature(p, c_plane["grid"]))
                if "xz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xz"], plane="xz"))
                if "xy" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xy"], plane="xy"))
                if "yz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["yz"], plane="yz"))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                if "grid" in plane_type:
                    c += self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
                if "xy" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
                if "yz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
                c = c.transpose(1, 2)

        p = p.float()

        if self.no_xyz:
            net = torch.zeros(p.size(0), p.size(1), self.hidden_size).to(p.device)
        else:
            net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


# different feature for different head
# 实际是用的这个！！
class LocalDecoderV1(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        task_feat_dim=768,
        hidden_size=256,
        n_blocks=5,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        feature_keys=None,
        concat_feat=True,
        padding=0.1,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.feature_keys = feature_keys
        self.concat_feat = concat_feat

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim+task_feat_dim, hidden_size) for i in range(n_blocks)]
            )

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for i in range(n_blocks)]
        )

        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        # 需改动，加上batch
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(
            c,
            vgrid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        # p: 需要输出feature的sample point. (query point, to be exact)
        # c: encode输出的grid/plane feature.
        # print('hi from sample_grid_feature')
        # print(f'p.shape: {p.shape}')

        # 根据p_nor生成vgrid喂进F.grid_sample
        # 需改动，加上batch
        p_nor = normalize_3d_coordinate(
            p.clone(), padding=self.padding
        )  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)

        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            # error here
            F.grid_sample(
                c,
                vgrid,
                padding_mode="border",
                align_corners=True,
                mode=self.sample_mode,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane, task_feature, **kwargs):
        # p: (batch_size, num_points, 3)
        # c_plane.keys(): dict_keys(['geo_grid', 'corr_xy', 'corr_yz', 'corr_xz'])
        # task_feature: (task_feat_dim) 这里没有batch_size作为第一个维度在实际跑的时候可能有问题
        # self.concat_feat: True
        # self.feature_keys: for occupancy: 'geo_grid'; for others: 'corr_xy'...
        batch_size = p.shape[0]

        if self.c_dim != 0:
            # plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                for k in self.feature_keys:
                    if "grid" in k:
                        c.append(self.sample_grid_feature(p, c_plane[k]))
                    elif "xy" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="xy"))
                    elif "yz" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="yz"))
                    elif "xz" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="xz"))
                c = torch.cat(c, dim=1)
                # c.shape: for occupancy: (batch_size, 64, 4096)
                # for others: (batch_size, 192, 2048)
                c = c.transpose(1, 2)  # 交换了后两个维度: (batch_size, num_points, 64)
                # task_feature: (batch_size, task_feat_dim)
                task_feature = task_feature.unsqueeze(1)
                task_feature = task_feature.expand(batch_size, c.shape[1], task_feature.shape[2])
                # 256是encoder中'task mlp'输出的维数

                c = torch.cat((c, task_feature), dim=2)
                # c: (batch_size, num_points, feat_dim+task_feature_dim)
            else:
                c = 0
                for k in self.feature_keys:
                    if "grid" in k:
                        c += self.sample_grid_feature(p, c_plane[k])
                    elif "xy" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="xy")
                    elif "yz" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="yz")
                    elif "xz" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="xz")
                c = c.transpose(1, 2)

        p = p.float()

        net = self.fc_p(p)
        # self.n_blocks: 5
        # net: (1, 2048, 64)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c) # net: (1, 2048, 64)
                # 5个mlp输出相加
            # 每次要过resnet层
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        # out: (batch_size, num_points, feature_dim)
        # 这里的num_points指的是对应输入的sample点数量，
        # 比如decode_occ就是config中的num_point_occ
        # feature_dim就是需要的feature维数，比如occupancy/segmentation/joint_type就都是1
        # joint_params相关的就是8或者4, 这里的输出就直接作为最终的结果了
        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

class LocalDecoderV2(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=64,
        task_feat_dim=768,
        hidden_size=256,
        n_blocks=5,
        out_dim=12,
        leaky=False,
        sample_mode="bilinear",
        feature_keys=None,
        concat_feat=True,
        padding=0.1,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.feature_keys = feature_keys
        self.concat_feat = concat_feat

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim+task_feat_dim, hidden_size) for i in range(n_blocks)]
            )

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for i in range(n_blocks)]
        )

        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        # 需改动，加上batch
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(
            c,
            vgrid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        # p: 需要输出feature的sample point. (query point, to be exact)
        # c: encode输出的grid/plane feature.
        # print('hi from sample_grid_feature')
        # print(f'p.shape: {p.shape}')

        # 根据p_nor生成vgrid喂进F.grid_sample
        # 需改动，加上batch
        p_nor = normalize_3d_coordinate(
            p.clone(), padding=self.padding
        )  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)

        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            # error here
            F.grid_sample(
                c,
                vgrid,
                padding_mode="border",
                align_corners=True,
                mode=self.sample_mode,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane, task_feature, **kwargs):
        # p: (batch_size, num_points, 3)
        # c_plane.keys(): dict_keys(['geo_grid', 'corr_xy', 'corr_yz', 'corr_xz'])
        # task_feature: (task_feat_dim) 这里没有batch_size作为第一个维度在实际跑的时候可能有问题
        # self.concat_feat: True
        # self.feature_keys: for occupancy: 'geo_grid'; for others: 'corr_xy'...
        batch_size = p.shape[0]

        if self.c_dim != 0:
            # plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                for k in self.feature_keys:
                    if "grid" in k:
                        c.append(self.sample_grid_feature(p, c_plane[k]))
                    elif "xy" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="xy"))
                    elif "yz" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="yz"))
                    elif "xz" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="xz"))
                c = torch.cat(c, dim=1)
                # c.shape: for occupancy: (batch_size, 64, 4096)
                # for others: (batch_size, 192, 2048)
                c = c.transpose(1, 2)  # 交换了后两个维度: (batch_size, num_points, 64)
                # task_feature: (batch_size, task_feat_dim)
                task_feature = task_feature.unsqueeze(1)
                task_feature = task_feature.expand(batch_size, c.shape[1], task_feature.shape[2])
                # 256是encoder中'task mlp'输出的维数

                c = torch.cat((c, task_feature), dim=2)
                # c: (batch_size, num_points, feat_dim+task_feature_dim)
            else:
                c = 0
                for k in self.feature_keys:
                    if "grid" in k:
                        c += self.sample_grid_feature(p, c_plane[k])
                    elif "xy" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="xy")
                    elif "yz" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="yz")
                    elif "xz" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="xz")
                c = c.transpose(1, 2)

        p = p.float()

        net = self.fc_p(p)
        # self.n_blocks: 5
        # net: (1, 2048, 64)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c) # net: (1, 2048, 64)
                # 5个mlp输出相加

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        # out: (batch_size, num_points, out_dim)
        # 这里的num_points指的是对应输入的sample点数量，
        # 比如decode_occ就是config中的num_point_occ
        # out_dim就是需要的feature维数，比如occupancy和segmentation就都是1
        # joint_params相关的就是8或者4
        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

class LocalDecoderV2_Ablation(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=64,
        task_feat_dim=768,
        hidden_size=256,
        n_blocks=5,
        out_dim=12,
        leaky=False,
        sample_mode="bilinear",
        feature_keys=None,
        concat_feat=True,
        padding=0.1,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.feature_keys = feature_keys
        self.concat_feat = concat_feat
        self.task_feat_dim = task_feat_dim

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim+task_feat_dim, hidden_size) for i in range(n_blocks)]
            )

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for i in range(n_blocks)]
        )

        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def forward(self, p, c, task_feature, **kwargs):
        # p: (batch_size, num_points, 3)
        # c: (batch_size, num_points, 64)
        # task_feature: (batch_size, 512)
        batch_size = p.shape[0]
        num_points = p.shape[1]

        task_feature = task_feature.unsqueeze(1)
        task_feature = task_feature.expand(batch_size, num_points, self.task_feat_dim)
        c = torch.cat((c, task_feature), dim=2)  # c: (B, N, 64+768)

        net = self.fc_p(p)
        # self.n_blocks: 5
        # net: (B, N, self.hidden_size)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)  # net: (B, N, self.hidden_size)
                # 5个mlp输出相加

            net = self.blocks[i](net)

        # net: (B, N, self.hidden_size)
        out = self.fc_out(self.actvn(net))  # out: (B, N, self.out_dim)

        # out: (batch_size, num_points, out_dim)
        # 这里的num_points指的是对应输入的sample点数量，
        # 比如decode_occ就是config中的num_point_occ
        # out_dim就是需要的feature维数，比如occupancy和segmentation就都是1
        # joint_params相关的就是8或者4
        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

class Six_dim_out(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=64,
        task_feat_dim=512,
        hidden_size=256,
        n_blocks=5,
        out_dim=6,
        leaky=False,
        sample_mode="bilinear",
        feature_keys=None,
        concat_feat=True,
        padding=0.1,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.feature_keys = feature_keys
        self.concat_feat = concat_feat

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim+task_feat_dim, hidden_size) for i in range(n_blocks)]
            )

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for i in range(n_blocks)]
        )

        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        # 需改动，加上batch
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(
            c,
            vgrid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        # p: 需要输出feature的sample point. (query point, to be exact)
        # c: encode输出的grid/plane feature.
        # print('hi from sample_grid_feature')
        # print(f'p.shape: {p.shape}')

        # 根据p_nor生成vgrid喂进F.grid_sample
        # 需改动，加上batch
        p_nor = normalize_3d_coordinate(
            p.clone(), padding=self.padding
        )  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)

        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            # error here
            F.grid_sample(
                c,
                vgrid,
                padding_mode="border",
                align_corners=True,
                mode=self.sample_mode,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane, task_feature, **kwargs):
        # p: (batch_size, num_points, 3)
        # c_plane.keys(): dict_keys(['geo_grid', 'corr_xy', 'corr_yz', 'corr_xz'])
        # task_feature: (task_feat_dim) 这里没有batch_size作为第一个维度在实际跑的时候可能有问题
        # self.concat_feat: True
        # self.feature_keys: for occupancy: 'geo_grid'; for others: 'corr_xy'...
        batch_size = p.shape[0]

        if self.c_dim != 0:
            # plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                for k in self.feature_keys:
                    if "grid" in k:
                        c.append(self.sample_grid_feature(p, c_plane[k]))
                    elif "xy" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="xy"))
                    elif "yz" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="yz"))
                    elif "xz" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="xz"))
                c = torch.cat(c, dim=1)
                # c.shape: for occupancy: (batch_size, 64, 4096)
                # for others: (batch_size, 192, 2048)
                c = c.transpose(1, 2)  # 交换了后两个维度: (batch_size, num_points, 64)
                # task_feature: (batch_size, task_feat_dim)
                task_feature = task_feature.unsqueeze(1)
                task_feature = task_feature.expand(batch_size, c.shape[1], task_feature.shape[2])
                # 256是encoder中'task mlp'输出的维数

                c = torch.cat((c, task_feature), dim=2)
                # c: (batch_size, num_points, feat_dim+task_feature_dim)
            else:
                c = 0
                for k in self.feature_keys:
                    if "grid" in k:
                        c += self.sample_grid_feature(p, c_plane[k])
                    elif "xy" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="xy")
                    elif "yz" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="yz")
                    elif "xz" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="xz")
                c = c.transpose(1, 2)

        p = p.float()

        net = self.fc_p(p)
        # self.n_blocks: 5
        # net: (1, 2048, 64)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c) # net: (1, 2048, 64)
                # 5个mlp输出相加

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        # out: (batch_size, num_points, out_dim)
        # 这里的num_points指的是对应输入的sample点数量，
        # 比如decode_occ就是config中的num_point_occ
        # out_dim就是需要的feature维数，比如occupancy和segmentation就都是1
        # joint_params相关的就是8或者4
        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

# class LocalDecoderV3(nn.Module):
#     """Decoder.
#         Instead of conditioning on global features, on plane/volume local features.
#
#     Args:
#         dim (int): input dimension
#         c_dim (int): dimension of latent conditioned code c
#         hidden_size (int): hidden size of Decoder network
#         n_blocks (int): number of blocks ResNetBlockFC layers
#         leaky (bool): whether to use leaky ReLUs
#         sample_mode (str): sampling feature strategy, bilinear|nearest
#         padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
#     """
#
#     def __init__(
#         self,
#         dim=3,
#         c_dim=2048+512,
#         hidden_size=8192,
#         out_dim=8192*3,
#         leaky=False,
#         sample_mode="bilinear",
#         feature_keys=None,
#         concat_feat=True,
#         padding=0.1,
#     ):
#         super().__init__()
#         self.c_dim = c_dim
#         self.hidden_size = hidden_size
#         self.feature_keys = feature_keys
#         self.concat_feat = concat_feat
#
#         self.linear1 = nn.Linear(8*8*8*32+512, 2048+512)
#         self.linear2 = nn.Linear(2048+512, 8192)
#         self.linear3 = nn.Linear(8192, 8192*3)
#
#     def forward(self, c_plane, task_feature, **kwargs):
#         # c_plane.keys(): dict_keys(['geo_grid', 'corr_xy', 'corr_yz', 'corr_xz'])
#         # self.concat_feat: True
#         # self.feature_keys: for occupancy: 'geo_grid'; for others: 'corr_xy'...
#
#         c = c_plane['geo_grid'] # (batch_size, 2048)
#         c = torch.cat((c, task_feature), dim=1) # (batch_size, 2048+512)
#
#         # self.n_blocks: 5
#         c = F.relu(self.linear1(c))
#         art_code = c[:, 0]
#         c = F.relu(self.linear2(c))
#         c = self.linear3(c)
#
#         # out: (batch_size, num_points, out_dim)
#         # 这里的num_points指的是对应输入的sample点数量，
#         # 比如decode_occ就是config中的num_point_occ
#         # out_dim就是需要的feature维数，比如occupancy和segmentation就都是1
#         # joint_params相关的就是8或者4
#         return c, art_code

class LocalDecoderV3(nn.Module):
    """Decoder.
            Instead of conditioning on global features, on plane/volume local features.

        Args:
            dim (int): input dimension
            c_dim (int): dimension of latent conditioned code c
            hidden_size (int): hidden size of Decoder network
            n_blocks (int): number of blocks ResNetBlockFC layers
            leaky (bool): whether to use leaky ReLUs
            sample_mode (str): sampling feature strategy, bilinear|nearest
            padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        """

    def __init__(
            self,
            dim=3,
            c_dim=32,
            task_feat_dim=512,
            hidden_size=256,
            n_blocks=5,
            out_dim=3,
            leaky=False,
            sample_mode="bilinear",
            feature_keys=None,
            concat_feat=True,
            padding=0.1,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.feature_keys = feature_keys
        self.concat_feat = concat_feat

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim + task_feat_dim, hidden_size) for i in range(n_blocks)]
            )

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for i in range(n_blocks)]
        )

        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        # 需改动，加上batch
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(
            c,
            vgrid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        # p: 需要输出feature的sample point. (query point, to be exact)
        # c: encode输出的grid/plane feature.
        # print('hi from sample_grid_feature')
        # print(f'p.shape: {p.shape}')

        # 根据p_nor生成vgrid喂进F.grid_sample
        # 需改动，加上batch
        p_nor = normalize_3d_coordinate(
            p.clone(), padding=self.padding
        )  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)

        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            # error here
            F.grid_sample(
                c,
                vgrid,
                padding_mode="border",
                align_corners=True,
                mode=self.sample_mode,
            )
                .squeeze(-1)
                .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane, task_feature, **kwargs):
        # p: (batch_size, num_points, 3)
        # c_plane.keys(): dict_keys(['geo_grid', 'corr_xy', 'corr_yz', 'corr_xz'])
        # task_feature: (task_feat_dim) 这里没有batch_size作为第一个维度在实际跑的时候可能有问题
        # self.concat_feat: True
        # self.feature_keys: for occupancy: 'geo_grid'; for others: 'corr_xy'...
        batch_size = p.shape[0]
        art_code = task_feature[:, 0]

        if self.c_dim != 0:
            # plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                for k in self.feature_keys:
                    if "grid" in k:
                        c.append(self.sample_grid_feature(p, c_plane[k]))
                    elif "xy" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="xy"))
                    elif "yz" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="yz"))
                    elif "xz" in k:
                        c.append(self.sample_plane_feature(p, c_plane[k], plane="xz"))
                c = torch.cat(c, dim=1)
                # c.shape: for occupancy: (batch_size, 64, 4096)
                # for others: (batch_size, 192, 2048)
                c = c.transpose(1, 2)  # 交换了后两个维度: (batch_size, num_points, 64)
                # task_feature: (batch_size, task_feat_dim)
                task_feature = task_feature.unsqueeze(1)
                task_feature = task_feature.expand(batch_size, c.shape[1], task_feature.shape[2])
                # 256是encoder中'task mlp'输出的维数

                c = torch.cat((c, task_feature), dim=2)
                # c: (batch_size, num_points, feat_dim+task_feature_dim)
            else:
                c = 0
                for k in self.feature_keys:
                    if "grid" in k:
                        c += self.sample_grid_feature(p, c_plane[k])
                    elif "xy" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="xy")
                    elif "yz" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="yz")
                    elif "xz" in k:
                        c += self.sample_plane_feature(p, c_plane[k], plane="xz")
                c = c.transpose(1, 2)

        p = p.float()

        net = self.fc_p(p)
        # self.n_blocks: 5
        # net: (1, 2048, 64)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)  # net: (1, 2048, 64)
                # 5个mlp输出相加

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        # out: (batch_size, num_points, out_dim)
        # 这里的num_points指的是对应输入的sample点数量，
        # 比如decode_occ就是config中的num_point_occ
        # out_dim就是需要的feature维数，比如occupancy和segmentation就都是1
        # joint_params相关的就是8或者4
        return out, art_code

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

class AblationDecoder_New(nn.Module):
    def __init__(
        self,
        dim=3,
        c_dim=64,
        task_feat_dim=512,
        hidden_size=64,
        n_blocks=5,
        out_dim=12,
        leaky=False,
        sample_mode="bilinear",
        feature_keys=None,
        concat_feat=True,
        padding=0.1,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.feature_keys = feature_keys
        self.concat_feat = concat_feat

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim+task_feat_dim, hidden_size) for i in range(n_blocks)]
            )

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for i in range(n_blocks)]
        )

        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def forward(self, p, c, task_feature, **kwargs):
        # p: (batch_size, num_points, 3)
        # c: (batch_size, num_points, 64)
        # task_feature: (batch_size, 512)
        batch_size = p.shape[0]
        num_points = p.shape[1]

        task_feature = task_feature.unsqueeze(1)
        task_feature = task_feature.expand(batch_size, num_points, 512)
        c = torch.cat((c, task_feature), dim=2) # c: (B, N, 64+512)

        net = self.fc_p(p)
        # self.n_blocks: 5
        # net: (B, N, self.hidden_size)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c) # net: (B, N, self.hidden_size)
                # 5个mlp输出相加

            net = self.blocks[i](net)

        # net: (B, N, self.hidden_size)
        out = self.fc_out(self.actvn(net)) # out: (B, N, self.out_dim)

        # out: (batch_size, num_points, out_dim)
        # 这里的num_points指的是对应输入的sample点数量，
        # 比如decode_occ就是config中的num_point_occ
        # out_dim就是需要的feature维数，比如occupancy和segmentation就都是1
        # joint_params相关的就是8或者4
        return out


class AblationDecoder_Implicit(nn.Module):
    def __init__(
            self,
            dim=3,
            c_dim=64,
            task_feat_dim=512,
            hidden_size=64,
            n_blocks=5,
            out_dim=3,
            leaky=False,
            sample_mode="bilinear",
            feature_keys=None,
            concat_feat=True,
            padding=0.1,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.feature_keys = feature_keys
        self.concat_feat = concat_feat

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim + task_feat_dim, hidden_size) for i in range(n_blocks)]
            )

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(hidden_size) for i in range(n_blocks)]
        )

        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def forward(self, p, c, task_feature, **kwargs):
        # p: (batch_size, num_points, 3)
        # c: (batch_size, num_points, 64)
        # task_feature: (batch_size, 512)
        batch_size = p.shape[0]
        num_points = p.shape[1]
        art_code = task_feature[:, 0]

        task_feature = task_feature.unsqueeze(1)
        task_feature = task_feature.expand(batch_size, num_points, 512)
        c = torch.cat((c, task_feature), dim=2)  # c: (B, N, 64+512)

        net = self.fc_p(p)
        # self.n_blocks: 5
        # net: (B, N, self.hidden_size)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)  # net: (B, N, self.hidden_size)
                # 5个mlp输出相加

            net = self.blocks[i](net)

        # net: (B, N, self.hidden_size)
        out = self.fc_out(self.actvn(net))  # out: (B, N, self.out_dim)

        # out: (batch_size, num_points, out_dim)
        # 这里的num_points指的是对应输入的sample点数量，
        # 比如decode_occ就是config中的num_point_occ
        # out_dim就是需要的feature维数，比如occupancy和segmentation就都是1
        # joint_params相关的就是8或者4
        return out, art_code