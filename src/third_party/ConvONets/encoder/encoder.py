import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean

from src.third_party.ConvONets.common import (
    coordinate2index,
    normalize_3d_coordinate,
    normalize_coordinate,
)
from src.third_party.ConvONets.encoder.pointnetpp_attn import PointNetPlusPlusAttnFusion, \
    PointNetPlusPlusAttnFusion_New,  PointNetPlusPlusAttnFusion_3frame
from src.third_party.ConvONets.encoder.pointnetpp_corr import PointNetPlusPlusCorrFusion
from src.third_party.ConvONets.encoder.unet import UNet
from src.third_party.ConvONets.encoder.unet3d import UNet3D
from src.third_party.ConvONets.layers import ResnetBlockFC

from pdb import set_trace

# 原版
class LocalPoolPointnetPPFusion(nn.Module):
    """PointNet++Attn-based encoder network with ResNet blocks for each point.
        The network takes two inputs and fuse them with Attention layer
        Number of input points are fixed.
        Separate features for geometry and articulation

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    """

    def __init__(
        self,
        c_dim=128,
        dim=3,
        hidden_dim=128,
        task_hidden_dim1=64,
        task_hidden_dim2=128,
        task_output_dim=256,
        scatter_type="max",
        mlp_kwargs=None,
        attn_kwargs=None,
        unet=False,
        unet_kwargs=None,
        unet3d=False,
        unet3d_kwargs=None,
        unet_corr=False,
        unet_kwargs_corr=None,
        unet3d_corr=False,
        unet3d_kwargs_corr=None,
        corr_aggregation=None,
        plane_resolution=None,
        grid_resolution=None,
        plane_type="xz",
        padding=0.0,
        n_blocks=5,
        feat_pos="attn",
        return_score=False,
    ):
        super().__init__()
        self.linear1 = nn.Linear(1, task_hidden_dim1)
        self.linear2 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear3 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.linear4 = nn.Linear(1, task_hidden_dim1)
        self.linear5 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear6 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.linear7 = nn.Linear(1, task_hidden_dim1)
        self.linear8 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear9 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.c_dim = c_dim
        self.return_score = return_score
        if feat_pos == "attn":
            self.feat_pos = PointNetPlusPlusAttnFusion(
                c_dim=hidden_dim * 2, attn_kwargs=attn_kwargs
            )
        elif feat_pos == "corr":
            self.feat_pos = PointNetPlusPlusCorrFusion(
                c_dim=hidden_dim * 2,
                mlp_kwargs=mlp_kwargs,
                corr_aggregation=corr_aggregation,
            )
        else:
            raise NotImplementedError(f"Encoder {feat_pos} not implemented!")

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.blocks_corr = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c_corr = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        if unet_corr:
            self.unet_corr = UNet(c_dim, in_channels=c_dim, **unet_kwargs_corr)
        else:
            self.unet_corr = None

        if unet3d_corr:
            self.unet3d_corr = UNet3D(**unet3d_kwargs_corr)
        else:
            self.unet3d_corr = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == "max":
            self.scatter = scatter_max
        elif scatter_type == "mean":
            self.scatter = scatter_mean
        else:
            raise ValueError("incorrect scatter type")

    def generate_plane_features(self, p, c, plane="xz", unet=None):
        # acquire indices of features in plane
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(
            p.size(0), self.c_dim, self.reso_plane, self.reso_plane
        )  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if unet is not None:
            fea_plane = unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c, unet3d=None):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type="3d")
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)  # B x C x reso^3
        fea_grid = fea_grid.reshape(
            p.size(0),
            self.c_dim,
            self.reso_grid,
            self.reso_grid,
            self.reso_grid,
        )  # sparce matrix (B x 512 x reso x reso)

        if unet3d is not None:
            fea_grid = unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == "grid":
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_grid ** 3,
                )
            else:
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_plane ** 2,
                )
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p, p2, state_start, state_end, state_label):
        batch_size, T, D = p.size()

        state_start = state_start.unsqueeze(1)
        state_start = F.relu(self.linear1(state_start))
        state_start = F.relu(self.linear2(state_start))
        state_start = self.linear3(state_start)

        state_end = state_end.unsqueeze(1)
        state_end = F.relu(self.linear4(state_end))
        state_end = F.relu(self.linear5(state_end))
        state_end = self.linear6(state_end)

        state_label = state_label.unsqueeze(1)
        state_label = F.relu(self.linear7(state_label))
        state_label = F.relu(self.linear8(state_label))
        state_label = self.linear9(state_label)

        state_feat = torch.cat((state_start, state_end, state_label), dim=1)

        # acquire the index for each point
        coord = {}
        index = {}
        if "xz" in " ".join(self.plane_type):
            coord["xz"] = normalize_coordinate(
                p.clone(), plane="xz", padding=self.padding
            )
            index["xz"] = coordinate2index(coord["xz"], self.reso_plane)
        if "xy" in " ".join(self.plane_type):
            coord["xy"] = normalize_coordinate(
                p.clone(), plane="xy", padding=self.padding
            )
            index["xy"] = coordinate2index(coord["xy"], self.reso_plane)
        if "yz" in " ".join(self.plane_type):
            coord["yz"] = normalize_coordinate(
                p.clone(), plane="yz", padding=self.padding
            )
            index["yz"] = coordinate2index(coord["yz"], self.reso_plane)
        if "grid" in " ".join(self.plane_type):
            coord["grid"] = normalize_3d_coordinate(p.clone(), padding=self.padding)
            index["grid"] = coordinate2index(
                coord["grid"], self.reso_grid, coord_type="3d"
            )
        _, net, net_corr = self.feat_pos(p, p2, return_score=self.return_score)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)

        net_corr = self.blocks_corr[0](net_corr)
        for block_corr in self.blocks_corr[1:]:
            pooled = self.pool_local(coord, index, net_corr)
            net_corr = torch.cat([net_corr, pooled], dim=2)
            net_corr = block_corr(net_corr)
        c_corr = self.fc_c_corr(net_corr)

        fea = {}
        for f in self.plane_type:
            k1, k2 = f.split("_")
            if k2 in ["xy", "yz", "xz"]:
                if k1 == "geo":
                    fea[f] = self.generate_plane_features(
                        p, c, plane=k2, unet=self.unet
                    )
                elif k1 == "corr":
                    fea[f] = self.generate_plane_features(
                        p, c_corr, plane=k2, unet=self.unet_corr
                    )
            elif k2 == "grid":
                if k1 == "geo":
                    fea[f] = self.generate_grid_features(p, c, unet3d=self.unet3d)
                elif k1 == "corr":
                    fea[f] = self.generate_grid_features(
                        p, c_corr, unet3d=self.unet3d_corr
                    )
        return fea, state_feat

# take 4维输入 (最后一维是segmentation的encoder) , 也是目前我们的方法使用的
class LocalPoolPointnetPPFusion_4dims(nn.Module):
    """PointNet++Attn-based encoder network with ResNet blocks for each point.
        The network takes two inputs and fuse them with Attention layer
        Number of input points are fixed.
        Separate features for geometry and articulation

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    """

    def __init__(
        self,
        c_dim=64,
        dim=4,
        hidden_dim=128,
        task_hidden_dim1=64,
        task_hidden_dim2=128,
        task_output_dim=256,
        scatter_type="max",
        mlp_kwargs=None,
        attn_kwargs=None,
        unet=False,
        unet_kwargs=None,
        unet3d=False,
        unet3d_kwargs=None,
        unet_corr=False,
        unet_kwargs_corr=None,
        unet3d_corr=False,
        unet3d_kwargs_corr=None,
        corr_aggregation=None,
        plane_resolution=None,
        grid_resolution=None,
        plane_type="xz",
        padding=0.0,
        n_blocks=5,
        feat_pos="attn",
        return_score=False,
    ):
        super().__init__()
        self.linear1 = nn.Linear(1, task_hidden_dim1)
        self.linear2 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear3 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.linear4 = nn.Linear(1, task_hidden_dim1)
        self.linear5 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear6 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.c_dim = c_dim
        self.return_score = return_score
        if feat_pos == "attn":
            # 默认是这个
            self.feat_pos = PointNetPlusPlusAttnFusion_New(
                c_dim=hidden_dim * 2, attn_kwargs=attn_kwargs
            )
        else:
            raise NotImplementedError(f"Encoder {feat_pos} not implemented!")

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.blocks_corr = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c_corr = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        if unet_corr:
            self.unet_corr = UNet(c_dim, in_channels=c_dim, **unet_kwargs_corr)
        else:
            self.unet_corr = None

        if unet3d_corr:
            self.unet3d_corr = UNet3D(**unet3d_kwargs_corr)
        else:
            self.unet3d_corr = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == "max":
            self.scatter = scatter_max
        elif scatter_type == "mean":
            self.scatter = scatter_mean
        else:
            raise ValueError("incorrect scatter type")

    def generate_plane_features(self, p, c, plane="xz", unet=None):
        # acquire indices of features in plane
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(
            p.size(0), self.c_dim, self.reso_plane, self.reso_plane
        )  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if unet is not None:
            fea_plane = unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c, unet3d=None):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type="3d")
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)  # B x C x reso^3
        fea_grid = fea_grid.reshape(
            p.size(0),
            self.c_dim,
            self.reso_grid,
            self.reso_grid,
            self.reso_grid,
        )  # sparce matrix (B x 512 x reso x reso)

        if unet3d is not None:
            fea_grid = unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == "grid":
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_grid ** 3,
                )
            else:
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_plane ** 2,
                )
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p, state_start, state_end):
        # p.shape: (B, N, 4)
        batch_size, T, D = p.size()

        state_start = state_start.unsqueeze(1)
        state_start = F.relu(self.linear1(state_start))
        state_start = F.relu(self.linear2(state_start))
        state_start = self.linear3(state_start)

        state_end = state_end.unsqueeze(1)
        state_end = F.relu(self.linear4(state_end))
        state_end = F.relu(self.linear5(state_end))
        state_end = self.linear6(state_end)

        state_feat = torch.cat((state_start, state_end), dim=1)

        # acquire the index for each point
        coord = {}
        index = {}
        if "xz" in " ".join(self.plane_type):
            coord["xz"] = normalize_coordinate(
                p.clone(), plane="xz", padding=self.padding
            )
            index["xz"] = coordinate2index(coord["xz"], self.reso_plane)
        if "xy" in " ".join(self.plane_type):
            coord["xy"] = normalize_coordinate(
                p.clone(), plane="xy", padding=self.padding
            )
            index["xy"] = coordinate2index(coord["xy"], self.reso_plane)
        if "yz" in " ".join(self.plane_type):
            coord["yz"] = normalize_coordinate(
                p.clone(), plane="yz", padding=self.padding
            )
            index["yz"] = coordinate2index(coord["yz"], self.reso_plane)
        if "grid" in " ".join(self.plane_type):
            coord["grid"] = normalize_3d_coordinate(p.clone(), padding=self.padding)
            index["grid"] = coordinate2index(
                coord["grid"], self.reso_grid, coord_type="3d"
            )

        _, net, net_corr = self.feat_pos(p, return_score=self.return_score)
        # net: (B, N, 256); net_corr: (B, N, 256)

        # 前缀geo或corr表示用block还是block_corr
        # 后缀grid或xy表示采的feature是什么样子的
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)
        # c: (B, N, 64)

        net_corr = self.blocks_corr[0](net_corr)
        for block_corr in self.blocks_corr[1:]:
            pooled = self.pool_local(coord, index, net_corr)
            net_corr = torch.cat([net_corr, pooled], dim=2)
            net_corr = block_corr(net_corr)
        c_corr = self.fc_c_corr(net_corr)
        # c_corr: (B, N, 64), 其中64是c_dim

        fea = {}
        for f in self.plane_type:
            k1, k2 = f.split("_")
            if k2 in ["xy", "yz", "xz"]:
                if k1 == "geo":
                    fea[f] = self.generate_plane_features(
                        p, c, plane=k2, unet=self.unet
                    )
                elif k1 == "corr":
                    fea[f] = self.generate_plane_features(
                        p, c_corr, plane=k2, unet=self.unet_corr
                    )
            elif k2 == "grid":
                if k1 == "geo":
                    fea[f] = self.generate_grid_features(p, c, unet3d=self.unet3d)
                elif k1 == "corr":
                    fea[f] = self.generate_grid_features(
                        p, c_corr, unet3d=self.unet3d_corr
                    )

        # fea['geo_grid']: (batch_size, unet3d_kwargs.out_channels, self.reso_grid, self.reso_grid, self.reso_grid)
        # fea['geo_grid']: (1, 32, 8, 8, 8), fea['corr_xy']: (1, 64, 64, 64) ...
        return fea, state_feat

class LocalPoolPointnetPPFusion_4dims_3frame(nn.Module):
    """PointNet++Attn-based encoder network with ResNet blocks for each point.
        The network takes two inputs and fuse them with Attention layer
        Number of input points are fixed.
        Separate features for geometry and articulation

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    """

    def __init__(
        self,
        c_dim=64,
        dim=4,
        hidden_dim=128,
        task_hidden_dim1=64,
        task_hidden_dim2=128,
        task_output_dim=256,
        scatter_type="max",
        mlp_kwargs=None,
        attn_kwargs=None,
        unet=False,
        unet_kwargs=None,
        unet3d=False,
        unet3d_kwargs=None,
        unet_corr=False,
        unet_kwargs_corr=None,
        unet3d_corr=False,
        unet3d_kwargs_corr=None,
        corr_aggregation=None,
        plane_resolution=None,
        grid_resolution=None,
        plane_type="xz",
        padding=0.0,
        n_blocks=5,
        feat_pos="attn",
        return_score=False,
        ablation=False,
    ):
        super().__init__()
        self.linear1 = nn.Linear(1, task_hidden_dim1)
        self.linear2 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear3 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.linear4 = nn.Linear(1, task_hidden_dim1)
        self.linear5 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear6 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.linear7 = nn.Linear(1, task_hidden_dim1)
        self.linear8 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear9 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.ablation = ablation

        self.c_dim = c_dim
        self.return_score = return_score
        if feat_pos == "attn":
            # 默认是这个
            self.feat_pos = PointNetPlusPlusAttnFusion_3frame(
                c_dim=hidden_dim * 2, attn_kwargs=attn_kwargs
            )
        else:
            raise NotImplementedError(f"Encoder {feat_pos} not implemented!")

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.blocks_corr = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c_corr = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        # model相关问题，注释这四行
        # if unet:
        #     self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        # else:
        #     self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        # model相关问题，注释这四行
        # if unet_corr:
        #     self.unet_corr = UNet(c_dim, in_channels=c_dim, **unet_kwargs_corr)
        # else:
        #     self.unet_corr = None

        if unet3d_corr:
            self.unet3d_corr = UNet3D(**unet3d_kwargs_corr)
        else:
            self.unet3d_corr = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == "max":
            self.scatter = scatter_max
        elif scatter_type == "mean":
            self.scatter = scatter_mean
        else:
            raise ValueError("incorrect scatter type")

    def generate_plane_features(self, p, c, plane="xz", unet=None):
        # acquire indices of features in plane
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(
            p.size(0), self.c_dim, self.reso_plane, self.reso_plane
        )  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if unet is not None:
            fea_plane = unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c, unet3d=None):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type="3d")
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)  # B x C x reso^3
        fea_grid = fea_grid.reshape(
            p.size(0),
            self.c_dim,
            self.reso_grid,
            self.reso_grid,
            self.reso_grid,
        )  # sparce matrix (B x 512 x reso x reso)

        if unet3d is not None:
            fea_grid = unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == "grid":
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_grid ** 3,
                )
            else:
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_plane ** 2,
                )
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p_start, p_end, state_start, state_end, state_target):
        # p.shape: (B, N, 4)
        batch_size, T, D = p_start.size()

        state_start = state_start.unsqueeze(1)
        state_start = F.relu(self.linear1(state_start))
        state_start = F.relu(self.linear2(state_start))
        state_start = self.linear3(state_start)

        state_end = state_end.unsqueeze(1)
        state_end = F.relu(self.linear4(state_end))
        state_end = F.relu(self.linear5(state_end))
        state_end = self.linear6(state_end)

        state_target = state_target.unsqueeze(1)
        state_target = F.relu(self.linear7(state_target))
        state_target = F.relu(self.linear8(state_target))
        state_target = self.linear9(state_target)

        state_feat = torch.cat((state_start, state_end, state_target), dim=1)

        # acquire the index for each point
        coord = {}
        index = {}
        if "xz" in " ".join(self.plane_type):
            coord["xz"] = normalize_coordinate(
                p_start.clone(), plane="xz", padding=self.padding
            )
            index["xz"] = coordinate2index(coord["xz"], self.reso_plane)
        if "xy" in " ".join(self.plane_type):
            coord["xy"] = normalize_coordinate(
                p_start.clone(), plane="xy", padding=self.padding
            )
            index["xy"] = coordinate2index(coord["xy"], self.reso_plane)
        if "yz" in " ".join(self.plane_type):
            coord["yz"] = normalize_coordinate(
                p_start.clone(), plane="yz", padding=self.padding
            )
            index["yz"] = coordinate2index(coord["yz"], self.reso_plane)
        if "grid" in " ".join(self.plane_type):
            coord["grid"] = normalize_3d_coordinate(p_start.clone(), padding=self.padding)
            index["grid"] = coordinate2index(
                coord["grid"], self.reso_grid, coord_type="3d"
            )

        _, net, net_corr = self.feat_pos(p_start, p_end, return_score=self.return_score)
        # net: (B, N, 256); net_corr: (B, N, 256)

        # 前缀geo或corr表示用block还是block_corr
        # 后缀grid或xy表示采的feature是什么样子的
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)
        # c: (B, N, 64)

        if self.ablation is True:
            # 如果是ablation实验就直接返回
            return c, state_feat

        # net_corr = self.blocks_corr[0](net_corr)
        # for block_corr in self.blocks_corr[1:]:
        #     pooled = self.pool_local(coord, index, net_corr)
        #     net_corr = torch.cat([net_corr, pooled], dim=2)
        #     net_corr = block_corr(net_corr)
        # c_corr = self.fc_c_corr(net_corr)
        # c_corr: (B, N, 64), 其中64是c_dim

        fea = {}
        for f in self.plane_type:
            k1, k2 = f.split("_")
            if k2 in ["xy", "yz", "xz"]:
                if k1 == "geo":
                    fea[f] = self.generate_plane_features(
                        p_start, c, plane=k2, unet=self.unet
                    )
                elif k1 == "corr":
                    fea[f] = self.generate_plane_features(
                        p_start, c_corr, plane=k2, unet=self.unet_corr
                    )
            elif k2 == "grid":
                if k1 == "geo":
                    fea[f] = self.generate_grid_features(p_start, c, unet3d=self.unet3d)
                elif k1 == "corr":
                    fea[f] = self.generate_grid_features(
                        p_start, c_corr, unet3d=self.unet3d_corr
                    )

        # fea['geo_grid']: (batch_size, unet3d_kwargs.out_channels, self.reso_grid, self.reso_grid, self.reso_grid)
        return fea, state_feat

class LocalPoolPointnetPPFusion_4dims_3frame_interpolation(nn.Module):
    """PointNet++Attn-based encoder network with ResNet blocks for each point.
        The network takes two inputs and fuse them with Attention layer
        Number of input points are fixed.
        Separate features for geometry and articulation

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    """

    def __init__(
        self,
        c_dim=64,
        dim=4,
        hidden_dim=128,
        task_hidden_dim1=64,
        task_hidden_dim2=128,
        task_output_dim=256,
        scatter_type="max",
        mlp_kwargs=None,
        attn_kwargs=None,
        unet=False,
        unet_kwargs=None,
        unet3d=False,
        unet3d_kwargs=None,
        unet_corr=False,
        unet_kwargs_corr=None,
        unet3d_corr=False,
        unet3d_kwargs_corr=None,
        corr_aggregation=None,
        plane_resolution=None,
        grid_resolution=None,
        plane_type="xz",
        padding=0.0,
        n_blocks=5,
        feat_pos="attn",
        return_score=False,
        ablation=False,
    ):
        super().__init__()
        self.linear1 = nn.Linear(1, task_hidden_dim1)
        self.linear2 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear3 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.linear4 = nn.Linear(1, task_hidden_dim1)
        self.linear5 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear6 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.linear7 = nn.Linear(1, task_hidden_dim1)
        self.linear8 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear9 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.ablation = ablation

        self.c_dim = c_dim
        self.return_score = return_score
        if feat_pos == "attn":
            # 默认是这个
            self.feat_pos = PointNetPlusPlusAttnFusion_3frame(
                c_dim=hidden_dim * 2, attn_kwargs=attn_kwargs
            )
        else:
            raise NotImplementedError(f"Encoder {feat_pos} not implemented!")

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.blocks_corr = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c_corr = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        # model相关问题，注释这四行
        # if unet:
        #     self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        # else:
        #     self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        # model相关问题，注释这四行
        # if unet_corr:
        #     self.unet_corr = UNet(c_dim, in_channels=c_dim, **unet_kwargs_corr)
        # else:
        #     self.unet_corr = None

        if unet3d_corr:
            self.unet3d_corr = UNet3D(**unet3d_kwargs_corr)
        else:
            self.unet3d_corr = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == "max":
            self.scatter = scatter_max
        elif scatter_type == "mean":
            self.scatter = scatter_mean
        else:
            raise ValueError("incorrect scatter type")

    def generate_plane_features(self, p, c, plane="xz", unet=None):
        # acquire indices of features in plane
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(
            p.size(0), self.c_dim, self.reso_plane, self.reso_plane
        )  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if unet is not None:
            fea_plane = unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c, unet3d=None):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type="3d")
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)  # B x C x reso^3
        fea_grid = fea_grid.reshape(
            p.size(0),
            self.c_dim,
            self.reso_grid,
            self.reso_grid,
            self.reso_grid,
        )  # sparce matrix (B x 512 x reso x reso)

        if unet3d is not None:
            fea_grid = unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == "grid":
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_grid ** 3,
                )
            else:
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_plane ** 2,
                )
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p_start, p_end, state_start, state_end, state_target):
        # p.shape: (B, N, 4)
        batch_size, T, D = p_start.size()

        interpolation_state = (state_start+state_end) / 2

        state_start = state_start.unsqueeze(1)
        state_start = F.relu(self.linear1(state_start))
        state_start = F.relu(self.linear2(state_start))
        state_start = self.linear3(state_start)

        state_end = state_end.unsqueeze(1)
        state_end = F.relu(self.linear4(state_end))
        state_end = F.relu(self.linear5(state_end))
        state_end = self.linear6(state_end)

        state_target = state_target.unsqueeze(1)
        state_target = F.relu(self.linear7(state_target))
        state_target = F.relu(self.linear8(state_target))
        state_target = self.linear9(state_target)

        # 将target的feature改变为start到target之间的平均值
        state_target = (state_start+state_target) / 2

        state_feat = torch.cat((state_start, state_end, state_target), dim=1)

        # acquire the index for each point
        coord = {}
        index = {}
        if "xz" in " ".join(self.plane_type):
            coord["xz"] = normalize_coordinate(
                p_start.clone(), plane="xz", padding=self.padding
            )
            index["xz"] = coordinate2index(coord["xz"], self.reso_plane)
        if "xy" in " ".join(self.plane_type):
            coord["xy"] = normalize_coordinate(
                p_start.clone(), plane="xy", padding=self.padding
            )
            index["xy"] = coordinate2index(coord["xy"], self.reso_plane)
        if "yz" in " ".join(self.plane_type):
            coord["yz"] = normalize_coordinate(
                p_start.clone(), plane="yz", padding=self.padding
            )
            index["yz"] = coordinate2index(coord["yz"], self.reso_plane)
        if "grid" in " ".join(self.plane_type):
            coord["grid"] = normalize_3d_coordinate(p_start.clone(), padding=self.padding)
            index["grid"] = coordinate2index(
                coord["grid"], self.reso_grid, coord_type="3d"
            )

        _, net, net_corr = self.feat_pos(p_start, p_end, return_score=self.return_score)
        # net: (B, N, 256); net_corr: (B, N, 256)

        # 前缀geo或corr表示用block还是block_corr
        # 后缀grid或xy表示采的feature是什么样子的
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)
        # c: (B, N, 64)

        if self.ablation is True:
            # 如果是ablation实验就直接返回
            return c, state_feat

        # net_corr = self.blocks_corr[0](net_corr)
        # for block_corr in self.blocks_corr[1:]:
        #     pooled = self.pool_local(coord, index, net_corr)
        #     net_corr = torch.cat([net_corr, pooled], dim=2)
        #     net_corr = block_corr(net_corr)
        # c_corr = self.fc_c_corr(net_corr)
        # c_corr: (B, N, 64), 其中64是c_dim

        fea = {}
        for f in self.plane_type:
            k1, k2 = f.split("_")
            if k2 in ["xy", "yz", "xz"]:
                if k1 == "geo":
                    fea[f] = self.generate_plane_features(
                        p_start, c, plane=k2, unet=self.unet
                    )
                elif k1 == "corr":
                    fea[f] = self.generate_plane_features(
                        p_start, c_corr, plane=k2, unet=self.unet_corr
                    )
            elif k2 == "grid":
                if k1 == "geo":
                    fea[f] = self.generate_grid_features(p_start, c, unet3d=self.unet3d)
                elif k1 == "corr":
                    fea[f] = self.generate_grid_features(
                        p_start, c_corr, unet3d=self.unet3d_corr
                    )

        # fea['geo_grid']: (batch_size, unet3d_kwargs.out_channels, self.reso_grid, self.reso_grid, self.reso_grid)
        return fea, state_feat

class LocalPoolPointnetPPFusion_4dims_3frame_extrapolation(nn.Module):
    """PointNet++Attn-based encoder network with ResNet blocks for each point.
        The network takes two inputs and fuse them with Attention layer
        Number of input points are fixed.
        Separate features for geometry and articulation

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    """

    def __init__(
        self,
        c_dim=64,
        dim=4,
        hidden_dim=128,
        task_hidden_dim1=64,
        task_hidden_dim2=128,
        task_output_dim=256,
        scatter_type="max",
        mlp_kwargs=None,
        attn_kwargs=None,
        unet=False,
        unet_kwargs=None,
        unet3d=False,
        unet3d_kwargs=None,
        unet_corr=False,
        unet_kwargs_corr=None,
        unet3d_corr=False,
        unet3d_kwargs_corr=None,
        corr_aggregation=None,
        plane_resolution=None,
        grid_resolution=None,
        plane_type="xz",
        padding=0.0,
        n_blocks=5,
        feat_pos="attn",
        return_score=False,
        ablation=False,
    ):
        super().__init__()
        self.linear1 = nn.Linear(1, task_hidden_dim1)
        self.linear2 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear3 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.linear4 = nn.Linear(1, task_hidden_dim1)
        self.linear5 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear6 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.linear7 = nn.Linear(1, task_hidden_dim1)
        self.linear8 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear9 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.ablation = ablation

        self.c_dim = c_dim
        self.return_score = return_score
        if feat_pos == "attn":
            # 默认是这个
            self.feat_pos = PointNetPlusPlusAttnFusion_3frame(
                c_dim=hidden_dim * 2, attn_kwargs=attn_kwargs
            )
        else:
            raise NotImplementedError(f"Encoder {feat_pos} not implemented!")

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.blocks_corr = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c_corr = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        # model相关问题，注释这四行
        # if unet:
        #     self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        # else:
        #     self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        # model相关问题，注释这四行
        # if unet_corr:
        #     self.unet_corr = UNet(c_dim, in_channels=c_dim, **unet_kwargs_corr)
        # else:
        #     self.unet_corr = None

        if unet3d_corr:
            self.unet3d_corr = UNet3D(**unet3d_kwargs_corr)
        else:
            self.unet3d_corr = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == "max":
            self.scatter = scatter_max
        elif scatter_type == "mean":
            self.scatter = scatter_mean
        else:
            raise ValueError("incorrect scatter type")

    def generate_plane_features(self, p, c, plane="xz", unet=None):
        # acquire indices of features in plane
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(
            p.size(0), self.c_dim, self.reso_plane, self.reso_plane
        )  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if unet is not None:
            fea_plane = unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c, unet3d=None):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type="3d")
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)  # B x C x reso^3
        fea_grid = fea_grid.reshape(
            p.size(0),
            self.c_dim,
            self.reso_grid,
            self.reso_grid,
            self.reso_grid,
        )  # sparce matrix (B x 512 x reso x reso)

        if unet3d is not None:
            fea_grid = unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == "grid":
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_grid ** 3,
                )
            else:
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_plane ** 2,
                )
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p_start, p_end, state_start, state_end, state_target):
        # p.shape: (B, N, 4)
        batch_size, T, D = p_start.size()

        interpolation_state = (state_start+state_end) / 2

        state_start = state_start.unsqueeze(1)
        state_start = F.relu(self.linear1(state_start))
        state_start = F.relu(self.linear2(state_start))
        state_start = self.linear3(state_start)

        state_end = state_end.unsqueeze(1)
        state_end = F.relu(self.linear4(state_end))
        state_end = F.relu(self.linear5(state_end))
        state_end = self.linear6(state_end)

        state_target = state_target.unsqueeze(1)
        state_target = F.relu(self.linear7(state_target))
        state_target = F.relu(self.linear8(state_target))
        state_target = self.linear9(state_target)

        # 将target的feature改变为start到target之间的平均值
        state_target = state_target + (state_target-state_start) * (3/7)

        state_feat = torch.cat((state_start, state_end, state_target), dim=1)

        # acquire the index for each point
        coord = {}
        index = {}
        if "xz" in " ".join(self.plane_type):
            coord["xz"] = normalize_coordinate(
                p_start.clone(), plane="xz", padding=self.padding
            )
            index["xz"] = coordinate2index(coord["xz"], self.reso_plane)
        if "xy" in " ".join(self.plane_type):
            coord["xy"] = normalize_coordinate(
                p_start.clone(), plane="xy", padding=self.padding
            )
            index["xy"] = coordinate2index(coord["xy"], self.reso_plane)
        if "yz" in " ".join(self.plane_type):
            coord["yz"] = normalize_coordinate(
                p_start.clone(), plane="yz", padding=self.padding
            )
            index["yz"] = coordinate2index(coord["yz"], self.reso_plane)
        if "grid" in " ".join(self.plane_type):
            coord["grid"] = normalize_3d_coordinate(p_start.clone(), padding=self.padding)
            index["grid"] = coordinate2index(
                coord["grid"], self.reso_grid, coord_type="3d"
            )

        _, net, net_corr = self.feat_pos(p_start, p_end, return_score=self.return_score)
        # net: (B, N, 256); net_corr: (B, N, 256)

        # 前缀geo或corr表示用block还是block_corr
        # 后缀grid或xy表示采的feature是什么样子的
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)
        # c: (B, N, 64)

        if self.ablation is True:
            # 如果是ablation实验就直接返回
            return c, state_feat

        # net_corr = self.blocks_corr[0](net_corr)
        # for block_corr in self.blocks_corr[1:]:
        #     pooled = self.pool_local(coord, index, net_corr)
        #     net_corr = torch.cat([net_corr, pooled], dim=2)
        #     net_corr = block_corr(net_corr)
        # c_corr = self.fc_c_corr(net_corr)
        # c_corr: (B, N, 64), 其中64是c_dim

        fea = {}
        for f in self.plane_type:
            k1, k2 = f.split("_")
            if k2 in ["xy", "yz", "xz"]:
                if k1 == "geo":
                    fea[f] = self.generate_plane_features(
                        p_start, c, plane=k2, unet=self.unet
                    )
                elif k1 == "corr":
                    fea[f] = self.generate_plane_features(
                        p_start, c_corr, plane=k2, unet=self.unet_corr
                    )
            elif k2 == "grid":
                if k1 == "geo":
                    fea[f] = self.generate_grid_features(p_start, c, unet3d=self.unet3d)
                elif k1 == "corr":
                    fea[f] = self.generate_grid_features(
                        p_start, c_corr, unet3d=self.unet3d_corr
                    )

        # fea['geo_grid']: (batch_size, unet3d_kwargs.out_channels, self.reso_grid, self.reso_grid, self.reso_grid)
        return fea, state_feat

# baseline 使用的, 直接生成不用trans_mat
class LocalPoolPointnetPPFusion_4dims_GlobalCode(nn.Module):
    """PointNet++Attn-based encoder network with ResNet blocks for each point.
        The network takes two inputs and fuse them with Attention layer
        Number of input points are fixed.
        Separate features for geometry and articulation

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    """

    def __init__(
        self,
        c_dim=64,
        dim=4,
        hidden_dim=128,
        task_hidden_dim1=64,
        task_hidden_dim2=128,
        task_output_dim=256,
        scatter_type="max",
        mlp_kwargs=None,
        attn_kwargs=None,
        unet=False,
        unet_kwargs=None,
        unet3d=False,
        unet3d_kwargs=None,
        unet_corr=False,
        unet_kwargs_corr=None,
        unet3d_corr=False,
        unet3d_kwargs_corr=None,
        corr_aggregation=None,
        plane_resolution=None,
        grid_resolution=None,
        plane_type="xz",
        padding=0.0,
        n_blocks=5,
        feat_pos="attn",
        return_score=False,
    ):
        super().__init__()
        self.linear1 = nn.Linear(1, task_hidden_dim1)
        self.linear2 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear3 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.linear4 = nn.Linear(1, task_hidden_dim1)
        self.linear5 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear6 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.c_dim = c_dim
        self.return_score = return_score
        if feat_pos == "attn":
            # 默认是这个
            self.feat_pos = PointNetPlusPlusAttnFusion_New(
                c_dim=hidden_dim * 2, attn_kwargs=attn_kwargs
            )
        else:
            raise NotImplementedError(f"Encoder {feat_pos} not implemented!")

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.blocks_corr = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c_corr = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        if unet_corr:
            self.unet_corr = UNet(c_dim, in_channels=c_dim, **unet_kwargs_corr)
        else:
            self.unet_corr = None

        if unet3d_corr:
            self.unet3d_corr = UNet3D(**unet3d_kwargs_corr)
        else:
            self.unet3d_corr = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == "max":
            self.scatter = scatter_max
        elif scatter_type == "mean":
            self.scatter = scatter_mean
        else:
            raise ValueError("incorrect scatter type")

    def generate_plane_features(self, p, c, plane="xz", unet=None):
        # acquire indices of features in plane
        xy = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(
            p.size(0), self.c_dim, self.reso_plane, self.reso_plane
        )  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if unet is not None:
            fea_plane = unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c, unet3d=None):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type="3d")
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)  # B x C x reso^3
        fea_grid = fea_grid.reshape(
            p.size(0),
            self.c_dim,
            self.reso_grid,
            self.reso_grid,
            self.reso_grid,
        )  # sparce matrix (B x 512 x reso x reso)

        if unet3d is not None:
            fea_grid = unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == "grid":
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_grid ** 3,
                )
            else:
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_plane ** 2,
                )
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p, state_start, state_end):
        # p.shape: (B, N, 4)
        batch_size, T, D = p.size()

        state_start = state_start.unsqueeze(1)
        state_start = F.relu(self.linear1(state_start))
        state_start = F.relu(self.linear2(state_start))
        state_start = self.linear3(state_start)

        state_end = state_end.unsqueeze(1)
        state_end = F.relu(self.linear4(state_end))
        state_end = F.relu(self.linear5(state_end))
        state_end = self.linear6(state_end)

        state_feat = torch.cat((state_start, state_end), dim=1)

        # acquire the index for each point
        coord = {}
        index = {}
        if "xz" in " ".join(self.plane_type):
            coord["xz"] = normalize_coordinate(
                p.clone(), plane="xz", padding=self.padding
            )
            index["xz"] = coordinate2index(coord["xz"], self.reso_plane)
        if "xy" in " ".join(self.plane_type):
            coord["xy"] = normalize_coordinate(
                p.clone(), plane="xy", padding=self.padding
            )
            index["xy"] = coordinate2index(coord["xy"], self.reso_plane)
        if "yz" in " ".join(self.plane_type):
            coord["yz"] = normalize_coordinate(
                p.clone(), plane="yz", padding=self.padding
            )
            index["yz"] = coordinate2index(coord["yz"], self.reso_plane)
        if "grid" in " ".join(self.plane_type):
            coord["grid"] = normalize_3d_coordinate(p.clone(), padding=self.padding)
            index["grid"] = coordinate2index(
                coord["grid"], self.reso_grid, coord_type="3d"
            )

        _, net, net_corr = self.feat_pos(p, return_score=self.return_score)
        # net: (B, N, 256); net_corr: (B, N, 256)

        # 前缀geo或corr表示用block还是block_corr
        # 后缀grid或xy表示采的feature是什么样子的
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)
        # c: (B, N, 64)

        net_corr = self.blocks_corr[0](net_corr)
        for block_corr in self.blocks_corr[1:]:
            pooled = self.pool_local(coord, index, net_corr)
            net_corr = torch.cat([net_corr, pooled], dim=2)
            net_corr = block_corr(net_corr)
        c_corr = self.fc_c_corr(net_corr)
        # c_corr: (B, N, 64), 其中64是c_dim

        fea = {}
        for f in self.plane_type:
            k1, k2 = f.split("_")
            if k2 in ["xy", "yz", "xz"]:
                if k1 == "geo":
                    fea[f] = self.generate_plane_features(
                        p, c, plane=k2, unet=self.unet
                    )
                elif k1 == "corr":
                    fea[f] = self.generate_plane_features(
                        p, c_corr, plane=k2, unet=self.unet_corr
                    )
            elif k2 == "grid":
                if k1 == "geo":
                    fea[f] = self.generate_grid_features(p, c, unet3d=self.unet3d)
                elif k1 == "corr":
                    fea[f] = self.generate_grid_features(
                        p, c_corr, unet3d=self.unet3d_corr
                    )

        # fea['geo_grid']: (batch_size, unet3d_kwargs.out_channels, self.reso_grid, self.reso_grid, self.reso_grid)
        # fea['geo_grid']: (1, 32, 8, 8, 8), fea['corr_xy']: (1, 64, 64, 64) ...
        return fea, state_feat

# ablation study 使用的, 连grid feature也不用了
class OnlyPointnet_4dims_GlobalCode(nn.Module):
    """PointNet++Attn-based encoder network with ResNet blocks for each point.
        The network takes two inputs and fuse them with Attention layer
        Number of input points are fixed.
        Separate features for geometry and articulation

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    """

    def __init__(
            self,
            c_dim=64,
            dim=4,
            hidden_dim=128,
            task_hidden_dim1=64,
            task_hidden_dim2=128,
            task_output_dim=256,
            scatter_type="max",
            mlp_kwargs=None,
            attn_kwargs=None,
            unet=False,
            unet_kwargs=None,
            unet3d=False,
            unet3d_kwargs=None,
            unet_corr=False,
            unet_kwargs_corr=None,
            unet3d_corr=False,
            unet3d_kwargs_corr=None,
            corr_aggregation=None,
            plane_resolution=None,
            grid_resolution=None,
            plane_type="xz",
            padding=0.0,
            n_blocks=5,
            feat_pos="attn",
            return_score=False,
    ):
        super().__init__()
        self.linear1 = nn.Linear(1, task_hidden_dim1)
        self.linear2 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear3 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.linear4 = nn.Linear(1, task_hidden_dim1)
        self.linear5 = nn.Linear(task_hidden_dim1, task_hidden_dim2)
        self.linear6 = nn.Linear(task_hidden_dim2, task_output_dim)

        self.c_dim = c_dim
        self.return_score = return_score
        if feat_pos == "attn":
            # 默认是这个
            self.feat_pos = PointNetPlusPlusAttnFusion_New(
                c_dim=hidden_dim * 2, attn_kwargs=attn_kwargs
            )
        else:
            raise NotImplementedError(f"Encoder {feat_pos} not implemented!")

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.blocks_corr = nn.ModuleList(
            [ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)]
        )
        self.fc_c_corr = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        if unet_corr:
            self.unet_corr = UNet(c_dim, in_channels=c_dim, **unet_kwargs_corr)
        else:
            self.unet_corr = None

        if unet3d_corr:
            self.unet3d_corr = UNet3D(**unet3d_kwargs_corr)
        else:
            self.unet3d_corr = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == "max":
            self.scatter = scatter_max
        elif scatter_type == "mean":
            self.scatter = scatter_mean
        else:
            raise ValueError("incorrect scatter type")

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == "grid":
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_grid ** 3,
                )
            else:
                fea = self.scatter(
                    c.permute(0, 2, 1),
                    index[key],
                    dim_size=self.reso_plane ** 2,
                )
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p, state_start, state_end):
        batch_size, T, D = p.size()

        state_start = state_start.unsqueeze(1)
        state_start = F.relu(self.linear1(state_start))
        state_start = F.relu(self.linear2(state_start))
        state_start = self.linear3(state_start)
        # (B, 256 which is self.task_out_dim)

        state_end = state_end.unsqueeze(1)
        state_end = F.relu(self.linear4(state_end))
        state_end = F.relu(self.linear5(state_end))
        state_end = self.linear6(state_end)
        # (B, 256 which is self.task_out_dim)

        state_feat = torch.cat((state_start, state_end), dim=1)

        # acquire the index for each point
        coord = {}
        index = {}
        if "xz" in " ".join(self.plane_type):
            coord["xz"] = normalize_coordinate(
                p.clone(), plane="xz", padding=self.padding
            )
            index["xz"] = coordinate2index(coord["xz"], self.reso_plane)
        if "xy" in " ".join(self.plane_type):
            coord["xy"] = normalize_coordinate(
                p.clone(), plane="xy", padding=self.padding
            )
            index["xy"] = coordinate2index(coord["xy"], self.reso_plane)
        if "yz" in " ".join(self.plane_type):
            coord["yz"] = normalize_coordinate(
                p.clone(), plane="yz", padding=self.padding
            )
            index["yz"] = coordinate2index(coord["yz"], self.reso_plane)
        if "grid" in " ".join(self.plane_type):
            coord["grid"] = normalize_3d_coordinate(p.clone(), padding=self.padding)
            index["grid"] = coordinate2index(
                coord["grid"], self.reso_grid, coord_type="3d"
            )
        _, net, net_corr = self.feat_pos(p, return_score=self.return_score)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)

        net_corr = self.blocks_corr[0](net_corr)
        for block_corr in self.blocks_corr[1:]:
            pooled = self.pool_local(coord, index, net_corr)
            net_corr = torch.cat([net_corr, pooled], dim=2)
            net_corr = block_corr(net_corr)
        c_corr = self.fc_c_corr(net_corr)

        # c & c_corr: (B, N, self.c_dim (which is 64 in current case) )
        # state_feat: (B, 256)
        # 不生成grid_feature

        return c, state_feat