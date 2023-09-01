"""
From the implementation of https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import torch
import torch.nn as nn

from src.models.modules.Transformer import Attn, TransformerAttn
from src.third_party.ConvONets.encoder.pointnetpp_utils import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
)
from pdb import set_trace


# Concat attn with original feature
# return score and abstract point index
# different decoder for occ and articulation
class PointNetPlusPlusAttnFusion(nn.Module):
    def __init__(self, dim=None, c_dim=128, padding=0.1, attn_kwargs=None):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=6,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )

        self.fp2 = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256, mlp=[256, 128, c_dim])

        self.fp2_corr = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        self.fp1_corr = PointNetFeaturePropagation(
            in_channel=256, mlp=[256, 128, c_dim]
        )
        attn_type = attn_kwargs.get("type", "Transformer")
        if attn_type == "simple":
            self.attn = Attn(attn_kwargs)
        elif attn_type == "Transformer":
            self.attn = TransformerAttn(attn_kwargs)

    def encode_deep_feature(self, xyz, return_xyz=False):
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points, l1_fps_idx = self.sa1(l0_xyz, l0_points, returnfps=True)
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points, returnfps=True)
        fps_idx = torch.gather(l1_fps_idx, 1, l2_fps_idx)
        if return_xyz:
            return l2_points, l2_xyz, fps_idx
        else:
            return l2_points

    def forward(self, xyz, xyz2, return_score=False):
        """
        xyz: B*N*3
        xyz2: B*N*3
        -------
        return:
        B*N'*3
        B*N'*C
        B*N'
        B*N'
        B*N'*N'
        """
        xyz = xyz.permute(0, 2, 1)
        l2_points_xyz2, l2_xyz2, fps_idx2 = self.encode_deep_feature(
            xyz2, return_xyz=True
        )

        l0_points = xyz # (B, 3, N)
        l0_xyz = xyz[:, :3, :] # (B, 3, N)

        l1_xyz, l1_points, l1_fps_idx = self.sa1(l0_xyz, l0_points, returnfps=True)
        # 选出其中有代表性的点，并根据xyz得到feature
        # (B, 3, sa1.npoint), (B, sa1.mlp[2], sa1.npoint), (B, sa1.npoint)
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points, returnfps=True)
        # l2_fps_idx中的值描述的是l1_xyz中的下标
        # (B, 3, sa2.npoint), (B, sa2.mlp[2], sa2.npoint), (B, sa2.npoint)
        fps_idx = torch.gather(l1_fps_idx, 1, l2_fps_idx)
        # fps_idx: 暂时没用, 所以l1_fps_idx和l2_fps_idx也都暂时没用
        attn, score = self.attn(l2_points, l2_points_xyz2, True) # l2_points_xyz2: (B, sa2.mlp[2], sa2.npoint)
        # return.score=False, 所以score没用
        l2_points = torch.cat((l2_points, attn), dim=1)

        # (B, 3, 512), (B, 3, 128), (B, 128, 512), (B, 512, 128)
        l1_points_back = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points_back)

        l1_points_corr = self.fp2_corr(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points_corr = self.fp1_corr(l0_xyz, l1_xyz, None, l1_points_corr)
        # return_score: 默认False
        if return_score:
            return (
                xyz.permute(0, 2, 1),
                l0_points.permute(0, 2, 1),
                l0_points_corr.permute(0, 2, 1),
                fps_idx,
                fps_idx2,
                score,
            )
        else:
            return (
                xyz.permute(0, 2, 1),
                l0_points.permute(0, 2, 1),
                l0_points_corr.permute(0, 2, 1),
            )

class PointNetPlusPlusAttnFusion_3frame(nn.Module):
    def __init__(self, dim=None, c_dim=128, padding=0.1, attn_kwargs=None):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=3+4,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )

        self.fp2 = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256, mlp=[256, 128, c_dim])

        self.fp2_corr = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        self.fp1_corr = PointNetFeaturePropagation(
            in_channel=256, mlp=[256, 128, c_dim]
        )
        attn_type = attn_kwargs.get("type", "Transformer")
        if attn_type == "simple":
            self.attn = Attn(attn_kwargs)
        elif attn_type == "Transformer":
            self.attn = TransformerAttn(attn_kwargs)

    def encode_deep_feature(self, xyz, return_xyz=False):
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points, l1_fps_idx = self.sa1(l0_xyz, l0_points, returnfps=True)
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points, returnfps=True)
        fps_idx = torch.gather(l1_fps_idx, 1, l2_fps_idx)
        if return_xyz:
            return l2_points, l2_xyz, fps_idx
        else:
            return l2_points

    def forward(self, xyz, xyz2, return_score=False):
        """
        xyz: B*N*3
        xyz2: B*N*3
        -------
        return:
        B*N'*3
        B*N'*C
        B*N'
        B*N'
        B*N'*N'
        """
        xyz = xyz.permute(0, 2, 1)
        l2_points_xyz2, l2_xyz2, fps_idx2 = self.encode_deep_feature(
            xyz2, return_xyz=True
        )

        l0_points = xyz # (B, 3, N)
        l0_xyz = xyz[:, :3, :] # (B, 3, N)

        l1_xyz, l1_points, l1_fps_idx = self.sa1(l0_xyz, l0_points, returnfps=True)
        # 选出其中有代表性的点，并根据xyz得到feature
        # (B, 3, sa1.npoint), (B, sa1.mlp[2], sa1.npoint), (B, sa1.npoint)
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points, returnfps=True)
        # l2_fps_idx中的值描述的是l1_xyz中的下标
        # (B, 3, sa2.npoint), (B, sa2.mlp[2], sa2.npoint), (B, sa2.npoint)
        fps_idx = torch.gather(l1_fps_idx, 1, l2_fps_idx)
        # fps_idx: 暂时没用, 所以l1_fps_idx和l2_fps_idx也都暂时没用
        attn, score = self.attn(l2_points, l2_points_xyz2, True) # l2_points_xyz2: (B, sa2.mlp[2], sa2.npoint)
        # return.score=False, 所以score没用
        l2_points = torch.cat((l2_points, attn), dim=1)

        # (B, 3, 512), (B, 3, 128), (B, 128, 512), (B, 512, 128)
        l1_points_back = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points_back)

        l1_points_corr = self.fp2_corr(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points_corr = self.fp1_corr(l0_xyz, l1_xyz, None, l1_points_corr)
        # return_score: 默认False
        if return_score:
            return (
                xyz.permute(0, 2, 1),
                l0_points.permute(0, 2, 1),
                l0_points_corr.permute(0, 2, 1),
                fps_idx,
                fps_idx2,
                score,
            )
        else:
            return (
                xyz.permute(0, 2, 1),
                l0_points.permute(0, 2, 1),
                l0_points_corr.permute(0, 2, 1),
            )

class PointNetPlusPlusAttnFusion_New(nn.Module):
    def __init__(self, dim=None, c_dim=128, padding=0.1, attn_kwargs=None):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=3 + 4,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )

        self.fp2 = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256, mlp=[256, 128, c_dim])

        self.fp2_corr = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        self.fp1_corr = PointNetFeaturePropagation(
            in_channel=256, mlp=[256, 128, c_dim]
        )

    def forward(self, xyz, return_score=False):
        """
        xyz: B*N*4
        -------
        return:
        B*N'*3
        B*N'*C
        B*N'
        B*N'
        B*N'*N'
        """
        xyz = xyz.permute(0, 2, 1) # (B, 4, N)

        l0_points = xyz # (B, 4, N)
        l0_xyz = xyz[:, :3, :] # (B, 3, N)

        l1_xyz, l1_points, l1_fps_idx = self.sa1(l0_xyz, l0_points, returnfps=True)
        # (B, 3, sa1.npoint), (B, sa1.mlp[2], sa1.npoint), (B, sa1.npoint)
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points, returnfps=True)
        # (B, 3, sa2.npoint), (B, sa2.mlp[2], sa2.npoint), (B, sa2.npoint)
        fps_idx = torch.gather(l1_fps_idx, 1, l2_fps_idx)

        l2_points = torch.cat((l2_points, l2_points), dim=1) # (B, 2*sa2.mlp[2], sa2.npoint)

        l1_points_back = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points_back)

        l1_points_corr = self.fp2_corr(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points_corr = self.fp1_corr(l0_xyz, l1_xyz, None, l1_points_corr)

        return (
            xyz.permute(0, 2, 1),
            l0_points.permute(0, 2, 1),
            l0_points_corr.permute(0, 2, 1),
        )

