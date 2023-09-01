import json
import os
from copy import deepcopy
from pprint import pprint

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import trimesh
from torch import nn, optim
from torchmetrics import AverageMeter, Precision, Recall

from src.models.modules import create_network
from src.models.modules.losses_dense_joint import PrismaticLoss, RevoluteLoss, ChamferLoss
from src.third_party.ConvONets.conv_onet.generation_two_stage import Generator3D
from src.utils import utils
from src.utils.chamfer import compute_trimesh_chamfer, compute_trimesh_chamfer_using_gtpc, \
    compute_pc_chamfer, compute_pc_chamfer_with_grad
from src.utils.joint_estimation import (
    aggregate_dense_prediction_r,
    eval_joint_p,
    eval_joint_r,
)
from src.utils.misc import get_gt_mesh_from_data
from src.utils.visual import as_mesh

from pdb import set_trace

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import open3d as o3d

log = utils.get_logger(__name__)
# different head for occupancy and segmentation
# predict dense joint
class GeoArtModelOurs_3frames(pl.LightningModule):
    def __init__(self, opt, network):
        super().__init__()
        self.opt = opt
        for k, v in opt.hparams.items():
            self.hparams[k] = v
        self.save_hyperparameters(self.hparams)
        self.model = create_network(network)
        self.cri_chamfer = compute_pc_chamfer_with_grad

        self.cri_part_chamfer = compute_pc_chamfer_with_grad

    def pc_visualization(self, pc, dir):
        # pc: (num_points, 3), np_ndarray
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='b', s=5)

        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

        plt.savefig(dir)

    def pc_visualization_comparision(self, pc1, pc2, dir):
        # pc: (num_points, 3), np_ndarray
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='b', s=5)
        ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='r', s=5)

        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

        plt.savefig(dir)

    def pc_visualization_comparision_with_arrow(self, pc1, pc2, arrow1, arrow2, dir):
        # pc: (num_points, 3), np_ndarray
        fig = plt.figure()
        ax = Axes3D(fig)
        sc1 = ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='b', s=5, alpha=0.5, zorder=0)
        sc2 = ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='r', s=5, alpha=0.5, zorder=1)

        arr1 = ax.quiver(0.175, 0.175, 0, 0.175+arrow1[0], 0.175+arrow1[1], arrow1[2], length=0.2, zorder=2)
        arr2 = ax.quiver(0.175, 0.175, 0, 0.175+arrow2[0], 0.175+arrow2[1], arrow2[2], length=0.2, zorder=3)

        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

        plt.savefig(dir)

    def save_array_as_ply(self, xyz1, xyz2, dir, color):
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(xyz1)
        pcd1.paint_uniform_color(color[0])

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(xyz2)
        pcd2.paint_uniform_color(color[1])

        pcd = pcd1 + pcd2
        o3d.io.write_point_cloud(dir, pcd)

    def save_pc_as_txt(self, xyz, dir, color):
        with open(dir, 'a') as f:
            for i in range(xyz.shape[0]):
                content = f'{xyz[i][0]} {xyz[i][1]} {xyz[i][2]} {color[0]} {color[1]} {color[2]}\n'
                f.write(content)

    def forward(self, *args):
        # self.model: from
        # /home/wuruihai/Ditto-master/src/third_party/ConvONets/conv_onet/models/__init__.py
        # type：ConvolutionalOccupancyNetworkGeoArt
        return self.model(*args)

    def bgs(self, d6s):
        # d6s(bsz, 3, 2)
        bsz = d6s.shape[0]
        b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)  # 单位化
        # b1: (bsz, 3)
        b1 = b1.unsqueeze(1)
        a2 = d6s[:, :, 1]
        # a2: (bsz, 3)
        a2 = a2.unsqueeze(2)
        tmp = torch.bmm(b1, a2)
        tmp = tmp.squeeze(2)
        tmp = tmp * b1.squeeze(1)
        b1 = b1.squeeze(1)
        a2 = a2.squeeze(2)
        b2 = F.normalize(a2 - tmp, p=2, dim=1)
        # b1.view(bsz, 1, -1): (bsz, 1, 3); a2.view(bsz, -1, 1): (bsz, 3, 1);
        # torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1): (bsz, 1)
        # torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1: 广播乘法
        # 这里b1是经过单位化的，所以b1和a2点乘结果就是a2在b1方向投影的模
        b3 = torch.cross(b1, b2, dim=1)
        # 最终结果是三个单位向量相乘
        return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

    def transform(self, query_points, trans_mat, residual=False, six_dim=False):
        # query_points: (batch_size, num_points, 3)
        if six_dim is True:
            # trans_mat: (batch_size, num_points, 9)
            batch_size = query_points.shape[0]
            num_points = query_points.shape[1]

            rotation_mat = trans_mat[:, :, 0:6]  # (batch_size, num_points, 6)
            rotation_mat = rotation_mat.reshape(batch_size * num_points, 3, 2)
            rotation_mat = self.bgs(rotation_mat) # (batch_size*num_points, 3, 3)
            rotation_mat = rotation_mat.reshape(batch_size, num_points, 3, 3)

            # if residual is True:
            #     identity = torch.eye(3).to(rotation_mat.device)
            #     identity = identity.unsqueeze(0)
            #     identity = identity.expand(num_points, 3, 3)
            #     identity = identity.unsqueeze(0)
            #     identity = identity.expand(batch_size, num_points, 3, 3)
            #     rotation_mat = rotation_mat + identity

            rotation_mat = rotation_mat.reshape(batch_size * num_points, 3, 3)
            query_points = query_points.reshape(batch_size * num_points, 3)
            query_points = query_points.unsqueeze(2)  # (batch_size*num_points, 3, 1)

            res = torch.bmm(rotation_mat, query_points)  # (batch_size*num_points, 3, 1)
            res = res.squeeze(2)  # (batch_size*num_points, 3)
            res = res.reshape(batch_size, num_points, 3)

            translation = trans_mat[:, :, 6:9]  # # (batch_size, num_points, 3)
            if residual is True:
                zero = torch.zeros(3).to(translation.device)
                zero = zero.unsqueeze(0)
                zero = zero.expand(num_points, 3)
                zero = zero.unsqueeze(0)
                zero = zero.expand(batch_size, num_points, 3)
                translation = translation + zero
            res = res + translation
        else:
            # trans_mat: (batch_size, num_points, 12)
            batch_size = query_points.shape[0]
            num_points = query_points.shape[1]

            rotation_mat = trans_mat[:, :, 0:9] # (batch_size, num_points, 9)
            rotation_mat = rotation_mat.reshape(batch_size, num_points, 3, 3)
            # if residual is True:
            #     identity = torch.eye(3).to(rotation_mat.device)
            #     identity = identity.unsqueeze(0)
            #     identity = identity.expand(num_points, 3, 3)
            #     identity = identity.unsqueeze(0)
            #     identity = identity.expand(batch_size, num_points, 3, 3)
            #     rotation_mat += identity

            rotation_mat = rotation_mat.reshape(batch_size*num_points, 3, 3)
            query_points = query_points.reshape(batch_size*num_points, 3)
            query_points = query_points.unsqueeze(2) # (batch_size*num_points, 3, 1)

            res = torch.bmm(rotation_mat, query_points) # (batch_size*num_points, 3, 1)
            res = res.squeeze(2) # (batch_size*num_points, 3)
            res = res.reshape(batch_size, num_points, 3)

            translation = trans_mat[:, :, 9:12] # # (batch_size, num_points, 3)
            if residual is True:
                zero = torch.zeros(3).to(translation.device)
                zero = zero.unsqueeze(0)
                zero = zero.expand(num_points, 3)
                zero = zero.unsqueeze(0)
                zero = zero.expand(batch_size, num_points, 3)
                translation += zero
            res = res + translation

        return res

    def select_part(self, pc, seg):
        # pc: (B, N, 3)
        # seg: (B, N)

        batch_size = pc.shape[0]
        num_points = pc.shape[1]

        res = pc[seg.bool()].reshape(batch_size, int(num_points/2), 3).to(pc.device)

        return res

    def training_step(self, data, batch_idx):
        # self.model.encoder: <class 'src.third_party.ConvONets.encoder.encoder.LocalPoolPointnetPPFusion'>
        # data['pc_start']: (batch_size, total_num_points=8192, 3)
        # data['pc_end']: (batch_size, total_num_points=8192, 3)
        # data['pc_seg_label_start']: (batch_size, total_num_points=8192)
        # data['state_start']: (batch_size)
        seg_start = data['pc_seg_label_start'].unsqueeze(2)
        seg_end = data['pc_seg_label_end'].unsqueeze(2)
        input_p_start = torch.cat((data['pc_start'], seg_start), dim=2)
        input_p_end = torch.cat((data['pc_end'], seg_end), dim=2)

        # segmentation_start/end: (batch_size, num_points, 1)
        # input_p_start/end: (batch_size, num_points, 4), xyz和segmentation_mask concat到一起

        (
            trans_mat
        ) = self(input_p_start, input_p_end, data['state_start'], data['state_end'], data['state_target'])

        transformed_points = self.transform(data['pc_start'], trans_mat,
                                            residual=self.hparams['residual'], six_dim=self.hparams['six_dim'])
        loss_generation_pc = self.cri_chamfer(transformed_points, data['pc_target'], 0, 1,
                                              use_emd=self.hparams['use_emd'])
        part_point = self.select_part(transformed_points, data['pc_seg_label_start'])
        part_gt_point = self.select_part(data['pc_target'], data['pc_seg_label_target'])
        loss_part = self.cri_part_chamfer(part_point, part_gt_point,
                                          0, 1, use_emd=self.hparams['use_emd'])

        loss = (self.hparams.loss_weight_chamfer * loss_generation_pc + loss_part)

        # loss = loss_occ
        self.log("train/loss_chamfer", loss_generation_pc)
        self.log("train/loss_part", loss_part)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, data, batch_idx):
        seg_start = data['pc_seg_label_start'].unsqueeze(2)
        seg_end = data['pc_seg_label_end'].unsqueeze(2)
        input_p_start = torch.cat((data['pc_start'], seg_start), dim=2)
        input_p_end = torch.cat((data['pc_end'], seg_end), dim=2)
        (
            trans_mat
        ) = self(input_p_start, input_p_end, data['state_start'], data['state_end'], data['state_target'])

        transformed_points = self.transform(data['pc_start'], trans_mat,
                                            residual=self.hparams['residual'], six_dim=self.hparams['six_dim'])
        loss_generation_pc = self.cri_chamfer(transformed_points, data['pc_target'], 0, 1,
                                              use_emd=self.hparams['use_emd'])
        part_point = self.select_part(transformed_points, data['pc_seg_label_start'])
        part_gt_point = self.select_part(data['pc_target'], data['pc_seg_label_target'])
        loss_part = self.cri_part_chamfer(part_point, part_gt_point,
                                          0, 1, use_emd=self.hparams['use_emd'])

        loss = (self.hparams.loss_weight_chamfer * loss_generation_pc + loss_part)
        # loss = loss_occ
        self.log("val/loss_chamfer", loss_generation_pc)
        self.log("val/loss_part", loss_part)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def log_meter(self, meter, name):
        val = meter.compute()
        meter.reset()
        self.log(f"val/{name}", val)

    def validation_epoch_end(self, val_step_outputs):
        return

    def test_step(self, data, batch_idx):
        # 无论什么情况batch_size都是1

        # data['data_path'][0]: './xx/xx/10040'
        file_name = data['data_path'][0].split('/')[3]+'_'+str(batch_idx)
        cat_name = data['data_path'][0].split('/')[2]

        save_dir = f"/home/duyushi/Ditto-master/results/{cat_name}_Original_latest1"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        def normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
            return tensor / ((tensor ** 2).sum(dim, keepdim=True).sqrt() + 1.0e-5)

        # only support batch size 1
        assert data["pc_end"].size(0) == 1

        seg_start = data['pc_seg_label_start'].unsqueeze(2)
        seg_end = data['pc_seg_label_end'].unsqueeze(2)
        input_p_start = torch.cat((data['pc_start'], seg_start), dim=2)
        input_p_end = torch.cat((data['pc_end'], seg_end), dim=2)
        # (
        #     trans_mat, grids, grid_mat
        # ) = self(input_p_start, input_p_end, data['state_start'], data['state_end'], data['state_target'])
        (
            trans_mat
        ) = self(input_p_start, input_p_end, data['state_start'], data['state_end'], data['state_target'])

        transformed_points = self.transform(data['pc_start'], trans_mat,
                                            residual=self.hparams['residual'], six_dim=self.hparams['six_dim'])
        loss_generation_pc = self.cri_chamfer(transformed_points, data['pc_target'], 0, 1,
                                              use_emd=self.hparams['use_emd'])

        part_point = self.select_part(transformed_points, data['pc_seg_label_start']) # (1, 4096, 3)
        base_point = self.select_part(transformed_points, (data['pc_seg_label_start']+1) % 2)
        part_gt_point = self.select_part(data['pc_target'], data['pc_seg_label_target'])
        base_gt_point = self.select_part(data['pc_target'], (data['pc_seg_label_start']+1) % 2)
        loss_part = self.cri_part_chamfer(part_point, part_gt_point,
                                          0, 1, use_emd=self.hparams['use_emd'])

        cd = (self.hparams.loss_weight_chamfer * loss_generation_pc + loss_part)*1000

        # new_grids = self.transform(grids, grid_mat, residual=self.hparams['residual'], six_dim=self.hparams['six_dim'])
        #
        # self.save_pc_as_txt(new_grids.cpu()[0], os.path.join(save_dir, file_name + '_grids.txt'), color=[1, 0, 0])
        # self.save_array_as_ply(new_grids.cpu()[0], new_grids.cpu()[0],
        #                        os.path.join(save_dir, file_name + '_grids.ply'), color=[[0, 0, 1], [1, 0, 0]])

        self.save_pc_as_txt(part_point.cpu()[0], os.path.join(save_dir, file_name+'.txt'),
                            color=[0, 136, 204])
        self.save_pc_as_txt(base_point.cpu()[0], os.path.join(save_dir, file_name+'.txt'),
                            color=[105, 105, 105])

        self.save_pc_as_txt(part_gt_point.cpu()[0], os.path.join(save_dir, file_name + 'gt.txt'),
                            color=[0, 136, 204])
        self.save_pc_as_txt(base_gt_point.cpu()[0], os.path.join(save_dir, file_name + 'gt.txt'),
                            color=[105, 105, 105])

        self.save_array_as_ply(data['pc_start'].cpu()[0], transformed_points.cpu()[0],
                                          os.path.join(save_dir, file_name+'.ply'), color=[[0, 0, 1], [1, 0, 0]])

        self.save_array_as_ply(data['pc_target'].cpu()[0], transformed_points.cpu()[0],
                               os.path.join(save_dir, file_name+'with_gt.ply'), color=[[0, 1, 0], [1, 0, 0]])

        result = {
            "geo": {
                "cd": cd,
            },
        }
        return result

    def test_epoch_end(self, outputs) -> None:
        # outputs = self.all_gather(outputs)
        results_all = {
            "geo": {
                "cd": [],
            },
        }
        for result in outputs:
            for k, v in result["geo"].items():
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                results_all["geo"][k].append(v)

        results_mean = deepcopy(results_all)
        for k, v in results_all["geo"].items():
            tmp = np.array(v).reshape(-1)
            tmp = np.mean([x for x in tmp if not np.isnan(x)])
            results_mean["geo"][k] = float(tmp)

        if self.trainer.is_global_zero:
            pprint(results_mean)
            utils.save_results(results_mean)
            log.info(f"Saved results to {os.getcwd()}")
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, self.hparams.lr_decay_gamma
        )
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.hparams.lr_decay_freq,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}
