import glob
import os
import random

import numpy as np
import torch
from numpy.lib.arraysetops import isin
from omegaconf import ListConfig
from torch.utils.data import Dataset

from src.utils.misc import occ_to_binary_label, sample_occ_points, sample_point_cloud
from src.utils.transform import Rotation

from ipdb import set_trace

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# different occ points and seg points
# represent articulation as dense joints
# include transformed surface points
class GeoArtDatasetV0(Dataset):
    def __init__(self, opt):
        if isinstance(opt["data_path"], ListConfig):
            # multi class
            self.path_list = []
            for data_path in opt["data_path"]:
                self.path_list.extend(
                    glob.glob(
                        os.path.join(opt["data_dir"], data_path, "scenes", "*.npz")
                    )
                )
        else:
            self.path_list = glob.glob(
                os.path.join(opt["data_dir"], opt["data_path"], "scenes", "*.npz")
            )

        if opt.get("num_data"):
            random.shuffle(self.path_list)
            self.path_list = self.path_list[: opt["num_data"]]

        self.num_point = opt["num_point"]
        self.num_point_occ = opt["num_point_occ"]
        self.num_point_seg = opt["num_point_seg"]
        self.norm = opt.get("norm", False)
        self.rand_rot = opt.get("rand_rot", False)
        self.rand_scale = min(opt.get("rand_scale", 0), np.pi * 2)
        self.rand_scale = max(self.rand_scale, 0)
        self.weighted_occ_sample = opt.get("weighted_occ_sample", False)
        if self.norm:
            self.norm_padding = opt.get("norm_padding", 0.1)

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

    def __getitem__(self, index):
        data = np.load(self.path_list[index])
        # data['pc_start']: (12438, 3), np.ndarray
        # pc_start: (8192, 3), np.ndarray
        print(f'Number of points: {self.num_point}')

        pc_start, pc_start_idx = sample_point_cloud(data["pc_start"], self.num_point)

        pc_end, pc_end_idx = sample_point_cloud(data["pc_end"], self.num_point)

        pc_start_end = data["pc_start_end"][pc_start_idx]

        pc_end_start = data["pc_end_start"][pc_end_idx]

        # start: 可视化
        self.pc_visualization(pc_start, '/home/wuruihai/Ditto-master/visual_for_debug/pc_start.jpg')
        self.pc_visualization(pc_end, '/home/wuruihai/Ditto-master/visual_for_debug/pc_end.jpg')
        self.pc_visualization_comparision(pc_start, pc_end,
                                          '/home/wuruihai/Ditto-master/visual_for_debug/pc_start_and_end.jpg')
        self.pc_visualization(pc_start_end, '/home/wuruihai/Ditto-master/visual_for_debug/pc_start_end.jpg')
        self.pc_visualization(pc_end_start, '/home/wuruihai/Ditto-master/visual_for_debug/pc_end_start.jpg')
        self.pc_visualization_comparision(pc_start_end, pc_start,
                                          '/home/wuruihai/Ditto-master/visual_for_debug/pc_start_end_and_start.jpg')
        # end

        pc_seg_label_start = data["pc_seg_start"][pc_start_idx]
        pc_seg_label_end = data["pc_seg_end"][pc_end_idx]
        state_start = data["state_start"]
        state_end = data["state_end"]
        screw_axis = data["screw_axis"]
        screw_moment = data["screw_moment"]

        # print(f'screw_axis first: {screw_axis}')
        # print(f'self.rand_rot: {self.rand_rot}')

        self.pc_visualization_comparision_with_arrow(pc_start, pc_start, screw_axis, screw_axis,
                                                     '/home/wuruihai/Ditto-master/visual_for_debug/pc_start_and_end_with_quiver.jpg')

        joint_type = data["joint_type"]
        joint_index = data["joint_index"]
        # shape2motion's 0 joint start from base object
        if "Shape2Motion" in self.path_list[index]:
            occ_label, seg_label = occ_to_binary_label(
                data["start_occ_list"], joint_index + 1
            )
        else:
            if "syn" in self.path_list[index]:
                joint_index = 1
            occ_label, seg_label = occ_to_binary_label(
                data["start_occ_list"], joint_index
            )

        # 这里得到的两个label都是针对100000个点的

        # print(f'data["start_occ_list"][0][0:10]: '
        #       f'{data["start_occ_list"][0][0:10]}')

        # process occ and seg points

        occ_label = occ_label.astype(np.bool)
        p_occ_start_postive = data["start_p_occ"][occ_label] # 取出了occ为true的sample point
        seg_label = seg_label[occ_label] # seg_label只取出来了原list中occ为true的部分
        p_seg_start, seg_idx_start = sample_point_cloud(
            p_occ_start_postive, self.num_point_seg
        ) # 即根据预设好的要取的num_point_seg即seg中的点数从p_occ_start_positive中sample出特定数量的点
        # p_seg_start表示sample出来的点
        # seg_idx_start表示sample的点在p_occ_start_positive中的下标
        seg_label = seg_label[seg_idx_start] # 然后根据sample点的下标相应地更改seg_label

        # 下面的occ同理
        if self.weighted_occ_sample: # Laptop_Overfit setting下是False
            start_occ_density = data["start_occ_density"]
            occ_idx_start = sample_occ_points(
                occ_label - 0.5, start_occ_density, self.num_point_occ
            )
            occ_label = occ_label[occ_idx_start]
            p_occ_start = data["start_p_occ"][occ_idx_start]
        else:
            p_occ_start, occ_idx_start = sample_point_cloud(
                data["start_p_occ"], self.num_point_occ
            )
            occ_label = occ_label[occ_idx_start]

        screw_point = np.cross(screw_axis, screw_moment) # (3,)

        self.pc_visualization_comparision(pc_start, screw_point.reshape(1, 3),
                                          '/home/wuruihai/Ditto-master/visual_for_debug/pc_start_and_end_with_screw.jpg')

        # random rotation, 这里为false
        if self.rand_rot:
            ax, ay, az = (torch.rand(3).numpy() - 0.5) * self.rand_scale
            rand_rot_mat = Rotation.from_euler("xyz", (ax, ay, az)).as_matrix()
            pc_start = rand_rot_mat.dot(pc_start.T).T
            pc_end = rand_rot_mat.dot(pc_end.T).T
            pc_start_end = rand_rot_mat.dot(pc_start_end.T).T
            pc_end_start = rand_rot_mat.dot(pc_end_start.T).T
            p_occ_start = rand_rot_mat.dot(p_occ_start.T).T
            p_seg_start = rand_rot_mat.dot(p_seg_start.T).T
            screw_axis = rand_rot_mat.dot(screw_axis)
            screw_point = rand_rot_mat.dot(screw_point)

        # normalize
        # scale和center用这种方法生成就可以了！
        bound_max = np.maximum(pc_start.max(0), pc_end.max(0))
        bound_min = np.minimum(pc_start.min(0), pc_end.min(0))
        center = (bound_min + bound_max) / 2
        scale = (bound_max - bound_min).max()
        scale = scale * (1 + self.norm_padding)

        print(f'self.norm_padding: {self.norm_padding}')

        pc_start = (pc_start - center) / scale
        pc_end = (pc_end - center) / scale
        p_occ_start = (p_occ_start - center) / scale
        p_seg_start = (p_seg_start - center) / scale
        pc_start_end = (pc_start_end - center) / scale
        pc_end_start = (pc_end_start - center) / scale

        # ((p - c) / s) X l
        # = (p X l) / s - (c X l) / s
        # = m / s - (c X l) / s
        screw_point = (screw_point - center) / scale

        screw_moment = np.cross(screw_point, screw_axis)

        # screw_moment = screw_moment / scale - np.cross(center, screw_axis) / scale
        if joint_type == 1:
            # prismatic joint, state change with scale
            state_start /= scale
            state_end /= scale

        # only change revolute joint
        # only change z-axis joints
        if screw_axis[2] <= -0.9 and joint_type == 0:
            screw_axis = -screw_axis
            screw_moment = -screw_moment
            state_start, state_end = state_end, state_start

        screw_point = np.cross(screw_axis, screw_moment)

        # print(screw_point.shape)
        # print(screw_point)
        self.pc_visualization_comparision(pc_start, screw_point.reshape(1, 3),
                                          '/home/wuruihai/Ditto-master/visual_for_debug/pc_start_and_end_with_screw2.jpg')

        # 生成这二者的方式, 注意也是直接通过已知信息计算即可得到
        p2l_vec, p2l_dist = batch_perpendicular_line(
            p_seg_start, screw_axis, screw_point
        )

        # print(joint_index)
        #
        # print(f'screw_axis second: {screw_axis}')

        self.pc_visualization(p_occ_start, '/home/wuruihai/Ditto-master/visual_for_debug/grid.jpg')

        self.pc_visualization(p_seg_start, '/home/wuruihai/Ditto-master/visual_for_debug/plane.jpg')

        self.pc_visualization_comparision(p_occ_start, p_seg_start, '/home/wuruihai/Ditto-master/visual_for_debug/sample.jpg')

        # print(f'pc_start:{pc_start.shape}')
        # print(f'pc_start_end:{pc_start_end.shape}')
        # print(f'pc_seg_label_start:{pc_seg_label_start.shape}')
        # print(f'pc_end:{pc_end.shape}')
        # print(f'pc_end_start:{pc_end_start.shape}')
        # print(f'pc_seg_label_end:{pc_seg_label_end.shape}')
        # print(f'state_start:{state_start.shape}')
        # print(f'state_end:{state_end.shape}')
        # print(f'screw_axis:{screw_axis.shape}')
        # print(f'screw_moment:{screw_moment.shape}')
        # print(f'occ_label:{occ_label.shape}')
        # print(f'seg_label:{seg_label.shape}')
        # print(f'p_occ_start: {p_occ_start.shape}')
        # print(f'p_seg_start: {p_seg_start.shape}')
        # print(f'scale: {scale.shape}')
        # print(f'center: {center.shape}')
        # print(f'p2l_vec: {p2l_vec.shape}')
        # print(f'p2l_dist: {p2l_dist.shape}')
        # print(f'data_path: {type(self.path_list[index])}')

        return_dict = {
            "pc_start": pc_start,  # N, 3
            "pc_end": pc_end,
            "pc_start_end": pc_start_end,
            "pc_end_start": pc_end_start,
            "pc_seg_label_start": pc_seg_label_start,
            "pc_seg_label_end": pc_seg_label_end,
            "state_start": state_start,
            "state_end": state_end,
            "screw_axis": screw_axis,
            "screw_moment": screw_moment,
            "p2l_vec": p2l_vec,
            "p2l_dist": p2l_dist,
            "joint_type": joint_type,
            "joint_index": np.array(joint_index),
            "p_occ": p_occ_start,
            "occ_label": occ_label, # 布尔型
            "p_seg": p_seg_start,
            "seg_label": seg_label, # int型
            "scale": scale,
            "center": center,
            "data_path": self.path_list[index],
        }

        # 将dict中的np.array转换为torch.tensor
        for k, v in return_dict.items():
            if isinstance(v, np.ndarray):
                return_dict[k] = torch.from_numpy(v).float()

        return return_dict

    def __len__(self):
        return len(self.path_list)


def batch_perpendicular_line(
    x: np.ndarray, l: np.ndarray, pivot: np.ndarray
) -> np.ndarray:
    """
    x: B * 3
    l: 3
    pivot: 3
    p_l: B * 3
    """
    offset = x - pivot
    p_l = offset.dot(l)[:, np.newaxis] * l[np.newaxis] - offset
    dist = np.sqrt(np.sum(p_l ** 2, axis=-1))
    p_l = p_l / (dist[:, np.newaxis] + 1.0e-5)
    return p_l, dist
