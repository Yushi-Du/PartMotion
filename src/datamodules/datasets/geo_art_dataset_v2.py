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
class GeoArtDatasetV2(Dataset):
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
        # print(self.path_list[index])
        # assert np.load(self.path_list[index])
        rd = np.load(self.path_list[index])
        return_dict = {}
        for key in rd.files:
            return_dict[key] = rd[key]

        return_dict['data_path'] = str(return_dict['data_path'])

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
