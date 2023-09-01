#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pdb import set_trace

import chamfer3D.dist_chamfer_3D
from emd import earth_mover_distance
# from chamfer_distance import ChamferDistance

def pc_visualization(dir, pc):
    # pc: (num_points, 3), np_ndarray
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='b', s=2, alpha=0.3)

    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

    plt.savefig(dir)

def pc_visualization_comparision(dir, pc1, pc2):
    # pc: (num_points, 3), np_ndarray
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='b', s=2, alpha=0.3)
    ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='r', s=2, alpha=0.3)

    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

    plt.savefig(dir)

def pc_visualization_comparision_with_arrow(dir, pc1, pc2, arrow1=None, arrow2=None, arrow3=None):
    # pc: (num_points, 3), np_ndarray
    fig = plt.figure()
    ax = Axes3D(fig)
    sc1 = ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='b', s=1, alpha=0.3, zorder=0)
    sc2 = ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='green', s=1, alpha=0.3, zorder=1)

    mid_x = (pc1[:, 0].max() + pc1[:, 0].min()) / 2
    mid_y = (pc1[:, 1].max() + pc1[:, 1].min()) / 2
    mid_z = (pc1[:, 2].max() + pc1[:, 2].min()) / 2

    arr1 = ax.quiver(mid_x, mid_y, mid_z, 0+arrow1[0], 0+arrow1[1], arrow1[2], color='orange', zorder=2)
    arr2 = ax.quiver(mid_x, mid_y, mid_z, 0+arrow2[0]*2, 0+arrow2[1]*2, arrow2[2]*2, color='r', zorder=3)
    arr3 = ax.quiver(0, 0, 0, 0 + arrow3[0], 0 + arrow3[1], arrow3[2], color='cyan', zorder=4)

    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

    plt.savefig(dir)

def compute_pc_chamfer_with_grad(
    gtpc, pred, offset, scale, num_mesh_samples=8192, use_emd=False
):
    if use_emd is True:
        d = earth_mover_distance(gtpc, pred, transpose=False)
        loss = torch.mean(d)
    else:
        chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

        dist1, dist2, idx1, idx2 = chamLoss(gtpc, pred)
        loss = torch.mean(dist1) + torch.mean(dist2)

    return loss

def compute_pc_chamfer(
    gtpc, pred, offset, scale, num_mesh_samples=8192, verbose=False
):
    """
        calculate chamfer distance using gt point cloud
    """
    if gtpc.shape[0] == 0 or pred.shape[0] == 0:
        return np.nan

    pred_points = pred
    gt_points = gtpc

    pred_points = pred_points.reshape(pred_points.shape[0]*pred_points.shape[1], pred_points.shape[2])
    gt_points = gt_points.reshape(gt_points.shape[0] * gt_points.shape[1], gt_points.shape[2])

    pred_points = pred_points.detach().numpy()
    gt_points = gt_points.detach().numpy()

    # pc_visualization('/home/wuruihai/Ditto-master/visual_for_debug/pred_points.jpg', pred_points)

    # print(f'gt_points.shape: {gt_points.shape}') # torch.Size([8192, 3])

    gt_points = (gt_points - offset) / scale

    # one direction
    pred_points_kd_tree = KDTree(pred_points)
    one_distances, one_vertex_ids = pred_points_kd_tree.query(gt_points)
    gt_to_pred_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_points)
    pred_to_gt_chamfer = np.mean(np.square(two_distances))

    if verbose:
        print(
            gt_to_pred_chamfer + pred_to_gt_chamfer,
            gt_to_pred_chamfer,
            pred_to_gt_chamfer,
        )
    return gt_to_pred_chamfer + pred_to_gt_chamfer


def compute_trimesh_chamfer_using_gtpc(
    gtpc, pred_mesh, offset, scale, num_mesh_samples=8192, verbose=False
):
    """
        calculate chamfer distance using gt point cloud
    """
    if gtpc.shape[0] == 0 or pred_mesh.vertices.shape[0] == 0:
        return np.nan

    pred_points = trimesh.sample.sample_surface(pred_mesh, num_mesh_samples)[0]
    gt_points = gtpc

    pc_visualization('/home/wuruihai/Ditto-master/visual_for_debug/pred_points.jpg', pred_points)

    # print(f'gt_points.shape: {gt_points.shape}') # torch.Size([8192, 3])

    gt_points = (gt_points - offset) / scale

    # one direction
    pred_points_kd_tree = KDTree(pred_points)
    one_distances, one_vertex_ids = pred_points_kd_tree.query(gt_points)
    gt_to_pred_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_points)
    pred_to_gt_chamfer = np.mean(np.square(two_distances))

    if verbose:
        print(
            gt_to_pred_chamfer + pred_to_gt_chamfer,
            gt_to_pred_chamfer,
            pred_to_gt_chamfer,
        )
    return gt_to_pred_chamfer + pred_to_gt_chamfer



def compute_trimesh_chamfer(
    gt_mesh, pred_mesh, offset, scale, num_mesh_samples=30000, verbose=False
):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_mesh: trimesh.base.Trimesh of ground truth mesh

    pred_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """
    if gt_mesh.vertices.shape[0] == 0 or pred_mesh.vertices.shape[0] == 0:
        return np.nan

    pred_points = trimesh.sample.sample_surface(pred_mesh, num_mesh_samples)[0]
    gt_points = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples)[0]

    pc_visualization('/home/wuruihai/Ditto-master/visual_for_debug/pred_points.jpg', pred_points)

    print(f'gt_points.shape: {gt_points.shape}')

    gt_points = (gt_points - offset) / scale

    # one direction
    pred_points_kd_tree = KDTree(pred_points)
    one_distances, one_vertex_ids = pred_points_kd_tree.query(gt_points)
    gt_to_pred_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_points)
    pred_to_gt_chamfer = np.mean(np.square(two_distances))

    if verbose:
        print(
            gt_to_pred_chamfer + pred_to_gt_chamfer,
            gt_to_pred_chamfer,
            pred_to_gt_chamfer,
        )
    return gt_to_pred_chamfer + pred_to_gt_chamfer
