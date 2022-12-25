import os, sys
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from nuscenes.nuscenes import NuScenes

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Calculate mean of length, width and height for NuScenes dataset #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
nuscenes_anno_dir = "/home/extra/rickyyuan/dataset/nuscenes/v1.0-trainval/train/annos"
nuscenes_lidar_dir = "/home/extra/rickyyuan/dataset/nuscenes/v1.0-trainval/train/lidar"

anno_files = os.listdir(nuscenes_anno_dir)
# lidar_files = os.listdir(nuscenes_lidar_dir)

nuscenes_labels = {1: 'Vehicle', 2: 'Pedestrian', 4: 'Cyclist'}
nuscenes_boxes = {1: np.empty((0,3), np.float32), 2: np.empty((0,3), np.float32), 4: np.empty((0,3), np.float32)}
nuscenes_lwh_mean = {1: np.zeros((1,3), np.float32), 2: np.zeros((1,3), np.float32), 4: np.zeros((1,3), np.float32)}

for anno_file in tqdm(anno_files):
    with open(os.path.join(nuscenes_anno_dir, anno_file), 'rb') as pk:
        anno = pickle.load(pk)
        for anno_obj in anno['objects']:
            label = anno_obj['label']
            box = anno_obj['box']
            if label in nuscenes_labels:
                nuscenes_boxes[label] = np.vstack((nuscenes_boxes[label], box[3:6]))

for label in nuscenes_labels:
    if np.any(nuscenes_boxes[label]):
        nuscenes_lwh_mean[label] = np.mean(nuscenes_boxes[label], axis=0)

print("nuscenes_lwh_mean = ", nuscenes_lwh_mean)
# {1: array([4.6190248, 1.9605241, 1.7350271], dtype=float32),
#  2: array([0.7286322 , 0.66744876, 1.7643346], dtype=float32),
#  4: array([1.9109839, 0.6932545, 1.3815018], dtype=float32)}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Calculate mean of length, width and height for Waymo dataset  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
waymo_anno_dir = "/project/mira/personal/timmy8986/3dal_pytorch/data/Waymo/train/annos"
waymo_lidar_dir = "/project/mira/personal/timmy8986/3dal_pytorch/data/Waymo/train/lidar"

anno_files = os.listdir(waymo_anno_dir)
# lidar_files = os.listdir(waymo_lidar_dir)

waymo_labels = {1: 'Vehicle', 2: 'Pedestrian', 4: 'Cyclist'}
waymo_boxes = {1: np.empty((0,3), np.float32), 2: np.empty((0,3), np.float32), 4: np.empty((0,3), np.float32)}
waymo_lwh_mean = {1: np.zeros((1,3), np.float32), 2: np.zeros((1,3), np.float32), 4: np.zeros((1,3), np.float32)}

sample_num = 10000
for anno_file in tqdm(anno_files[:sample_num]):
    with open(os.path.join(waymo_anno_dir, anno_file), 'rb') as pk:
        anno = pickle.load(pk)
        for anno_obj in anno['objects']:
            label = anno_obj['label']
            box = anno_obj['box']
            if label in waymo_labels:
                waymo_boxes[label] = np.vstack((waymo_boxes[label], box[3:6]))

for label in waymo_labels:
    if np.any(waymo_boxes[label]):
        waymo_lwh_mean[label] = np.mean(waymo_boxes[label], axis=0)

print("waymo_lwh_mean = ", waymo_lwh_mean)
# sample_num = 10000
# {1: array([4.757425 , 2.0977216, 1.7924396], dtype=float32), 
#  2: array([0.915268 , 0.8720095, 1.714554 ], dtype=float32), 
#  4: array([1.7481283 , 0.82206863, 1.7092806 ], dtype=float32)}


# # # # # # # # # # # # # # # # # #
# Statistical normalization (SN)  #
# # # # # # # # # # # # # # # # # #
n2w_anno_dir = "/project/mira/personal/nuscenes_to_waymo/train/annos"
n2w_lidar_dir = "/project/mira/personal/nuscenes_to_waymo/train/lidar"
n2w_deltas = {1: np.zeros((1,3), np.float32), 2: np.zeros((1,3), np.float32), 4: np.zeros((1,3), np.float32)}

for label in waymo_labels:
    n2w_deltas[label] = waymo_lwh_mean[label] - nuscenes_lwh_mean[label]

print("n2w_deltas = ", n2w_deltas)
# {1: array([0.13840008, 0.1371975 , 0.05741251], dtype=float32),
#  2: array([ 0.18663579,  0.20456076, -0.04978061], dtype=float32),
#  4: array([-0.16285563,  0.12881416,  0.32777882], dtype=float32)}

input_dict = {  'nuscenes_anno_dir': nuscenes_anno_dir,
                'nuscenes_lidar_dir': nuscenes_lidar_dir,
                'waymo_labels': waymo_labels,
                'n2w_deltas': n2w_deltas,
                'n2w_anno_dir': n2w_anno_dir,
                'n2w_lidar_dir': n2w_lidar_dir  }

def nuscenes_to_waymo(input_dict, anno_file):
    nuscenes_anno_dir = input_dict['nuscenes_anno_dir']
    nuscenes_lidar_dir = input_dict['nuscenes_lidar_dir']
    waymo_labels = input_dict['waymo_labels']
    n2w_deltas = input_dict['n2w_deltas']
    n2w_anno_dir = input_dict['n2w_anno_dir']
    n2w_lidar_dir = input_dict['n2w_lidar_dir']

    with open(os.path.join(nuscenes_anno_dir, anno_file), 'rb') as anno_pk:
        anno = pickle.load(anno_pk)
    with open(os.path.join(nuscenes_lidar_dir, anno_file), 'rb') as lidar_pk:
        lidar = pickle.load(lidar_pk)
        world_xyz = lidar['lidars']['points_xyz']

    for anno_obj in anno['objects']:
        # for anno
        label = anno_obj['label']
        if label not in waymo_labels:
            continue
        source_box = anno_obj['box']
        target_box = np.copy(source_box)
        target_box[3:6] = target_box[3:6] + n2w_deltas[label]
        anno_obj['box'] = target_box

        # for lidar
        X, Y, Z, Ls, Ws, Hs, _, _, heading_angle = source_box
        heading_vector = np.array([np.cos(heading_angle), np.sin(heading_angle)])
        lateral_vector = np.cross(np.array([0,0,1]), np.append(heading_vector, np.zeros(1)))[:2]

        # Xs, Ys, Zs is the bottom right corner of the bounding box (x: forward / y: left / z: upward)
        Xs, Ys = np.array([X, Y]) - heading_vector * Ls / 2 - lateral_vector * Ws / 2
        Zs = Z - Hs / 2

        # world coordinate system to bounding box coordinate system
        # w = R_wb * b + t_wb
        R_wb = np.array([[np.cos(heading_angle), -np.sin(heading_angle), 0],
                         [np.sin(heading_angle),  np.cos(heading_angle), 0],
                         [                    0,                      0, 1]])
        t_wb = np.array([[Xs, Ys, Zs]])
        
        bbox_xyz = np.linalg.inv(R_wb) @ (world_xyz - t_wb).T
        bbox_xyz = bbox_xyz.T

        bbox_center_xyz = np.array([[Ls, Ws, Hs]]) / 2
        
        mask = (bbox_xyz[:,0]>=0) & (bbox_xyz[:,0]<=Ls) & \
               (bbox_xyz[:,1]>=0) & (bbox_xyz[:,1]<=Ws) & \
               (bbox_xyz[:,2]>=0) & (bbox_xyz[:,2]<=Hs)
        
        rescale_ratio = target_box[3:6]/source_box[3:6]
        bbox_rescale_xyz = bbox_center_xyz + (bbox_xyz[mask] - bbox_center_xyz) * rescale_ratio

        # bounding box coordinate system to world coordinate system
        world_new_xyz = R_wb @ bbox_rescale_xyz.T + t_wb.T
        world_new_xyz = world_new_xyz.T

        lidar['lidars']['points_xyz'][mask] = world_new_xyz
        
    # Save DA annotations
    with open(os.path.join(n2w_anno_dir, anno_file), 'wb') as anno_pk:
        pickle.dump(anno, anno_pk)

    # Save DA lidar
    with open(os.path.join(n2w_lidar_dir, anno_file), 'wb') as lidar_pk:
        pickle.dump(lidar, lidar_pk)


with Pool() as pool:
    for anno_file in anno_files:
        pool.apply_async(nuscenes_to_waymo, args=(input_dict, anno_file))
    pool.close()
    pool.join()
