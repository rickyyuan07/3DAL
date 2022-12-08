import os, sys
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from nuscenes.nuscenes import NuScenes

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Calculate mean of length, width and height for Waymo dataset  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
waymo_anno_dir = "/project/mira/personal/timmy8986/3dal_pytorch/data/Waymo/train/annos"
waymo_lidar_dir = "/project/mira/personal/timmy8986/3dal_pytorch/data/Waymo/train/lidar"

anno_files = os.listdir(waymo_anno_dir)
lidar_files = os.listdir(waymo_lidar_dir)

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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Calculate mean of length, width and height for NuScenes dataset #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
nuscenes_dir = '/tmp2/tkyen/nuscenes'
nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_dir, verbose=True)

nuscenes_labels = waymo_labels
nuscenes_boxes = {1: np.empty((0,3), np.float32), 2: np.empty((0,3), np.float32), 4: np.empty((0,3), np.float32)}
nuscenes_lwh_mean = {1: np.zeros((1,3), np.float32), 2: np.zeros((1,3), np.float32), 4: np.zeros((1,3), np.float32)}

for sample_annotation in nusc.sample_annotation:
    category_name = sample_annotation['category_name']
    size = np.array(sample_annotation['size'])

    if category_name in ['vehicle.car']:
        nuscenes_boxes[1] = np.vstack((nuscenes_boxes[1], size))
    elif 'human.pedestrian' in category_name:
        nuscenes_boxes[2] = np.vstack((nuscenes_boxes[2], size))
    if category_name in ['vehicle.bicycle', 'vehicle.motorcycle']:
        nuscenes_boxes[4] = np.vstack((nuscenes_boxes[4], size))

for label in nuscenes_labels:
    if np.any(nuscenes_boxes[label]):
        lwh_mean = np.mean(nuscenes_boxes[label], axis=0) # w, l, h
        lwh_mean[[0, 1]] = lwh_mean[[1, 0]]
        nuscenes_lwh_mean[label] = lwh_mean               # l, w, h

print("nuscenes_lwh_mean = ", nuscenes_lwh_mean)


# # # # # # # # # # # # # # # # # #
# Statistical normalization (SN)  #
# # # # # # # # # # # # # # # # # #
w2n_anno_dir = "/tmp2/tkyen/3DAL/da/waymo_to_nuscenes/train/annos"
w2n_lidar_dir = "/tmp2/tkyen/3DAL/da/waymo_to_nuscenes/train/lidar"
w2n_deltas = {1: np.zeros((1,3), np.float32), 2: np.zeros((1,3), np.float32), 4: np.zeros((1,3), np.float32)}

for label in waymo_labels:
    w2n_deltas[label] = nuscenes_lwh_mean[label] - waymo_lwh_mean[label]

print("w2n_deltas = ", w2n_deltas)

input_dict = {  'waymo_anno_dir': waymo_anno_dir,
                'waymo_lidar_dir': waymo_lidar_dir,
                'waymo_labels': waymo_labels,
                'w2n_deltas': w2n_deltas,
                'w2n_anno_dir': w2n_anno_dir,
                'w2n_lidar_dir': w2n_lidar_dir  }

def waymo_to_nuscenes(input_dict, anno_file):
    waymo_anno_dir = input_dict['waymo_anno_dir']
    waymo_lidar_dir = input_dict['waymo_lidar_dir']
    waymo_labels = input_dict['waymo_labels']
    w2n_deltas = input_dict['w2n_deltas']
    w2n_anno_dir = input_dict['w2n_anno_dir']
    w2n_lidar_dir = input_dict['w2n_lidar_dir']

    with open(os.path.join(waymo_anno_dir, anno_file), 'rb') as anno_pk:
        anno = pickle.load(anno_pk)
    with open(os.path.join(waymo_lidar_dir, anno_file), 'rb') as lidar_pk:
        lidar = pickle.load(lidar_pk)
        world_xyz = lidar['lidars']['points_xyz']

    for anno_obj in anno['objects']:
        # for anno
        label = anno_obj['label']
        if label not in waymo_labels:
            continue
        source_box = anno_obj['box']
        target_box = np.copy(source_box)
        target_box[3:6] = target_box[3:6] + w2n_deltas[label]
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
    with open(os.path.join(w2n_anno_dir, anno_file), 'wb') as anno_pk:
        pickle.dump(anno, anno_pk)

    # Save DA lidar
    with open(os.path.join(w2n_lidar_dir, anno_file), 'wb') as lidar_pk:
        pickle.dump(lidar, lidar_pk)


mp_pool = Pool(8)
for anno_file in tqdm(anno_files):
    mp_pool.apply_async(waymo_to_nuscenes, args=(input_dict, anno_file))

mp_pool.close()
mp_pool.join()