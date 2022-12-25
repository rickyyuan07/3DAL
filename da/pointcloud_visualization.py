import os, sys
import pickle
import numpy as np
from tqdm import tqdm
import open3d as o3d

# ext_matrix is from world to imu
def imu2world(imu_xyz, ext_matrix):
    num = imu_xyz.shape[0]
    imu_xyz = np.hstack([imu_xyz.copy(), np.ones([num, 1])])
    world_xyz = ext_matrix @ imu_xyz.T
    world_xyz = world_xyz.T[:, :3]
    return world_xyz

# ext_matrix is from world to imu
def world2imu(world_xyz, ext_matrix):
    num = world_xyz.shape[0]
    world_xyz = np.hstack([world_xyz.copy(), np.ones([num, 1])])
    imu_xyz = np.linalg.inv(ext_matrix) @ world_xyz.T
    imu_xyz = imu_xyz.T[:, :3]
    return imu_xyz

def load_point_cloud(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

if __name__ == '__main__':
    frame_type = 'imu' #'global' or 'imu'
    root_dir = "/home/tkyen"
    # root_dir = "/home/tkyen/workspace/opencv_practice"
    waymo_anno_dir = os.path.join(root_dir, "3DAL/da/Waymo/train/annos")
    waymo_lidar_dir = os.path.join(root_dir, "3DAL/da/Waymo/train/lidar")
    w2n_anno_dir = os.path.join(root_dir, "3DAL/da/Waymo/train_w2n/annos")
    w2n_lidar_dir = os.path.join(root_dir, "3DAL/da/Waymo/train_w2n/lidar")

    nuscenes_anno_dir = os.path.join(root_dir, "3DAL/da/Nuscenes/train/annos")
    nuscenes_lidar_dir = os.path.join(root_dir, "3DAL/da/Nuscenes/train/lidar")
    n2w_anno_dir = os.path.join(root_dir, "3DAL/da/Nuscenes/train_n2w/annos")
    n2w_lidar_dir = os.path.join(root_dir, "3DAL/da/Nuscenes/train_n2w/lidar")

    nuscenes_anno_dir = os.path.join(root_dir, "3DAL/nuscenes_dataset/val/annos")
    nuscenes_lidar_dir = os.path.join(root_dir, "3DAL/nuscenes_dataset/val/lidar")

    anno_dir = nuscenes_anno_dir
    lidar_dir = nuscenes_lidar_dir

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    vis.add_geometry(frame)

    anno_files = os.listdir(anno_dir)
    anno_files = sorted(anno_files, key = lambda anno_file : int(os.path.splitext(anno_file)[0].split('_')[1]) * 1000 + int(os.path.splitext(anno_file)[0].split('_')[-1]))

    for i, anno_file in enumerate(anno_files):
        print(anno_file)
        if i==20:
            break

        with open(os.path.join(anno_dir, anno_file), 'rb') as anno_pk:
            anno = pickle.load(anno_pk)
            ext_matrix = anno['veh_to_global'].reshape((4, 4)) # global to imu
            if i == 0:
                ext_matrix_0 = ext_matrix

        with open(os.path.join(lidar_dir, anno_file), 'rb') as lidar_pk:
            lidar = pickle.load(lidar_pk)
            imu_xyz = lidar['lidars']['points_xyz']

        assert frame_type=='global' or frame_type=='imu', "frame_type should be 'global' or 'imu'"
        if frame_type == 'global':
            world_xyz = imu2world(imu_xyz, ext_matrix)
            pcd = load_point_cloud(world_xyz)
        elif frame_type == 'imu':
            if i == 0:
                pcd = load_point_cloud(imu_xyz)
            else:
                world_xyz = imu2world(imu_xyz, ext_matrix)
                imu_xyz = world2imu(world_xyz, ext_matrix_0)
                pcd = load_point_cloud(imu_xyz)
        vis.add_geometry(pcd)

        if i == 0:
            color = np.array([1, 0, 0])
        else:
            color = np.random.rand(3)
        for anno_obj in anno['objects']:
            # for anno
            label = anno_obj['label']
            bbox = anno_obj['box']

            # X, Y, Z is the center of the bounding box in imu frame
            X, Y, Z, Ls, Ws, Hs, _, _, heading_angle = bbox

            # for the center of the bounding box
            if frame_type == 'global':
                center_imu_xyz = bbox[:3][None, :]
                center_world_xyz = imu2world(center_imu_xyz, ext_matrix)
                X, Y, Z = np.squeeze(center_world_xyz)
            elif frame_type == 'imu' and i > 0:
                center_imu_xyz = bbox[:3][None, :]
                center_world_xyz = imu2world(center_imu_xyz, ext_matrix)
                center_imu_xyz = world2imu(center_world_xyz, ext_matrix_0)
                X, Y, Z = np.squeeze(center_imu_xyz)

            # for the heading_vector
            heading_vector_imu = np.array([np.cos(heading_angle), np.sin(heading_angle)])
            if frame_type == 'global':
                vector_imu_xyz = np.empty((0,3))
                vector_imu_xyz = np.append(vector_imu_xyz, np.array([[0, 0, 0]]), axis=0)
                vector_imu_xyz = np.append(vector_imu_xyz, np.array([[heading_vector_imu[0], heading_vector_imu[1], 0]]), axis=0)
                vector_world_xyz = imu2world(vector_imu_xyz, ext_matrix)
                heading_vector_world = vector_world_xyz[1, :] - vector_world_xyz[0, :]
                heading_vector = heading_vector_world[:2]
            elif frame_type == 'imu':
                if i == 0:
                    heading_vector = heading_vector_imu
                else:
                    vector_imu_xyz = np.empty((0,3))
                    vector_imu_xyz = np.append(vector_imu_xyz, np.array([[0, 0, 0]]), axis=0)
                    vector_imu_xyz = np.append(vector_imu_xyz, np.array([[heading_vector_imu[0], heading_vector_imu[1], 0]]), axis=0)
                    vector_world_xyz = imu2world(vector_imu_xyz, ext_matrix)
                    vector_imu_xyz = world2imu(vector_world_xyz, ext_matrix_0)
                    heading_vector_imu = vector_imu_xyz[1, :] - vector_imu_xyz[0, :]
                    heading_vector = heading_vector_imu[:2]

            lateral_vector = np.cross(np.array([0,0,1]), np.append(heading_vector, np.zeros(1)))[:2]

            # Xs, Ys, Zs is the bottom right corner of the bounding box (x: forward / y: left / z: upward)
            Xs, Ys = np.array([X, Y]) - heading_vector * Ls / 2 - lateral_vector * Ws / 2
            Zs = Z - Hs / 2
            origin = np.array([Xs, Ys, Zs])
            heading_vector = np.pad(heading_vector, (0, 1)) * Ls
            lateral_vector = np.pad(lateral_vector, (0, 1)) * Ws
            height_vector = np.array([0, 0, 1]) * Hs

            points = [
                (origin).tolist(),
                (origin + heading_vector).tolist(),
                (origin + lateral_vector).tolist(),
                (origin + heading_vector + lateral_vector).tolist(),
                (origin + height_vector).tolist(),
                (origin + height_vector + heading_vector).tolist(),
                (origin + height_vector + lateral_vector).tolist(),
                (origin + height_vector + heading_vector + lateral_vector).tolist(),
            ]

            lines = [
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 3],
                [4, 5],
                [4, 6],
                [5, 7],
                [6, 7],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
            colors = [color for i in range(len(lines))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(line_set)

    vis.run()
    vis.destroy_window()
