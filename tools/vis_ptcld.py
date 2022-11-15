import os
import pickle
import argparse
import numpy as np
import open3d as o3d
from math import sin, cos
import numpy.matlib as matlib

class bcolors:
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    ENDC = '\033[0m'

def get_lineset(points: np.ndarray, color: list):
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    colors = [color for i in range(len(lines))]
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    
    return lineset

def get_points(xmin: np.float, xmax: np.float, ymin: np.float, ymax: np.float, zmin: np.float, zmax: np.float):
    points = [
        [xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
        [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]
    ]
    return np.asarray(points)

def rotz(angle: np.float):
    c = np.cos(angle)
    s = np.sin(angle)
    rotz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    return rotz

def draw_3dbbox(bboxs: np.ndarray, vis: o3d.visualization.Visualizer, color: list, scores: np.ndarray=None, thresh: np.float=0.5):
    for i, bbox in enumerate(bboxs):
        if (scores is not None) and scores[i] < thresh:
            continue
        
        x, y, z, l, w, h, heading = bbox
        points = get_points(-l / 2, l / 2, -w / 2, w / 2, -h / 2, h / 2)
        if scores is not None:
            print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Box: ({x:.2f}, {y:.2f}, {z:.2f}, {l:.2f}, {w:.2f}, {h:.2f}, {heading:.2f})')
        
        points = rotz(heading) @ points.T + bbox[:3, np.newaxis]
        points = points.T
        
        lineset = get_lineset(points=points, color=color)
        vis.add_geometry(lineset)

def euler_to_so3(rpy: list):
    R_x = np.matrix([
        [1,           0,            0],
        [0, cos(rpy[0]), -sin(rpy[0])],
        [0, sin(rpy[0]),  cos(rpy[0])],
    ])
    R_y = np.matrix([
        [ cos(rpy[1]), 0, sin(rpy[1])],
        [           0, 1,           0],
        [-sin(rpy[1]), 0, cos(rpy[1])],
    ])
    R_z = np.matrix([
        [cos(rpy[2]), -sin(rpy[2]), 0],
        [sin(rpy[2]),  cos(rpy[2]), 0],
        [          0,            0, 1],
    ])
    return R_z * R_y * R_x

def build_se3_transform(xyzrpy: list):
    se3 = matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3

def sort_detections(detections):
    indices = []
    for det in detections:
        indices.append(det['frame_id'])

    rank = list(np.argsort(np.array(indices)))
    detections = [detections[r] for r in rank]
    return detections

if __name__ == '__main__':
    # python vis_ptcld.py --lidar waymo/seq_0_frame_0/lidar.pkl --annos waymo/seq_0_frame_0/annos.pkl --preds1 waymo/det_annos.pkl --preds2 waymo/one_box_est.pkl --token seq_0_frame_0.pkl
    parser = argparse.ArgumentParser()
    parser.add_argument('--lidar', help='Path to lidar.pkl.')
    parser.add_argument('--annos', help='Path to annos.pkl.')
    parser.add_argument('--preds1', help='Path to predicted bbox w/o temporal.')
    parser.add_argument('--preds2', help='Path to predicted bbox w/ temporal.')
    parser.add_argument('--token', help='Token name.')
    args = parser.parse_args()

    with open(args.lidar, 'rb') as f:
        lidar = pickle.load(f)
    points = lidar['lidars']['points_xyz']
    print(f'[{bcolors.OKBLUE}Info{bcolors.ENDC}] Point cloud shape: {points.shape}')
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=args.lidar, left=0, top=40)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
    render_option.point_color_option = o3d.visualization.PointColorOption.ZCoordinate
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(coordinate_frame)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points.astype(np.float64)))
    vis.add_geometry(pcd)

    # Draw groundtruth bbox (red)
    with open(args.annos, 'rb') as f:
        annos = pickle.load(f)
    objects = annos['objects']

    bboxs = np.array([obj['box'] for obj in objects])
    bboxs = bboxs[:, [0, 1, 2, 3, 4, 5, -1]]
    draw_3dbbox(bboxs, vis, color=[255, 0, 0])

    # Draw detection bbox (green)
    print(f'{bcolors.OKGREEN}> Detection{bcolors.ENDC}')
    with open(args.preds1, 'rb') as f:
        preds1 = pickle.load(f)
    preds1 = sort_detections(preds1)

    scores = preds1['scores'].numpy()
    label_preds = preds1['name'].numpy()
    box3d_lidar = preds1['boxes_lidar'].numpy()
    
    # box3d_lidar[:, -1] = -box3d_lidar[:, -1] - np.pi / 2
    # box3d_lidar = box3d_lidar[:, [0, 1, 2, 4, 3, 5, -1]]
    draw_3dbbox(box3d_lidar, vis, color=[0, 255, 0], scores=scores)
    
    # Run Visualizer
    view_control = vis.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = build_se3_transform([0, 0, 10, np.pi / 2, -np.pi / 2, 0])
    view_control.convert_from_pinhole_camera_parameters(params)
    vis.run()