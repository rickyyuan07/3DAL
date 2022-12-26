import os, sys
import pickle
import numpy as np
from tqdm import tqdm
import open3d as o3d
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import label_pb2

# gt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_w2n_new/gt_preds.bin"
# gt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_w2n_new/gt_preds_2sw.bin"
# dt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_w2n_new/detection_pred.bin"

# gt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_waymo/waymo_centerpoint_val_gt_preds.bin"
# dt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_waymo/waymo_centerpoint_val_detection_pred.bin"

gt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_w2n_sn/gt_preds.bin"
# dt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_w2n_sn/detection_pred.bin"
dt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_nuscenes/detection_pred.bin"

obj_gt = metrics_pb2.Objects()
f = open(gt_bin_path, "rb")
obj_gt.ParseFromString(f.read())
f.close()

obj_dt = metrics_pb2.Objects()
f = open(dt_bin_path, "rb")
obj_dt.ParseFromString(f.read())
f.close()

print(dir(objects))
print("gt_lenth = ", len(obj_gt.objects))
print("dt_lenth = ", len(obj_dt.objects))

gt_first_time = obj_gt.objects[0].frame_timestamp_micros
gt_last_time = obj_gt.objects[-1].frame_timestamp_micros

dt_first_time = obj_dt.objects[0].frame_timestamp_micros
dt_last_time = obj_dt.objects[-1].frame_timestamp_micros
print("gt_first_time = ", gt_first_time, "; gt_last_time = ", gt_last_time)
print("dt_first_time = ", dt_first_time, "; dt_last_time = ", dt_last_time)


gt_time_list = list()
gt_first_obj_list = list()
last_time = -1
for obj in obj_gt.objects:
    if obj.frame_timestamp_micros == gt_first_time:
        gt_first_obj_list.append(obj)

    if obj.frame_timestamp_micros != last_time:
        gt_time_list.append(obj.frame_timestamp_micros)
        last_time = obj.frame_timestamp_micros
print("gt_time_length = ", len(gt_time_list))

dt_time_list = list()
dt_first_obj_list = list()
last_time = -1
mismatch_num = 0
for obj in obj_dt.objects:
    if obj.frame_timestamp_micros == gt_first_time:
        dt_first_obj_list.append(obj)

    if obj.frame_timestamp_micros != last_time:
        if obj.frame_timestamp_micros not in gt_time_list:
            mismatch_num = mismatch_num + 1
        dt_time_list.append(obj.frame_timestamp_micros)
        last_time = obj.frame_timestamp_micros
print("dt_time_length = ",len(dt_time_list))
print("mismatch_num = ",mismatch_num)


vis = o3d.visualization.Visualizer()
vis.create_window()
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
vis.add_geometry(frame)

def draw_3d_bbox(obj_list, color, vis):
    for obj in obj_list:
        X = obj.object.box.center_x
        Y = obj.object.box.center_y
        Z = obj.object.box.center_z
        W = obj.object.box.width
        L = obj.object.box.length
        H = obj.object.box.height
        heading_angle = obj.object.box.heading

        # for the heading_vector
        heading_vector = np.array([np.cos(heading_angle), np.sin(heading_angle)])
        lateral_vector = np.cross(np.array([0,0,1]), np.append(heading_vector, np.zeros(1)))[:2]

        # Xs, Ys, Zs is the bottom right corner of the bounding box (x: forward / y: left / z: upward)
        X, Y = np.array([X, Y]) - heading_vector * L / 2 - lateral_vector * W / 2
        Z = Z - H / 2
        origin = np.array([X, Y, Z])
        heading_vector = np.pad(heading_vector, (0, 1)) * L
        lateral_vector = np.pad(lateral_vector, (0, 1)) * W
        height_vector = np.array([0, 0, 1]) * H

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

color = np.array([1, 0, 0])
draw_3d_bbox(gt_first_obj_list, color, vis)

color = np.array([0, 0, 1])
draw_3d_bbox(dt_first_obj_list, color, vis)

vis.run()
vis.destroy_window()
