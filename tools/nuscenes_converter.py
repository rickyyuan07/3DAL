import pickle
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

dataroot = '/home/master/10/cytseng/data/sets/nuscenes/v1.0-mini/'
version = 'v1.0-mini'
nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

for seq_id, scene in enumerate(nusc.scene):
    print('processed scene: ', scene['name'], '(', seq_id+1, '/', len(nusc.scene), ')')
    # Get first sample token
    sample_token = scene['first_sample_token']
    frame_id = 0
    while sample_token:
        sample = nusc.get('sample', sample_token)
        # Get the pointcloud and lidar sensor
        sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        # Read the pointcloud
        pc = LidarPointCloud.from_file(nusc.get_sample_data_path(sample['data']['LIDAR_TOP']))

        # Transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']))
        pc.translate(np.array(cs_record['translation']))

        # Get points in lidar frame
        points = pc.points[:3, :].T
        intensity = pc.points[3, :]
        n = points.shape[0]
        points_feature = np.stack([intensity, np.zeros((n, ))], axis=1)

        # Prepare lidar data and write to file (as pickle)
        lidar = {
            'scene_name': scene['name'],
            'frame_name': f"{scene['name']}_location_sf_Day_{sd_record['timestamp']}",
            'frame_id': frame_id,
            'lidars': {
                'points_xyz': points,
                'points_feature': points_feature
            }
        }
        filename = f"seq_{seq_id}_frame_{frame_id}.pkl"
        with open('/home/extra/rickyyuan/dataset/nuscenes/v1.0-mini/lidar/' + filename, 'wb') as f:
            pickle.dump(lidar, f)


        # Get ego vehicle pose transformation matrix
        ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])
        ego_translation = ego_pose['translation']
        ego_rotation = Quaternion(ego_pose['rotation'])
        ego_to_global = np.eye(4)
        ego_to_global[:3, 3] = ego_translation
        ego_to_global[:3, :3] = ego_rotation.rotation_matrix

        objects = []
        # Read annotation
        for obj_idx, ann_token in enumerate(sample['anns']):
            ann = nusc.get('sample_annotation', ann_token)

            lbl = -1
            if "human.pedestrian" in ann['category_name']:
                lbl = 2
            elif "vehicle.bicycle" in ann['category_name'] or "vehicle.motorcycle" in ann['category_name']:
                lbl = 3
            elif "vehicle.car" in ann['category_name'] or "vehicle.bus" in ann['category_name'] or "vehicle.truck" in ann['category_name']:
                lbl = 1
            
            # print(ann['category_name'], lbl)

            if lbl == -1:
                continue

            box = nusc.get_box(ann_token)
            box_ego = box.copy()
            box_ego.translate(-np.array(ego_translation))
            box_ego.rotate(Quaternion(ego_rotation.inverse))
            box_ego = np.concatenate([box_ego.center, box_ego.wlh, [box_ego.orientation.yaw_pitch_roll[0]]])
            # Get object
            obj = {
                'id': obj_idx,
                'name': ann['instance_token'],
                'label': ann['category_name'],
                'box': box_ego,
                'num_points': ann['num_lidar_pts'],
                'detection_difficulty_level': 0,
                'combined_difficulty_level': 0,
                'global_speed': 0,
                'global_accel': 0
            }
            objects.append(obj)

        # Prepare object data and write to file (as pickle)
        annos = {
            'scene_name': scene['name'],
            'frame_name': f"{scene['name']}_location_sf_Day_{sd_record['timestamp']}",
            'frame_id': frame_id,
            'veh_to_global': ego_to_global,
            'objects': objects
        }

        frame_id += 1
        sample_token = sample['next']

        # Write to file
        with open('/home/extra/rickyyuan/dataset/nuscenes/v1.0-mini/annos/' + filename, 'wb') as f:
            pickle.dump(annos, f)