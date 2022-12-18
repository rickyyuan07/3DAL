import pickle
import numpy as np
from pyquaternion.quaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

val_scenes = \
    ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']

# version = 'v1.0-mini'
version = 'v1.0-trainval'
dataroot = f'/home/master/10/cytseng/data/sets/nuscenes/{version}/'
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
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
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
                'points_xyz': points.astype(np.float32),
                'points_feature': points_feature.astype(np.float32),
            }
        }
        filename = f"seq_{seq_id}_frame_{frame_id}.pkl"
        if scene['name'] in val_scenes:
            with open(f'/home/extra/rickyyuan/dataset/nuscenes/{version}/val/lidar/' + filename, 'wb') as f:
                pickle.dump(lidar, f)
        else:
            with open(f'/home/extra/rickyyuan/dataset/nuscenes/{version}/train/lidar/' + filename, 'wb') as f:
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
                lbl = 4
            elif "vehicle.car" in ann['category_name']:
                lbl = 1
            
            # print(ann['category_name'], lbl)

            if lbl == -1:
                continue

            box = nusc.get_box(ann_token)
            box_ego = box.copy()
            box_ego.translate(-np.array(ego_translation))
            box_ego.rotate(Quaternion(ego_rotation).inverse)
            # swap w and l
            box_ego.wlh = np.array([box_ego.wlh[1], box_ego.wlh[0], box_ego.wlh[2]])
            box_ego = np.concatenate([box_ego.center, box_ego.wlh, [0, 0, box_ego.orientation.yaw_pitch_roll[0]]])
            # Get object
            obj = {
                'id': obj_idx,
                'name': ann['instance_token'],
                'label': lbl,
                'box': box_ego.astype(np.float32),
                'num_points': ann['num_lidar_pts'],
                'detection_difficulty_level': 0,
                'combined_difficulty_level': 0,
                'global_speed': np.array([0, 0]).astype(np.float32),
                'global_accel': np.array([0, 0]).astype(np.float32),
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

        # Write to file
        if scene['name'] in val_scenes:
            with open(f'/home/extra/rickyyuan/dataset/nuscenes/{version}/val/annos/' + filename, 'wb') as f:
                pickle.dump(annos, f)
        else:
            with open(f'/home/extra/rickyyuan/dataset/nuscenes/{version}/train/annos/' + filename, 'wb') as f:
                pickle.dump(annos, f)

        frame_id += 1
        sample_token = sample['next']