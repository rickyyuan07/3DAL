import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm

def main():
    # python3 tools/trackData.py --work_dir work_dirs/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel/train
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', help='Path to working dir.')
    args = parser.parse_args()

    with open(os.path.join(args.work_dir, 'trackData_one.pkl'), 'rb') as f:
        track_one = pickle.load(f)
    with open(os.path.join(args.work_dir, 'trackData_two.pkl'), 'rb') as f:
        track_two = pickle.load(f)
    track = dict(list(track_one.items()) + list(track_two.items()))
    n_ped, n_veh, n_cyc = 0, 0, 0
    tracking = {}
    for token, frame in tqdm(track.items()):
        ids, types, bboxs = frame['id'], frame['type'], frame['bbox']
        scores, points, matchs = frame['score'], frame['point'], frame['match']
        
        for idx in range(len(ids)):
            if matchs[idx] != None:
                if int(types[idx]) == 2:
                    n_ped += 1
                elif int(types[idx]) == 1:
                    n_veh += 1
                elif int(types[idx]) == 4:
                    n_cyc += 1
            
            if ids[idx] not in tracking:
                tracking[ids[idx]] = {}
                tracking[ids[idx]]['type'] = [types[idx]]
                tracking[ids[idx]]['bbox'] = [bboxs[idx]]
                tracking[ids[idx]]['score'] = [scores[idx]]
                tracking[ids[idx]]['point'] = [points[idx]]
                tracking[ids[idx]]['match'] = [matchs[idx]]
                tracking[ids[idx]]['token'] = [token]
            else:
                tracking[ids[idx]]['type'].append(types[idx])
                tracking[ids[idx]]['bbox'].append(bboxs[idx])
                tracking[ids[idx]]['score'].append(scores[idx])
                tracking[ids[idx]]['point'].append(points[idx])
                tracking[ids[idx]]['match'].append(matchs[idx])
                tracking[ids[idx]]['token'].append(token)
    print(n_veh, n_ped, n_cyc)
    tracking_list = list(tracking.items())
    '''
    tracking_one = dict(tracking_list[:len(tracking_list) // 2])
    tracking_two = dict(tracking_list[len(tracking_list) // 2:])
    with open(os.path.join(args.work_dir, 'track_one.pkl'), 'wb') as f:
        pickle.dump(tracking_one, f)
    with open(os.path.join(args.work_dir, 'track_two.pkl'), 'wb') as f:
        pickle.dump(tracking_two, f)
    '''
    with open(os.path.join(args.work_dir, 'track.pkl'), 'wb') as f:
        pickle.dump(dict(tracking_list), f)
if __name__ == '__main__':
    main()
