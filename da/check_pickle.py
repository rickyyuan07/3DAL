import os
import pickle
from multiprocessing import Pool, Lock
import time

lock = Lock()

w2n_anno_dir = "/project/mira/personal/waymo_to_nuscenes/train/annos"
w2n_lidar_dir = "/project/mira/personal/waymo_to_nuscenes/train/lidar"

anno_files = os.listdir(w2n_anno_dir)
# anno_files = ['seq_134_frame_93.pkl','seq_135_frame_81.pkl','seq_515_frame_57.pkl']

def check_pickle(anno_file):
    try:
        with open(os.path.join(w2n_anno_dir, anno_file), 'rb') as pk:
            anno = pickle.load(pk)

        with open(os.path.join(w2n_lidar_dir, anno_file), 'rb') as pk:
            lidar = pickle.load(pk)
    except:
        lock.acquire()
        print(anno_file)
        lock.release()

start_time = time.time()
with Pool() as pool:
    for anno_file in anno_files:
        pool.apply_async(check_pickle, args=(anno_file,))
    pool.close()
    pool.join()
print("time duration = ", time.time()-start_time)