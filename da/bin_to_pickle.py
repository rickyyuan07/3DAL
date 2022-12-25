from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import label_pb2

# gt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_w2n_new/gt_preds.bin"
# dt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_w2n_new/detection_pred.bin"

gt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_waymo/waymo_centerpoint_val_gt_preds.bin"
dt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_waymo/waymo_centerpoint_val_detection_pred.bin"

# gt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_w2n_sn/gt_preds.bin"
# dt_bin_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_w2n_sn/detection_pred.bin"

gt_pickle_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_w2n_new/gt_preds.pickle"
dt_pickle_path = "/home/tkyen/3DAL/waymo-open-dataset/evaluation_w2n_new/detection_pred.pickle"

obj_gt = metrics_pb2.Objects()
f = open(gt_bin_path, "rb")
obj_gt.ParseFromString(f.read())
f.close()

obj_dt = metrics_pb2.Objects()
f = open(dt_bin_path, "rb")
obj_dt.ParseFromString(f.read())
f.close()

# print(dir(objects))
print("gt_lenth = ", len(obj_gt.objects))
print("dt_lenth = ", len(obj_dt.objects))

gt_first_time = obj_gt.objects[0].frame_timestamp_micros
gt_last_time = obj_gt.objects[-1].frame_timestamp_micros

dt_first_time = obj_dt.objects[0].frame_timestamp_micros
dt_last_time = obj_dt.objects[-1].frame_timestamp_micros
print("gt_first_time = ", gt_first_time, "; gt_last_time = ", gt_last_time)
print("dt_first_time = ", dt_first_time, "; dt_last_time = ", dt_last_time)


gt_time_list = list()
last_time = -1
for obj in obj_gt.objects:
    if obj.frame_timestamp_micros != last_time:
        gt_time_list.append(obj.frame_timestamp_micros)
        last_time = obj.frame_timestamp_micros
print("gt_time_length = ", len(gt_time_list))

dt_time_list = list()
last_time = -1
mismatch_num = 0
for obj in obj_dt.objects:
    if obj.frame_timestamp_micros != last_time:
        if obj.frame_timestamp_micros not in gt_time_list:
            mismatch_num = mismatch_num + 1
        dt_time_list.append(obj.frame_timestamp_micros)
        last_time = obj.frame_timestamp_micros
print("dt_time_length = ",len(dt_time_list))
print("mismatch_num = ",mismatch_num)
