import numpy as np
import os
import pickle as pkl
import pandas as pd
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--axis", type=int, help="axis", default=3)
parser.add_argument("--slide", type=int, help="slide", default=96)
parser.add_argument("--overlap", type=int, help="overlap", default=80)
args = parser.parse_args()

# ----定義資料裁切參數------------------
slide = args.slide
step = int(slide*((100-args.overlap)/100))
#------------------------------------
save_file = f"./{args.slide}"

if not os.path.exists(save_file):
    os.makedirs(save_file)

category = ['sitting', 'fall_down', 'sit_down', 'stand_up', 'walking', 'walk_stairs', 'push_up', 'jumping']

file_dir = "./SelfDataCollected"
save_path = f"./{args.slide}/data.pkl"

if args.axis == 3:
    channels = ["FreeAcc_X", "FreeAcc_Y", "FreeAcc_Z"]
elif args.axis == 6:
    channels = ["FreeAcc_X", "FreeAcc_Y", "FreeAcc_Z", "Gyr_X",	"Gyr_Y", "Gyr_Z"]
elif args.axis == 9:
    channels = ["Euler_X","Euler_Y", "Euler_Z",	"FreeAcc_X", "FreeAcc_Y", "FreeAcc_Z", "Gyr_X",	"Gyr_Y", "Gyr_Z"]

print(f"axis: {args.axis}\nwindows size: {args.slide}\noverlap: {args.overlap}%") 

def check_feature(three_dim_cut_map, category_idx):
    mean, std = np.mean(three_dim_cut_map, axis=2), np.std(three_dim_cut_map, axis=2)
    feature = True
    if category[category_idx] == 'fall':
        UCL = mean+2*std
        LCL = mean-2*std
    elif category[category_idx] == 'sit_down' or category[category_idx] == 'stand_up':
        UCL = mean+std
        LCL = mean-std
    else:
        UCL = np.full(args.axis, math.inf)
        LCL = np.full(args.axis, -math.inf)
    
    for idx in range(args.axis):
        if np.any(three_dim_cut_map[idx,:] > UCL[idx]) or np.any(three_dim_cut_map[idx,:] < LCL[idx]):
            feature = False
            return feature

    return 0

def sliding_window(two_dim_data, category_idx):
    # print(two_dim_data.shape) #(4936, 3)
    start = 0
    end = slide
    labels = [] # 放置每個視窗對應的標籤
    slide_array = np.empty((slide, len(channels), 0)) #存放視窗切割的數據(96, 3, 0)
    # print(slide.shape)
    while end <= two_dim_data.shape[0]:
        three_dim_cut_map = np.expand_dims(two_dim_data[start:end], axis=2) # axis=2增加1在第二個維度(96, 3, 1)
        # feature_in = check_feature(three_dim_cut_map, category_idx)
        # if feature_in is not None:
        slide_array = np.concatenate([slide_array, three_dim_cut_map], axis=2) #每切一片three_dim_cut_map就存進slide
        # print(slide_array.shape)
        # print(cut_map.shape)
        labels.append(category_idx) # 依照近來的動作存放標籤
        start += step
        end += step
        # print(labels)
    return slide_array.transpose(2, 0, 1), np.asarray(labels) # slide_array(資料片數, 每片長度, 通道數) labels(每片的標籤, )

def read_file(data_path, channels):
    data = pd.read_csv(data_path, skiprows=11)
    selected_columns = [data[channel].values for channel in channels]
    channels_array = np.stack(selected_columns, axis=1)  # shape: [T, 3]
    return channels_array


outer_keys = ['train', 'val', 'test']
inner_keys = ['data', 'label']
store_data = {key: {inner: [] for inner in inner_keys} for key in outer_keys}

for stage in os.listdir(file_dir):

    data_array = np.empty((0, slide, len(channels))) #定義好資料結構(資料數, 資料長度, 通道數)
    label_array = np.empty(0)

    for person_name in os.listdir(os.path.join(file_dir, stage)):

        for action in os.listdir(os.path.join(file_dir, stage, person_name)):

            if action == "sitting.csv" or  action == "walking.csv": # 針對CSV檔做data及label處理
                two_dim_data = read_file(os.path.join(file_dir, stage, person_name, action), channels)
                # print(f"{stage}:{person_name}:{action}:")
                for category_idx, category_name in enumerate(category):
                    # 用索引當成標籤
                    if category_name in action:
                        slide_data, labels = sliding_window(two_dim_data, category_idx)
                        # sitting and walking最終存放的動作資料及標籤
                        data_array = np.concatenate([data_array, slide_data])
                        label_array = np.concatenate([label_array, labels])
                         
            else:
                for action_file in os.listdir(os.path.join(file_dir, stage, person_name, action)):
                    for CSV in os.listdir(os.path.join(file_dir, stage, person_name, action, action_file)):
                        two_dim_data = read_file(os.path.join(file_dir, stage, person_name, action, action_file, CSV), channels) 
                        # print(f"{action}:{two_dim_data.shape}")
                        for category_idx, category_name in enumerate(category):
                            if category_name in action:
                                slide_data, labels = sliding_window(two_dim_data, category_idx)
                                data_array = np.concatenate([data_array, slide_data])
                                label_array = np.concatenate([label_array, labels])

            print(f"{stage}||{person_name}||{action}:{data_array.shape}")
            print(f"label:{label_array.shape}")
    print("\n")
    # 儲存每個階段的資料及標籤
    store_data[stage]["data"].append(data_array) # 'train': {'data': [], 'label': []}
    store_data[stage]["label"].append(label_array)       
# train_data_array = np.array(store_data["train"]["data"])
# print(train_data_array.shape) 
# 轉為列表
for stage in store_data:
    for data in store_data[stage]:
        store_data[stage][data] = np.concatenate(store_data[stage][data], axis=0)

with open(save_path,"wb") as f:
    pkl.dump(store_data, f)


                    
    
