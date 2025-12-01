import os
import numpy as np
import pickle as pkl
import argparse
from imblearn.over_sampling import SMOTE
from preprocess_utils import get_data, save_data

classes_amount = 8

def count_data_classes(data_count):
    cnt = np.zeros(classes_amount)
    for i in data_count:
        cnt[int(i)] += 1
    print(cnt)

parser = argparse.ArgumentParser()
parser.add_argument('--axis', type=int, help='axis', default=3)
parser.add_argument('--overlap', type=int, help='overlap', default=50)
parser.add_argument('--slide', type=int, help='slide', default=96)
args = parser.parse_args()

save_path = f"./data.pkl"
with open(save_path, "rb") as f:
    data = pkl.load(f)

train_data, train_label, val_data, val_label, test_data, test_label = get_data(data)

print("Original data distribution")
print('================================train_data==============================')
count_data_classes(train_label)
print('================================val_data================================')
count_data_classes(val_label)
print('================================test_data===============================')
count_data_classes(test_label)
print("")
# dim 3 to dim 2
train_data = train_data.reshape(train_data.shape[0], -1)
val_data = val_data.reshape(val_data.shape[0], -1)

# smote
smote = SMOTE()
train_data, train_label = smote.fit_resample(train_data, train_label)
val_data, val_label = smote.fit_resample(val_data, val_label)

# dim 3 to dim 2
train_data = train_data.reshape(-1, args.slide, args.axis)
val_data = val_data.reshape(-1, args.slide, args.axis)

print("SMOTE data distribution")
print('================================train_data==============================')
count_data_classes(train_label)
print('================================val_data================================')
count_data_classes(val_label)
print('================================test_data===============================')
count_data_classes(test_label)
print("")

data = save_data(data, train_data, train_label, val_data, val_label)
print("train data:", data["train"]["data"].shape)
print("train label:", data["train"]["label"].shape)
print("val data:", data["val"]["data"].shape)
print("val label:", data["val"]["label"].shape)
print("test data:", data["test"]["data"].shape)
print("test label:", data["test"]["label"].shape)

with open(save_path, 'wb') as f:
    pkl.dump(data, f)


















