import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix
from itertools import chain
import os
import seaborn as sns

def get_data(data):
    train_data = data["train"]["data"]
    train_label = data["train"]["label"]
    val_data = data["val"]["data"]
    val_label = data["val"]["label"]
    test_data = data["test"]["data"]
    test_label = data["test"]["label"]

    return train_data, train_label, val_data, val_label, test_data, test_label


class UCI_HAR_Dataset(Dataset):
    def __init__(self, time_data,frequency_data, labels):
        self.time_data = time_data
        self.frequency_data = frequency_data
        self.labels = labels

    def __len__(self):
        return len(self.time_data)
    
    def __getitem__(self, index): #__len__會自動呼叫到_getitem__
        data_T = self.time_data[index]
        data_F = self.frequency_data[index]
        y = self.labels[index]
        return data_T, data_F, y

def store_plot_data(train_acc, train_loss, val_acc, val_loss):
    all_data = {}
    all_data['train_acc'] = train_acc
    all_data['train_loss'] = train_loss
    all_data['val_acc'] = val_acc
    all_data['val_loss'] = val_loss

    return all_data

def plot_acc_curve(all_data, save_path):
    plt.figure()
    train_acc = np.asarray(all_data['train_acc'])
    x_axis = range(1, len(train_acc) + 1)
    val_acc = np.asarray(all_data['val_acc'])

    y_ticks_acc = np.arange(0, 1.05, 0.05) 
    plt.plot(x_axis, train_acc, label = "Training_acc", color = "r")
    plt.plot(x_axis, val_acc, label = "validation_acc", color = "g")
    plt.legend(loc = "lower right")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy curve")
    plt.grid()
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
    Acc_name = "Accuracy.png"

    acc_save_path = os.path.join(save_path, Acc_name)
    plt.savefig(acc_save_path)
    print("Save acc curve")

def plot_loss_curve(all_data, save_path):
    plt.figure()
    train_acc = np.asarray(all_data['train_loss'])
    x_axis = range(1, len(train_acc) + 1)
    val_acc = np.asarray(all_data['val_loss'])

    y_ticks_acc = np.arange(0, 1.05, 0.05) 
    plt.plot(x_axis, train_acc, label = "Training_loss", color = "r")
    plt.plot(x_axis, val_acc, label = "validation_loss", color = "g")
    plt.legend(loc = "lower right")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy curve")
    plt.grid()
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
    Acc_name = "Loss.png"

    acc_save_path = os.path.join(save_path, Acc_name)
    plt.savefig(acc_save_path)
    print("Save loss curve")

def CM(label_idx, output_idx, CM_path, classes):
    label_flatten = list(chain(*label_idx))# 展平成一維列表
    pred_flatten = list(chain(*output_idx))
    plt.figure(figsize = (10, 8))# 創一個新圖形

    confusion = confusion_matrix(label_flatten, pred_flatten)# 套函式
    # sns用來對confusion matrix進行處理
    # annot: 顯示數值
    # fmt: 一般方式顯示數值
    sns.heatmap(confusion, annot = True, fmt = 'g', cmap = "Reds", xticklabels = classes, yticklabels = classes)
    # 決定X, Y軸是是預測的還是真實標籤
    plt.xlabel("Predicted")
    plt.ylabel("Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()  # 自動調整布局
    plt.savefig(CM_path, bbox_inches='tight')  # 保存時包括圖邊界
    plt.savefig(CM_path)

