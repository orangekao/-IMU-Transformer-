import numpy as np
import pandas as pd


def get_data(data):
    train_data = data["train"]["data"]
    train_label = data["train"]["label"]
    val_data = data["val"]["data"]
    val_label = data["val"]["label"]
    test_data = data["test"]["data"]
    test_label = data["test"]["label"]

    return train_data, train_label, val_data, val_label, test_data, test_label

def save_data(data, train_data, train_label, val_data, val_label):
    data["train"]["data"] = train_data
    data["train"]["label"] = train_label
    data["val"]["data"] = val_data
    data["val"]["label"] = val_label

    return data

def turn_to_fft(data):
    data["train"]["data"] = np.abs(np.fft.fft(data["train"]["data"], axis=1))
    data["val"]["data"] = np.abs(np.fft.fft(data["val"]["data"], axis=1))
    data["test"]["data"] = np.abs(np.fft.fft(data["test"]["data"], axis=1))

    return data



