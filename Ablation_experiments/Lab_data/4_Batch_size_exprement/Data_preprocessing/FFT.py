from preprocess_utils import get_data, turn_to_fft
import pickle as pkl
import argparse
import numpy as np

data_path = f'./data.pkl'
fft_data_path = f'./data_fft.pkl'

with open(data_path, "rb") as f:
    data = pkl.load(f)

data = turn_to_fft(data)

with open(fft_data_path, "wb") as f:
    pkl.dump(data, f)

print("Turn to FFT")







