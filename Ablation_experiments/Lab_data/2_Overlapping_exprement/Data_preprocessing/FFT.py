from preprocess_utils import get_data, turn_to_fft
import pickle as pkl
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--axis", type=int, help="axis", default=3)
parser.add_argument("--slide", type=int, help="slide", default=96)
parser.add_argument("--overlap", type=int, help="overlap", default=80)
args = parser.parse_args()

# 超參數實驗要改
variable  = args.overlap

data_path = f'./{variable}/data.pkl'
fft_data_path = f'./{variable}/data_fft.pkl'

with open(data_path, "rb") as f:
    data = pkl.load(f)

data = turn_to_fft(data)

with open(fft_data_path, "wb") as f:
    pkl.dump(data, f)

print("Turn to FFT")







