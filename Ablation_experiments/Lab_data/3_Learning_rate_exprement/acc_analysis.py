import numpy as np
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--slide", type=int, help="slide", default=96)
parser.add_argument("--lr", type=float, help="lr", default=5e-3)
parser.add_argument("--overlap", type=int, help="overlap", default=50)
args = parser.parse_args()

# 超參數實驗改這個
variable  = args.lr

# 匯入資料
test_acc_array = []
acc_max = 0
idx = 0
for i in range(1, 31):

    test_dir = './result'
    test_file_path = f'{variable:.0e}/time_{i}/Test_acc.txt'
    test_acc_path = os.path.join(test_dir, test_file_path)
    
    with open(test_acc_path, 'r') as f:
        data = f.readline().strip()

    accuracy = data.split(":")[1].strip()
    test_acc_array.append(float(accuracy))

    if float(accuracy) > acc_max:
        acc_max = float(accuracy)
        idx = i
# 最小值
min = np.min(test_acc_array)
# 計算標準差
std = np.std(test_acc_array)

# 計算中位數
median = np.median(test_acc_array)

# 計算平均值
mean = np.mean(test_acc_array)

save_path = test_dir + f"/{variable:.0e}" + f"/time_{idx}_best_acc.txt"

with open(save_path, 'w') as f:
    f.write(f"Max: {float(acc_max)}\n")
    f.write(f"Min: {float(min)}\n")
    f.write(f"Std: {float(std)}\n")
    f.write(f"Median: {float(median)}\n")
    f.write(f"Mean: {float(mean)}")

print(f"Time_{idx} is the best")
print(f"Max: {float(acc_max)}")
print(f"Min: {float(min)}")
print(f"Std: {float(std)}")
print(f"Median: {float(median)}")
print(f"Mean: {float(mean)}")
    
