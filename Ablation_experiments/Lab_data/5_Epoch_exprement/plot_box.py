import os
import matplotlib.pyplot as plt
import numpy as np  
import seaborn as sns

epochs = ['50', '100', '150', '200']
data_txt = "Test_acc.txt"
save_data = 'data.pkl'
vector = {}

for idx in epochs:
    opt_file = f'./result/{idx}'
    vector.setdefault(idx, [])
    for time in os.listdir(opt_file):
        time_file = os.path.join(opt_file, time)

        if not os.path.isdir(time_file):
            continue  # 如果不是資料夾，則跳過這次迭代 

        time_path = os.path.join(opt_file, time, data_txt)
        with open(time_path, 'r') as f:
            acc = f.readline().strip().split()[-1]
            vector[idx].append(float(acc))

data = [vector[idx] for idx in epochs]
# 繪製箱型圖
colors = sns.color_palette("Set3", 10)  # 一次拿 10 種顏色
plt.figure(figsize=(9, 6))
box = plt.boxplot(data, patch_artist=True, labels=epochs)
plt.boxplot(data, labels=epochs)
# 給每個 box 上不同顏色
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7) # 調整透明度

for flier in box['fliers']:
    flier.set(marker='o', color='black', alpha=0.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Comparison of test accuracy for epochs', fontsize=18)
plt.ylabel('Test accuracy', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0.9, 1.0)
plt.savefig('boxplot.png', dpi=300)
