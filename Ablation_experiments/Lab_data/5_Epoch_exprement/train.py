import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import pickle as pkl
from utils import get_data, UCI_HAR_Dataset, store_plot_data, plot_acc_curve, plot_loss_curve
from Models import Model
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# -----Hyperparameters-----
category = 6
epochs = 250
times = 1
batch_size = 8
learning_rate = 5e-3
patience = 10
gamma = 0.1
slide = 64
single_in = False
# -------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--slide", type=int, default=slide, help='slide')
parser.add_argument('--overlap', type=int, default=slide, help='overlap')
parser.add_argument('--category', type=int, default=category, help='category')
parser.add_argument('--epochs', type=int, default=epochs, help='epochs')
parser.add_argument('--times', type=int, default=times, help='times')
parser.add_argument('--batch', type=int, default=batch_size, help='batch')
parser.add_argument('--lr', type=float, default=learning_rate, help='lr')
parser.add_argument('--patience', type=int, default=patience, help='patience')
parser.add_argument('--gamma', type=float, default=gamma, help='gamma')
parser.add_argument('--single_in', type=lambda x: (str(x).lower() == 'true'), default=single_in, help='single_in')
args = parser.parse_args()

print("-----------------manaul_seed--------------" + str(args.times),"\n")
torch.manual_seed(args.times)
variable = args.epochs
print(f"【Slide size】: {args.slide}")
print(f"【Category】: {args.category}")
print(f"【Epochs】: {args.epochs}")
print(f"【Batch size】: {args.batch}")
print(f"【Learning rate】: {args.lr}")
print(f"【Patience】: {args.patience}")
print(f"【Gamma】: {args.gamma}")
print(f"【Single inpute】: {args.single_in}\n")

# 取得資料路徑定義以及儲存資料定義
# 取得
data_file = 'Data_preprocessing'
data_path = f'data.pkl'
fft_data_path = f'data_fft.pkl'
data = os.path.join(data_file, data_path)
fft_data = os.path.join(data_file, fft_data_path)

with open(data, 'rb') as f:
    data = pkl.load(f)
with open(fft_data, 'rb') as f:
    fft_data = pkl.load(f)
# 儲存
save_file = 'result'
save_file_path = f'./{variable}/time_{args.times}'
save_path = os.path.join(save_file, save_file_path)

train_data, train_label, val_data, val_label, _, _ = get_data(data)
fft_train_data, _, fft_val_data, _, _, _ = get_data(fft_data)

# 將資料切成batch size可整除
train_cut = (train_data.shape[0] // args.batch) * args.batch
val_cut = (val_data.shape[0] // args.batch) * args.batch
train_data = train_data[:train_cut, :, :]
train_label = train_label[:train_cut]
val_data = val_data[:val_cut, :, :]
val_label = val_label[:val_cut]
fft_train_data = fft_train_data[:train_cut, :, :]
fft_val_data = fft_val_data[:val_cut, :, :]

# numpy to torch
train_data = torch.from_numpy(train_data[:-1]) # 取頭到倒數第二筆
train_label = torch.from_numpy(train_label[:-1])
val_data = torch.from_numpy(val_data[:-1])
val_label = torch.from_numpy(val_label[:-1])
fft_train_data = torch.from_numpy(fft_train_data[:-1])
fft_val_data = torch.from_numpy(fft_val_data[:-1])

# label 從folat64轉成int64以及 to one hot
train_label = F.one_hot(train_label.long(), num_classes = args.category) # int64
val_label = F.one_hot(val_label.long(), num_classes = args.category)
# print(train_label)

# 把資料跟標籤包在一起並分成batch再打亂
train_dataset = UCI_HAR_Dataset(train_data, fft_train_data, train_label)
val_dataset = UCI_HAR_Dataset(val_data, fft_val_data, val_label)

train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)

# 使用GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 模型
model = Model.main_model(data_shape=train_data.shape, category=args.category)
model.to(device)

# 模型計算參數量
total_parameters = sum(p.numel() for p in model.parameters())
print("Parameters:", total_parameters)

# 損失函數
loss_func = nn.CrossEntropyLoss()

# 優化器
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor=args.gamma, patience=args.patience, verbose=True)

# 創建要儲存的資料夾，路徑為"儲存"所設置的位置
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 初始化
max_acc = 0
train_loss = []
train_acc = []
val_loss = []
val_acc = []

# 訓練
for epoch in range(args.epochs):
    model.train()
    pbar_train = tqdm(train_loader, total=len(train_loader), leave=False)
    label_total = 0
    correct = 0
    loss_storage = []
    for input_t, input_f, label in pbar_train:
        input_t = input_t.to(device)
        input_f = input_f.to(device)
        label = label.to(device)

        "前項傳遞"
        if args.single_in is False:
            output = model(input_t.float(), input_f.float())
        else:
            output = model(input_t.float())# 模型預測

        _, label_idx = torch.max(label, dim=1) #在label的第一個維度取最大值並返回索引
        _, output_idx = torch.max(output, dim=1) # 在output中找最大值索引

        "計算損失函數"
        loss = loss_func(output, label_idx) # 計算損失

        "倒傳遞"
        optimizer.zero_grad()# 梯度清零
        loss.backward()# 反向推導
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #L1正規劃防止梯度爆炸
        optimizer.step() # 更新權重

        pbar_train.set_postfix({'loss': loss.item() })# 終端機右下角更新損失值用
        loss_storage.append(loss.detach().cpu().numpy())# 繪圖用 

        correct += (output_idx == label_idx).sum().item()# 計算成功預測數量 補充(.sum()把bool轉int, .item()合起來該張量為1的數)
        label_total += label.size(0)
    accuracy = correct / label_total
    print(f"Epoch: {epoch + 1}\nLearning rate: {optimizer.param_groups[0]['lr']}\nTraining accuracy: {accuracy}")

    # 繪訓練曲線用
    train_loss.append(np.mean(loss_storage)) 
    train_acc.append(accuracy)

    # 驗證
    model.eval()
    correct = 0
    with torch.no_grad():
        loss_storage = []
        accuracy = 0   
        label_total = 0
        for input_t, input_f, label in val_loader:
            input_t = input_t.to(device)
            input_f = input_f.to(device)
            label = label.to(device)

            if args.single_in is False:
                output = model(input_t.float(), input_f.float())
            else:
                output = model(input_t.float())
            
            _, output_idx = torch.max(output, dim=1)
            label_idx = torch.argmax(label, dim=1)
            loss = loss_func(output, label_idx)
            loss_storage.append(loss.detach().cpu().numpy())
            
            correct += (output_idx == label_idx).sum().item()
            label_total += label.size(0)
        accuracy = correct / label_total
        print(f"Validation accuracy: {accuracy}\nValidation total: {label_total}\nValidation correct: {correct}")

        # 繪驗證曲線用
        val_loss.append(np.mean(loss_storage))
        val_acc.append(accuracy)

        # 存最好的模型權重
        best_model_name = 'best_model.pth'
        model_save_path = os.path.join(save_path, best_model_name)

        if accuracy > max_acc:
            print("-------------------Best model-------------------")
            max_acc = accuracy
            torch.save(model, model_save_path)
        
        scheduler.step(accuracy)

# 繪圖
all_data = store_plot_data(train_acc, train_loss, val_acc, val_loss)
plot_acc_curve(all_data, save_path)
plot_loss_curve(all_data, save_path)









        
        
