import os
import torch
import torch.nn.functional as F
import pickle as pkl
import argparse
from utils import get_data, UCI_HAR_Dataset, CM
from torch.utils.data import DataLoader

# -----Hyperparameters-----
batch_size = 16
slide = 64
times = 1
category = 6
single_in = False
# -------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--slide', type=int, default=slide, help='slide')
parser.add_argument('--batch', type=int, default=batch_size, help='batch')
parser.add_argument('--times', type=int, default=times, help='times')
parser.add_argument('--category', type=int, default=category, help='category')
parser.add_argument('--single_in', type=lambda x: (str(x).lower() == 'true'), default=single_in, help='single_in')
args = parser.parse_args()

# 超參數實驗改這個
variable  = args.slide

classes = ['sitting', 'fall_down', 'sit_down', 'stand_up', 'walking', 'walk_stairs', 'push_up', 'jumping']

# 取得測試資料
data_file = 'Data_preprocessing'
data_path = f'{variable}/data.pkl'
fft_data_path = f'{variable}/data_fft.pkl'
data = os.path.join(data_file, data_path)
fft_data = os.path.join(data_file, fft_data_path)

with open(data, 'rb') as f:
    data = pkl.load(f)

with open(fft_data, 'rb') as f:
    fft_data = pkl.load(f)

# 共同路徑
file_path = f'./result/{variable}/time_{args.times}'
# 取得模型權重
model_name = 'best_model.pth'
model_path = os.path.join(file_path, model_name)
model = torch.load(model_path)

# 定義儲存資料位置
# 準確度
test_acc_name = 'Test_acc.txt'
test_acc_path = os.path.join(file_path, test_acc_name)
# 混淆矩陣
CM_name = 'Confusion_Matrix.jpg'
CM_path = os.path.join(file_path, CM_name)

_, _, _, _, test_data, test_label = get_data(data)
_, _, _, _, fft_test_data, _ = get_data(fft_data)

test_data = torch.from_numpy(test_data[:-1])
test_label = torch.from_numpy(test_label[:-1])
fft_test_data = torch.from_numpy(fft_test_data[:-1])

test_label = F.one_hot(test_label.long(), num_classes=args.category)

test_dataset = UCI_HAR_Dataset(test_data, fft_test_data, test_label)
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model.to(device)

total_parameters = sum(p.numel() for p in model.parameters())

correct = 0
label_total = 0
CM_output_idx = []
CM_label_idx = []

model.eval()
for input_t, input_f, label in test_loader:
    torch.manual_seed(args.times)
    input_t = input_t.to(device)
    input_f = input_f.to(device)
    label = label.to(device)

    if args.single_in is False:
        output = model(input_t.float(), input_f.float())
    else:
        output = model(input_t.float())
    
    _, label_idx = torch.max(label, dim=1)
    _, output_idx = torch.max(output, dim=1)

    # 混淆矩陣
    CM_output_idx.append(list(output_idx.cpu().numpy()))
    CM_label_idx.append(list(label_idx.cpu().numpy()))

    correct += (output_idx == label_idx).sum().item()
    label_total += label.size(0)

accuracy = correct / label_total
print(f"Test accuracy: {accuracy}")

CM(CM_label_idx, CM_output_idx, CM_path, classes)
with open(test_acc_path, 'w') as f:
    f.write(f"Test accuracy: {accuracy}\nParameters: {total_parameters}")













