import os
import asyncio
import numpy as np
import sys
import torch
from bleak import BleakClient, BleakScanner
import argparse
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import sys
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import signal
import pandas as pd
import time
import pickle as pkl
import cv2  # Import OpenCV
from PIL import Image, ImageTk  # Import PIL for image handling
import customtkinter as ctk

recognition_record, time_record = [], []
# Choice labels
choice_8 = ['sitting', 'fall', 'sit_down', 'stand up', 'walking', 'walking_stairs', 'push_up', 'jumping']
stop_flag = False

cap = cv2.VideoCapture(0)
video_label = None

start_time = None
relative_times = []

# ==========GUI=============
win = tk.Tk()
win.title("即時動作預測")
win.geometry("1500x600") # (寬, 高)
win.configure(bg="lightblue")
# win.minsize(width = 1200, height = 750)
win.attributes("-topmost", 0) # 決定是否視窗在最上層

# 設置列寬度
win.grid_columnconfigure(0, weight=1)
win.grid_columnconfigure(1, weight=1)
win.grid_columnconfigure(2, weight=1)
win.grid_columnconfigure(3, weight=1)

# =========Video label===========
video_canvas = tk.Canvas(win, width=420, height=340, bg="lightblue", highlightthickness=0)
video_canvas.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=1, pady=1)

# ================MAC Address================
MAC_label = tk.Label(win, text="Device:", font=("Arial", 18), bg="lightblue")
MAC_label.place(x=1200, y=390)   # 在第1行第2列

MAC_area = tk.Label(win, font=("Arial", 18), width=10, height=1, bg="lightblue", anchor="w")
MAC_area.place(x=1280, y=390)

# ================UUID Address================
UUID_label = tk.Label(win, text="Mode:", font=("Arial", 18), bg="lightblue")
UUID_label.place(x=1200, y=430)

UUID_area = tk.Label(win, font=("Arial", 18), width=12, height=1, bg="lightblue", anchor="w")
UUID_area.place(x=1265, y=430)

def update_MAC(message):
    MAC_area.config(text=message)

def update_UUID(message):
    UUID_area.config(text=message)

#=============Mode form================
fig = Figure(figsize=(8, 4), dpi=100)
fig.patch.set_facecolor("lightblue")
fig.subplots_adjust(
    left=0.05,   # 圖表距 Figure 左邊界 10%
    right=0.99, # 圖表距 Figure 右邊界 5%
    bottom=0.1, # 圖表距 Figure 底部 10%
    top=0.95    # 圖表距 Figure 頂部 5%
)
ax = fig.add_subplot(111)
ax.set_facecolor("white")
lines = [ax.plot([], [], lw=2, label=label)[0] for label in ['acc-x', 'acc-y', 'acc-z']] 
ax.set_xlim(0, 100)
ax.set_ylim(-1, 1)
# 關閉自動縮放
ax.set_autoscalex_on(False)
ax.set_autoscaley_on(False)
ax.grid()
ax.legend()
ax.legend(loc='upper right')
canvas = FigureCanvasTkAgg(fig, master=win)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0, padx=1, pady=1, sticky="nsew")

def update_plot(data, x_dim=0, continue_running=True):
    if continue_running:
        data = np.squeeze(data, axis=0)
        for i in range(3):  # 假設有三個通道
            lines[i].set_ydata(data[:, i])
            lines[i].set_xdata(np.arange(len(data)))
        ax.set_xlim(0, len(data))
        ax.set_ylim(np.min(data), np.max(data))
        canvas.draw()
    else:
        for i in range(3):  # 假設有三個通道
            lines[i].set_ydata(data[:, i])
            lines[i].set_xdata(x_dim)
        ax.set_xlim(np.min(x_dim), np.max(x_dim))
        ax.set_ylim(np.min(data), np.max(data))
        ax.set_xlabel('Seconds')  # 设置横轴标签为秒
        ax.set_ylabel('acc')
        ax.grid()
        canvas.draw()
        with open('./data.pkl', 'wb') as f:
            pkl.dump(data, f)
            
def update_video():
    global stop_flag
    ret, frame = cap.read()
    if ret:
        # 1) 先縮放影像到 420x340
        frame_resized = cv2.resize(frame, (420, 340))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # 2) 轉成 PIL Image，再做成 ImageTk
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # 3) 在 canvas 上繪製圖像
        #    先刪除舊的圖像 (以 tag=vid_img)
        video_canvas.delete("vid_img")
        #    在 (0, -20) 位置繪製，anchor='nw' 表示圖像左上角對齊這個座標
        video_canvas.create_image(
            0, 20,  # y = -20 往上移 20 像素
            anchor="nw",
            image=imgtk,
            tags="vid_img"
        )

        # 4) 為避免垃圾回收，需要保留這張圖像的參考
        video_canvas.imgtk = imgtk

    if not stop_flag:
        # 使用 after() 讓它不斷更新
        video_canvas.after(10, update_video)
    else:
        cap.release()



def update_plot(data, x_dim = 0, continue_running = True):
    if continue_running:
        data = np.squeeze(data, axis = 0)

        for i in range(3):  # 假設有三個通道
            lines[i].set_ydata(data[:, i])
            lines[i].set_xdata(np.arange(len(data)))
        ax.set_xlim(0, len(data))
        ax.set_ylim(np.min(data), np.max(data))
        canvas.draw()
    else:
        for i in range(3):  # 假設有三個通道
            lines[i].set_ydata(data[:, i])
            lines[i].set_xdata(x_dim)
        ax.set_xlim(np.min(x_dim), np.max(x_dim))
        ax.set_ylim(np.min(data), np.max(data))
        ax.set_xlabel('Seconds')  # 设置横轴标签为秒
        ax.set_ylabel('acc')
        ax.grid()
        canvas.draw()
        with open('./data.pkl', 'wb') as f:
            pkl.dump(data, f)


def store_state(recognition_record, time_record):
    
    # 字典定义
    dictionary = {
        'time': time_record,
        'actions': recognition_record
    }

    # 将字典转换为DataFrame
    df = pd.DataFrame(dictionary)
    df['time'] = pd.to_datetime(df['time'])

    # 存储到Excel
    df.to_excel('output.xlsx', index=False)

# def draw_figure():
#     global collect_for_drawing, relative_times, time_record , recognition_record
#     # x = np.arange(int(collect_for_drawing.shape[0]))
#     x = np.array(relative_times)
#     print('================================================================')
#     print(x.shape)
#     print(collect_for_drawing.shape)
#     print('================================================================')
#     # 绘制每个通道的数据
#     plt.plot(x, collect_for_drawing[:, 0], label='axis-x')
#     plt.plot(x, collect_for_drawing[:, 1], label='axis-y')
#     plt.plot(x, collect_for_drawing[:, 2], label='axis-z')

#     # 添加标题和标签
#     plt.title('History waveform')
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Waveform')
#     # 添加图例
#     plt.legend()

#     # win.after(1000, win.destroy)
#     # 显示图表
#     store_state(recognition_record, time_record)
#     plt.savefig('record.png')
#     plt.close()
#     update_plot(collect_for_drawing, x_dim = x, continue_running = False)
    
def draw_figure():
    global collect_for_drawing, relative_times, time_record, recognition_record
    print(len(collect_for_drawing)) #這是每一筆資料
    print(len(relative_times))

    print(len(time_record))
    print(len(recognition_record)) # 這是預測的資料，重疊率90%幾乎10筆資料預測一次

    min_len = min(len(relative_times), len(collect_for_drawing), len(recognition_record))
    x = np.array(relative_times[:min_len])
    y = collect_for_drawing[:min_len, :]
    recognition_record = recognition_record[:min_len]

    fig, ax = plt.subplots(figsize=(10, 5))

    # 畫三軸
    axis = ["x", "y", "z"]
    for i in range(3):
        ax.plot(x, y[:, i], label=f'axis-{axis[i]}')
    ax.set_title('History waveform')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Waveform')
    ax.legend()
    ax.grid()

    # tooltip 初始化
    tooltip = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                          textcoords="offset points",
                          bbox=dict(boxstyle="round", fc="w"),
                          arrowprops=dict(arrowstyle="->"))
    tooltip.set_visible(False)

    # 滑鼠事件：顯示動作名稱
    def on_hover(event):
        if event.inaxes == ax:
            x_val = event.xdata
            if x_val is None:
                return
            index = np.abs(x - x_val).argmin()
            if index < len(recognition_record):
                tooltip.xy = (x[index], y[index, 0])
                tooltip.set_text(f"Action:{recognition_record[index]}\ntime:{time_record[index]}")
                tooltip.set_visible(True)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    # 儲存圖與 Excel
    store_state(recognition_record, time_record)
    plt.savefig('record.png')
    plt.show()

    update_plot(collect_for_drawing, x_dim=x, continue_running=False)    

def stop_program():
    global stop_flag
    stop_flag = True
    update_video() # stop 
    draw_figure()

# ====================================================icon================================================
# 抓圖片
pic_path = "./activity/"
icons_on = []
icons_off = []
height = 120
width = 120

for i in range(1, 9):
    icon_on = ImageTk.PhotoImage(Image.open(pic_path + f"icon{i}_on.png").resize((width, height)))
    icon_off = ImageTk.PhotoImage(Image.open(pic_path + f"icon{i}_off.png").resize((width, height)))
    icons_on.append(icon_on)
    icons_off.append(icon_off)

# print(len(icons_on))
# print(len(icons_off))

# 顯示圖示
frame = tk.Frame(win, bg="lightblue")
frame.grid(row=3, column=0, padx=10, pady=10)

# 建立一個frame放入八個label
labels = []
icon_frame = tk.Frame(win)
icon_frame.place(relx=0.85, rely=0.05, anchor = 'ne')  # 調整這裡的位置參數以適應您的需求
for i in range(len(choice_8)):
    label = tk.Label(frame, image=icons_off[i], bg="lightblue")
    label.pack(side="left", padx=10)
    labels.append(label)



argsparser = argparse.ArgumentParser()
argsparser.add_argument('--xsens_num', type=int, default=1)
args = argsparser.parse_args()

address = None
short_payload_characteristic_uuid = None

if args.xsens_num == 1:
    address = 'D4:22:CD:00:38:5A'
    short_payload_characteristic_uuid = '15172003-4947-11e9-8646-d663bd873d93'
elif args.xsens_num == 2:
    address = 'D4:22:CD:00:38:5B'
    short_payload_characteristic_uuid = '15172003-4947-11e9-8646-d663bd873d93'
else:
    address = 'D4:22:CD:00:38:55'
    short_payload_characteristic_uuid = '15172003-4947-11e9-8646-d663bd873d93'

measurement_characteristic_uuid = '15172001-4947-11e9-8646-d663bd873d93'

# Real-time settings
channel = 3
two_axis = False #XYZ:False XYY:True
slide_length = 108
step_ratio = 0.7
step = int(slide_length * step_ratio)
first = True

torch.manual_seed(0)
get_data = np.empty((0, channel))
all_data = np.empty((0, slide_length, channel))
collect_for_drawing = get_data

# Load model
model = torch.load('best_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


payload_modes = {
    "High Fidelity (with mag)": [1, b'\x01'],
    "Extended (Quaternion)": [2, b'\x02'],
    "Complete (Quaternion)": [3, b'\x03'],
    "Orientation (Euler)": [4, b'\x04'], #ok
    "Orientation (Quaternion)": [5, b'\x05'], 
    "Free acceleration": [6, b'\x06'], #OK
    "Extended (Euler)": [7, b'\x07'],
    "Complete (Euler)": [16, b'\x10'],
    "High Fidelity": [17, b'\x11'],
    "Delta quantities (with mag)": [18, b'\x12'],
    "Delta quantities": [19, b'\x13'],
    "Rate quantities (with mag)": [20, b'\x14'],
    "Rate quantities": [21, b'\x15'],
    "Custom mode 1": [22, b'\x16'],
    "Custom mode 2": [23, b'\x17'],
    "Custom mode 3": [24, b'\x18'],
    "Custom mode 4": [25, b'\x19'],
    "Custom mode 5": [26, b'\x1A'],
}


def real_time(data):
    global recognition_record, time_record
    # update_gui(f'{data.shape}')
    
    with torch.no_grad():
        f_input = np.abs(np.fft.fft(data, axis=1))
        t_input = torch.tensor(data).to(device)
        f_input = torch.tensor(f_input).to(device)
        
        # outputs_T, outputs_F, alpha = model(t_input.float(), f_input.float())
        outputs = model(t_input.float(), f_input.float())
        _, outputs = torch.max(outputs, dim=1)
        print(choice_8[outputs])
        
        # for idx, predict in enumerate(outputs):
        #     # print(idx) 
        #     current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) #current time

        #     recognition_record.append(choice_8[predict]) # 紀錄預測結果
        #     time_record.append(current_time)

        #     update_icons(predict) # 對icon
        for predict in outputs:
            predicted_label = choice_8[predict]
            # 對應當前推論對應的 step 長度資料
            repeated_actions = [predicted_label] * step
            recognition_record.extend(repeated_actions)
            
            # 同步時間紀錄
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(current_time, "\n")
            repeated_times = [current_time] * step
            time_record.extend(repeated_times)
            update_icons(predict)

        if data.shape[0] > 0:
            update_plot(data)  # 假設要顯示第一個通道的數據


# 圖示
def update_icons(predict):
    # i是索引
    for i, label in enumerate(labels):
        if i == predict:
            label.config(image=icons_on[i])
        else:
            label.config(image=icons_off[i])

def first_process():
    global get_data, all_data, first
    # [1, T, C]
    all_data = np.expand_dims(get_data, axis=0)

    # 模型預測
    real_time(all_data)
    all_data = all_data[:, step:, :]
    first = False
    get_data = np.empty([0, channel])

def not_first_process():
    global get_data, all_data, first
    changed_shape_GetData = np.expand_dims(get_data, axis=0)
    all_data = np.concatenate([all_data, changed_shape_GetData], axis=1)
    real_time(all_data)
    get_data = np.empty([0, channel])
    all_data = all_data[:, step:, :]

def encode_free_accel_bytes_to_string(bytes_):
    data_segments = np.dtype([
        ('timestamp', np.uint32),
        ('Euler_X', np.float32),
        ('Euler_Y', np.float32),
        ('Euler_Z', np.float32),
        ('FreeAcc_X', np.float32),
        ('FreeAcc_Y', np.float32),
        ('FreeAcc_Z', np.float32),
        ('Ang_X', np.float32),
        ('Ang_Y', np.float32),
        ('Ang_Z', np.float32)])
    formatted_data = np.frombuffer(bytes_, dtype=data_segments)
    return formatted_data

# data 是 Short Payload內容
def handle_short_payload_notification(sender, data):
    global get_data, all_data, channel, slide_length, step_ratio, first, collect_for_drawing,stop_flag, timestamps, start_time, relative_times
    # 存資料用
    record = []
    # 輸入進來的會是二禁制所以轉乘np數組
    formatted_data = encode_free_accel_bytes_to_string(data)

    for field_name in formatted_data.dtype.names:
        if channel == 3:
            if not two_axis:
                # 執行這個
                if field_name != 'timestamp' and ('Ang' not in field_name) and ('Euler' not in field_name):
                    record.append(formatted_data[field_name])
            else:
                if field_name != 'timestamp' and ('Ang' not in field_name) and ('Euler' not in field_name) and field_name != 'FreeAcc_Z':
                    record.append(formatted_data[field_name])
        elif channel == 6:
            if field_name != 'timestamp' and ('Ang' not in field_name):
                record.append(formatted_data[field_name])
        elif channel == 9:
            if field_name != 'timestamp':
                record.append(formatted_data[field_name])
    if not two_axis:
        
        record = np.asarray(record).transpose(1, 0)
        # [1, 3]
        # print("record:", record.shape)
        # print(record)
    else:
        # 兩軸才要 目的消除Z軸資料 複製Y軸資料
        raw_data = np.asarray(record).transpose(1, 0)
        copy_data = np.expand_dims(raw_data[:,1], axis = 1)
        # print(copy_data.shape)
        # print(raw_data.shape)
        record = np.hstack((raw_data, copy_data))
        # print(record.shape)

    get_data = np.concatenate([record, get_data], axis=0)
    # print(get_data)
    # print(get_data.shape)
    if not stop_flag:
        collect_for_drawing = np.concatenate([collect_for_drawing, record], axis=0)
        if start_time is None:
            start_time = time.time()
        current_time = time.time()
        relative_times.extend([current_time - start_time] * record.shape[0])  # 记录相时间
    
    if first:
        # 長度到了的預處理
        if len(get_data) == slide_length:
            first_process()
    else:
        # 還未到的處理
        if len(get_data) == step:
            not_first_process()


# 跑這個程式
async def main(ble_address):
    global stop_flag
    # update_gui(f'Looking for Bluetooth LE device at address `{ble_address}`...')
    # 有找到指定的MAC就else 沒有找到就if

    # 不使用await就不會等待20秒 直接跳下一行(可以想成await是一個卡點) (因為async會同時執行不同程式，不使用await會阻塞)
    device = await BleakScanner.find_device_by_address(ble_address, timeout=20.0)

    if device is None:
        update_MAC('NO device')
    else:
        if ble_address == 'D4:22:CD:00:38:5A':
            update_MAC('Xsens DOT 1')
        elif ble_address == 'D4:22:CD:00:38:5B':
            update_MAC('Xsens DOT 2')
        else:
            update_MAC('Xsens DOT 3')
        # update_gui(f'Connecting...')

        # BleakClient(device)指定為client
        async with BleakClient(device) as client:
            # 手動連接
            if client.is_connected:
                update_UUID('Custom mode 1')

            # 接收指定特徵(選擇特徵、預處理、模型預測)
            await client.start_notify(short_payload_characteristic_uuid, handle_short_payload_notification)
            # update_gui('Notifications turned on.')

            payload_mode_values = payload_modes["Custom mode 1"]
            payload_mode = payload_mode_values[1]
            measurement_default = b'\x01'
            start_measurement = b'\x01'
            full_turnon_payload = measurement_default + start_measurement + payload_mode
            # update_gui(f'Setting payload with binary: {full_turnon_payload}')
            await client.write_gatt_char(measurement_characteristic_uuid, full_turnon_payload, True)
            # update_gui(f'Streaming turned on.')

            update_video()  # Start video capture and display

            while not stop_flag:
                await asyncio.sleep(0.001)  # Small delay to avoid busy-waiting

            # 預處理結束
            await client.stop_notify(short_payload_characteristic_uuid)
            await client.disconnect()
            # update_gui(f'Streaming turned off.')

            # await asyncio.sleep(100.0)
            # update_gui(f'Streaming turned off.')

        btn_con.config(bg=win.cget('bg'))
            # update_gui(f'Disconnected from `{ble_address}`')

# asyncio可讓程式不只執行單現程，而是多現程
def run_ble_program():
    asyncio.run(main(address))

btn_stop = ctk.CTkButton(
    win,
    text="Stop",
    corner_radius=5,
    width=120,              # 按鈕寬度(px)
    height=50,      
    fg_color="#FF2D2D",
    command=stop_program,
    font=("Arial", 20)        
)
btn_stop.place(x=1330, y=515)

btn_con = ctk.CTkButton(
    win,
    text="Connect",
    corner_radius=5, 
    width=120,              # 按鈕寬度(px)
    height=50,     
    fg_color="green",
    command=lambda: threading.Thread(target=run_ble_program).start(),        
    font=("Arial", 20)
)
btn_con.place(x=1180, y=515)



win.mainloop()

