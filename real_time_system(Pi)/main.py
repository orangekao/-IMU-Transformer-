from bleak import BleakClient, BleakScanner
import serial
import subprocess
import time
import asyncio
import numpy as np
import torch
import json
import os
stop_flag = False 

classes = ['sitting', 'fall', 'sit_down', 'stand_up', 'walking', 'walk_stairs', 'push_up', 'jumping']

# xsens
MAC = 'D4:22:CD:00:38:5B'

# Get data uuid from Xsens
UUID = '15172003-4947-11e9-8646-d663bd873d93'

# write instruction uuid to Xsens
Xsen_button = '15172001-4947-11e9-8646-d663bd873d93'

# settings
channel = 3
get_data = np.empty((0, channel))
slide = 108
step_ratio = 1
step = int(slide * step_ratio)
first = True
all_data = None

# load model
model = torch.load('best_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def ble_connect():
    print('Wait for Device connect...')
    rfcomm_proc = subprocess.Popen(["sudo", "rfcomm", "watch", "hci0"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True)
    while not os.path.exists("/dev/rfcomm0"):
        time.sleep(0.5)
    print('Device connect!\n')
    ser = serial.Serial('/dev/rfcomm0', baudrate=9600, timeout=1)

    return ser



def first_process(): 
    global get_data, all_data, first
    all_data = np.expand_dims(get_data, axis = 0)
    inference(all_data)
    all_data = all_data[:, step:, :]
    get_data = np.empty((0, channel))
    first = False

def after_first_process():
    global get_data, all_data
    get_data = np.expand_dims(get_data, axis = 0)
    all_data = np.concatenate([all_data, get_data], axis = 1)
    inference(all_data)
    get_data = np.empty([0, channel])
    all_data = all_data[:, step:, :]


def encoder_data(bytes_):
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
    numpy_data = np.frombuffer(bytes_, dtype=data_segments)
    return numpy_data


def Xsens_data_process(sender, data):
    global get_data, slide, all_data, first, step
    record = []
    numpy_data = encoder_data(data)
    for name in numpy_data.dtype.names:
        if 'FreeAcc' in name:
            record.append(numpy_data[name])
    # numpy struct to numpy and [c, t]->[t, c]
    record = np.asarray(record).transpose(1, 0)

    get_data = np.concatenate([record, get_data], axis = 0)
    if first == True:
        if len(get_data) == slide:
            first_process()
    else:
        if len(get_data) == step:
            after_first_process()

def inference(t_data):
    global ser
    with torch.no_grad():

        f_input = np.abs(np.fft.fft(t_data, axis = 1))
        f_input = torch.tensor(f_input).to(device)
        t_input = torch.tensor(t_data).to(device)

        output_tensor = model(t_input.float(), f_input.float())
        _, output = torch.max(output_tensor, dim = 1)

        print(classes[output])

        prob_list = output_tensor[0].tolist()
        signal = t_data.squeeze(0).tolist()

        all_output_data = {
        'inference': classes[output],
        'probability': prob_list,
        'data':signal
        }

        message = json.dumps(all_output_data) + '\n'
        try:
            ser.write(message.encode('utf-8'))
            time.sleep(0.01)
        except Exception as e:
            print('Error')





async def main(MAC):
    global stop_flag

    device = await BleakScanner.find_device_by_address(MAC, timeout=20.0)
    if device is None:
        print('Device is not found')
    else:
        print(f'Find device : {MAC}')
        async with BleakClient(device) as client:
            # choose the button
            operate_button = b'\x01\x01' + [22, b'\x16'][1]

            # press the button
            await client.write_gatt_char(Xsen_button, operate_button, True)

            # Get subscribe specify characteristic and processing data
            await client.start_notify(UUID, Xsens_data_process)

            # if stop_flag not true program would not stop
            while not stop_flag:
                await asyncio.sleep(0.01)
            
            await client.stop_notify(UUID)
            await client.disconnect()


# try:
    # ser = ble_connect()
    asyncio.run(main(MAC))
    # ser.flush()
# except Exception as e:
    # print(f"Failed to connect: {e}")
