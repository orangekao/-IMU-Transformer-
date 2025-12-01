import torch
import torch.nn as nn
import torch.nn.functional as F

class main_model(nn.Module):
    def __init__(self, data_shape, category, channels = 32, embedding_channels = 64):
        super(main_model, self).__init__()

        self.CNN_space = CNN(in_channels = data_shape[2], out_channels = channels,  stride = 1)
        self.CNN_freq = CNN(in_channels = data_shape[2], out_channels = channels, stride = 1)
        self.MSDA_t = MultiScaleDoubleletAttention()
        self.MSDA_f = MultiScaleDoubleletAttention()
        self.LSTM_t = LSTM(in_channels = channels, out_channels = channels)
        self.LSTM_f = LSTM(in_channels = channels, out_channels = channels)
        self.CrossA_t = CrossAttention(input_size = channels)
        self.CrossA_f = CrossAttention(input_size = channels)
        self.FC = Classification(category = category, in_channels = embedding_channels, sequence = data_shape[1])
        
    def forward(self, t, f):
        t = self.CNN_space(t)
        f = self.CNN_freq(f)

        t = self.MSDA_t(t)
        f = self.MSDA_f(f)

        t = self.LSTM_t(t)
        f = self.LSTM_f(f)

        Cross_t = self.CrossA_t(t, f)
        Cross_f = self.CrossA_f(f, t)
        out = torch.cat((Cross_t, Cross_f), dim = 2)
        out = self.FC(out)
        
        return out

# ===========================TOOL================================#
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = 5, stride = 1, padding = 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool1d(kernel_size = 3 ,stride = stride ,padding = 1)
        self.detect = in_channels
    def forward(self, x):
 
        if self.detect == x.shape[2]:
            x = x.squeeze(dim = 1)
            x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        # x = nn.Dropout(0.3)(x)
        x = self.relu(x)
        x = self.maxpooling(x)

        return x
    
class LSTM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LSTM, self).__init__()
        self.LSTM_1 = nn.LSTM(input_size = in_channels, hidden_size = out_channels, batch_first = True)
        self.LSTM_2 = nn.LSTM(input_size = in_channels, hidden_size = out_channels, batch_first = True)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, last = self.LSTM_1(x)
        # x = nn.Dropout(0.2)(x)
        x, last = self.LSTM_2(x)
        # x = nn.Dropout(0.2)(x)
        x = x.permute(0, 2, 1)
        return x

class Classification(nn.Module):
    def __init__(self, category, in_channels, sequence):
        super(Classification, self).__init__()
        self.linear = nn.Linear(sequence, category)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, x):
        t = torch.mean(x, dim = 2)
        # c = torch.mean(x, dim = 2)
        # tc = torch.cat((t, c), dim = 1)
        t = self.linear(t)
        t = self.softmax(t)

        return t


class MultiScaleDoubleletAttention(nn.Module):
    def __init__(self):
        super(MultiScaleDoubleletAttention, self).__init__()
        self.conv_5 = nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = 5, stride = 1, padding = (5 - 1)//2)
        self.conv_7 = nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = 7, stride = 1, padding = (7 - 1)//2) 
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        alpha_t_5, alpha_t_7 = self.Zpooling(x)
        alpha_c_5, alpha_c_7 = self.Zpooling(x.permute(0, 2, 1))

        alpha_c_5 = alpha_c_5.permute(0, 2, 1)
        alpha_c_7 = alpha_c_7.permute(0, 2, 1)

        x_c_5 = x * alpha_c_5
        x_t_5 = x * alpha_t_5

        x_c_7 = x * alpha_c_7
        x_t_7 = x * alpha_t_7

        x_5 = (x_c_5 + x_t_5) / 2
        x_7 = (x_c_7 + x_t_7) / 2
        x = x_5 + x_7
        return x

    def Zpooling(self, x):
        GAP = torch.mean(x, dim = 1, keepdim = True)
        GMP = torch.max(x, dim = 1, keepdim = True)[0]
        out = torch.cat((GAP, GMP), dim = 1)
        out_5 = self.conv_5(out)
        out_7 = self.conv_7(out)
        out_5 = self.sigmoid(out_5)
        out_7 = self.sigmoid(out_7)
        
        return out_5, out_7

class CrossAttention(nn.Module):
    def __init__(self, input_size):
        super(CrossAttention, self).__init__()
        self.conv_q = nn.Conv1d(in_channels = input_size, out_channels = input_size, kernel_size = 1, stride = 1)
        self.conv_x = nn.Conv1d(in_channels = input_size, out_channels = input_size, kernel_size = 1, stride = 1)
        self.conv_k = nn.Conv1d(in_channels = input_size, out_channels = input_size, kernel_size = 1, stride = 1)
        self.conv_v = nn.Conv1d(in_channels = input_size, out_channels = input_size, kernel_size = 1, stride = 1)
        self.layer_norm = nn.LayerNorm(input_size)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x, another):
        q = self.conv_q(another)
        q_x = self.conv_x(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        d_k = q.size(-2)
        attention_source = torch.bmm(q, k.transpose(2, 1)) / int(d_k**0.5)
        attention_source_x = torch.bmm(q_x, k.transpose(2, 1)) / int(d_k**0.5)
        attention_weight = self.softmax(attention_source)
        attention_source_x = self.softmax(attention_source_x)
        out_x = torch.bmm(attention_source_x, v)
        out = torch.bmm(attention_weight, v)
        # out = nn.Dropout(0.3)(out)
        # out_x = nn.Dropout(0.3)(out_x)
        out = out_x + out
        out = out.permute(0, 2, 1)
        out = self.layer_norm(out)
        
        return out







