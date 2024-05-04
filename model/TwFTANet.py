import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from Layers import EncoderLayer
from ftam import FTAM


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # print(x.shape)
        x = x[:, :, :-self.chomp_size, :].contiguous()
        # print(x.shape)
        return x


# Single_Local_SelfAttn_Module
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_kernels, kernel_size, window, n_multiv, stride, dilation, padding,
                 dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.window = window
        self.n_multiv = n_multiv
        self.n_input = n_inputs
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (kernel_size, 1),
                                           stride=stride, padding=(padding, 0), dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.ftam1 = FTAM(n_kernels)
        self.dropout1 = nn.Dropout(dropout)

        self.bn = nn.BatchNorm2d(32)

        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (kernel_size, 1),
                                           stride=stride, padding=(padding, 0), dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.ftam2 = FTAM(n_kernels)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.ftam1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.ftam2, self.dropout2, )
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.ftam3 = FTAM(n_kernels)
        self.ftam4 = FTAM(n_kernels)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.view(-1, self.n_input, self.window, self.n_multiv)
        out = self.net(x)
        out = self.ftam3(out)
        res = x if self.downsample is None else self.downsample(x)
        res = self.ftam4(res)
        result = self.relu(out + res)
        return result


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, d_model, n_kernels, window, n_multiv,
                 d_k, d_v, d_inner, n_layers, n_head, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, n_kernels, kernel_size, window, n_multiv, stride=1,
                                     dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, out_channels)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, x, return_attns=False):
        x = self.network(x)
        x = self.pooling1(x)
        x = torch.squeeze(x, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)
        enc_slf_attn_list = []
        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


# global convolution + self-attenetion module
class Single_Global_SelfAttn_Module(nn.Module):

    def __init__(
            self,
            window, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):
        super(Single_Global_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv2 = nn.Conv2d(1, n_kernels, (window, w_kernel))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, return_attns=False):
        # print('global self_attn:')

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        # print('x:', x.shape)
        x2 = F.relu(self.conv2(x))
        # print('Global C:', x2.shape)
        x2 = nn.Dropout(p=self.drop_prob)(x2)
        x = torch.squeeze(x2, 2)
        x = torch.transpose(x, 1, 2)
        # print('x:', x.shape)
        src_seq = self.in_linear(x)
        # print('src_seq:', src_seq.shape)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


# AR module
class AR(nn.Module):

    def __init__(self, window, output_window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, output_window)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x


class TwFTANet(LightningModule):

    def __init__(self, n_multiv, batch_size, window, output_window, n_kernels, d_model, n_head):
        """
        Pass in parsed HyperOptArgumentParser to the model
        """
        super(TwFTANet, self).__init__()
        self.batch_size = batch_size

        # parameters from dataset
        self.window = window
        self.output_window = output_window
        self.n_multiv = n_multiv
        self.input_dims = n_multiv
        self.n_kernels = n_kernels

        # 不太修改
        self.local = 3
        self.w_kernel = 1

        # hyperparameters of model
        self.d_model = d_model
        self.d_inner = 32
        self.n_layers = 5
        self.n_head = n_head

        # 不太修改
        self.d_k = int(self.d_model / self.n_head)
        self.d_v = int(self.d_model / self.n_head)
        self.drop_prob = 0.1

        # build model
        self.__build_model()

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        """
        self.sgsf = Single_Global_SelfAttn_Module(
            window=self.window, n_multiv=self.n_multiv, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.TCN = TemporalConvNet(
            num_inputs=1, num_channels=[32, 32, 32], d_model=self.d_model, n_kernels=self.n_kernels, window=self.window,
            n_multiv=self.n_multiv, d_k=self.d_k, d_v=self.d_v, d_inner=self.d_inner, n_layers=self.n_layers,
            n_head=self.n_head, kernel_size=3, dropout=self.drop_prob)
        self.ar = AR(window=self.window, output_window=self.output_window)
        self.W_output1 = nn.Linear(2 * self.n_kernels, self.output_window)
        self.dropout = nn.Dropout(p=self.drop_prob)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        """
        sgsf_output, *_ = self.sgsf(x)
        slsf_output, *_ = self.TCN(x)
        sf_output = torch.cat((sgsf_output, slsf_output), 2)

        sf_output = self.dropout(sf_output)
        sf_output = self.W_output1(sf_output)

        sf_output = torch.transpose(sf_output, 1, 2)
        ar_output = self.ar(x)
        output = sf_output + ar_output
        return output
