import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import pickle
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
# import h5py
# from torchsummary import summary
#import pytorch_model_summary as pms
# from scipy import ndimage
from tqdm import tqdm
import torch.tensor as tensor


class conv_block(nn.Module):
    def __init__(self, t_size, n_step):
        super(conv_block, self).__init__()
        self.cov1d_1 = nn.Conv1d(t_size, t_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.cov1d_2 = nn.Conv1d(t_size, t_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.cov1d_3 = nn.Conv1d(t_size, t_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.attention_1 = nn.MultiheadAttention(n_step, 1)
        self.attention_2 = nn.MultiheadAttention(n_step, 1)
        self.attention_3 = nn.MultiheadAttention(n_step, 1)
        self.norm_1 = nn.LayerNorm([n_step])
        self.norm_2 = nn.LayerNorm([n_step])
        self.norm_3 = nn.LayerNorm([n_step])
        self.drop = nn.Dropout(p=0.2)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu_3 = nn.ReLU()
        self.relu_4 = nn.ReLU()

    def forward(self, x):
        x_input = x
        x_k_v = x_input.transpose(0, 1)
        x = self.cov1d_1(x)
        x = self.norm_1(x)
        x = self.relu_1(x)
        x = x.transpose(0, 1)
        # implement self-attention layer
        x, _ = self.attention_1(x, x_k_v, x_k_v)
        x = x.transpose(0, 1)

        x = self.cov1d_2(x)
        x = self.norm_2(x)
        x = self.relu_2(x)
        x = x.transpose(0, 1)
        # implement self-attention layer
        x, _ = self.attention_2(x, x_k_v, x_k_v)
        x = x.transpose(0, 1)

        x = self.cov1d_3(x)
        x = self.norm_3(x)
        x = self.relu_3(x)
        x = x.transpose(0, 1)
        # implement self-attention layer
        x, _ = self.attention_3(x, x_k_v, x_k_v)
        x = x.transpose(0, 1)

        x = self.relu_4(x)
        x = x.add(x_input)
        output = self.drop(x)

        return output


class identity_block(nn.Module):
    def __init__(self, t_size, n_step):
        super(identity_block, self).__init__()
        self.cov1d_1 = nn.Conv1d(t_size, t_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.attention_1 = nn.MultiheadAttention(n_step, 1)
        self.norm_1 = nn.LayerNorm([n_step])
        self.drop = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_input = x
        x_k_v = x_input.transpose(0, 1)
        x = self.cov1d_1(x)
        x = self.norm_1(x)
        x = self.relu(x)
        x = x.transpose(0, 1)
        # implement self-attention layer
        x, _ = self.attention_1(x, x_k_v, x_k_v)
        x = x.transpose(0, 1)
        output = self.drop(x)

        return output


class Encoder(nn.Module):
    def __init__(self, num_k_sparse, n_step, number_of_blocks):
        super(Encoder, self).__init__()
        self.num_k_sparse = num_k_sparse
        self.cov1d = nn.Conv1d(1, 128, 3, stride=1, padding=1, padding_mode='zeros')
        self.cov1d_1 = nn.Conv1d(128, 1, 3, stride=1, padding=1, padding_mode='zeros')
        self.conv_block_1 = conv_block(t_size=128, n_step=96)
        self.conv_block_2 = conv_block(t_size=128, n_step=48)
        self.conv_block_3 = conv_block(t_size=128, n_step=24)
        self.iden_block_1 = identity_block(t_size=128, n_step=96)
        self.iden_block_2 = identity_block(t_size=128, n_step=48)
        self.iden_block_3 = identity_block(t_size=128, n_step=24)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu_3 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(2, stride=2)
        self.maxpool_2 = nn.MaxPool1d(2, stride=2)
        self.maxpool_3 = nn.MaxPool1d(2, stride=2)
        self.lstm = nn.LSTM(12, 8, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(16, 8)

    #        self.sd = nn.Linear(32,8)
    #        self.logspike = nn.Linear(32,8)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cov1d(x)
        x = self.iden_block_1(x)
        x = self.conv_block_1(x)
        x = self.maxpool_1(x)
        x = self.iden_block_2(x)
        x = self.conv_block_2(x)
        x = self.maxpool_2(x)
        x = self.iden_block_3(x)
        x = self.conv_block_3(x)
        x = self.maxpool_3(x)
        x = self.cov1d_1(x)
        encode, (_, __) = self.lstm(x)
        selection = self.dense(encode)
        selection = self.relu_1(selection)
        #        k = self.num_k_sparse
        #        if k < selection.shape[2]:
        #            for raw in selection:
        #                indices = torch.topk(raw, k)[1].cuda()
        #                mask = torch.ones(raw.shape, dtype=bool).cuda()
        #                mask[:,indices] = False
        #                raw[mask] = 0
        selection = selection.transpose(1, 2)
        #         mn = self.mn(encode)
        #  #       mn = self.relu_1(mn)
        #         sd = self.sd(encode)
        #  #       sd = self.relu_2(sd)
        #         std = torch.exp(sd*0.5).cuda()
        #         eps = torch.normal(1,1,size=std.size()).cuda()
        #         gaussian = eps.mul(std).add_(mn).cuda()

        #         log_spike = -F.relu(-self.logspike(encode))
        #         eta = torch.normal(1,1,size=std.size()).cuda()
        #         selection = F.relu(50.0*(eta + log_spike.exp() - 1)).cuda()
        #         selection = selection.mul(gaussian).cuda()
        #         selection = self.relu_3(selection)

        #         x_sample = self.relu_3(x_sample)
        #         x_sample = x_sample.transpose(1,2)

        return selection


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cov1d = nn.Conv1d(128, 1, 3, stride=1, padding=1, padding_mode='zeros')
        self.conv_block_1 = conv_block(t_size=128, n_step=8)
        self.conv_block_2 = conv_block(t_size=128, n_step=16)
        self.conv_block_3 = conv_block(t_size=128, n_step=32)
        self.conv_block_4 = conv_block(t_size=128, n_step=96)
        self.iden_block_1 = identity_block(t_size=128, n_step=8)
        self.iden_block_2 = identity_block(t_size=128, n_step=16)
        self.iden_block_3 = identity_block(t_size=128, n_step=32)
        self.iden_block_4 = identity_block(t_size=128, n_step=96)
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.upsample_3 = nn.Upsample(scale_factor=3, mode='linear', align_corners=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x.repeat([1, 128, 1])
        x = self.iden_block_1(x)
        x = self.conv_block_1(x)
        x = self.upsample_1(x)
        x = self.iden_block_2(x)
        x = self.conv_block_2(x)
        x = self.upsample_2(x)
        x = self.iden_block_3(x)
        x = self.conv_block_3(x)
        x = self.upsample_3(x)
        x = self.iden_block_4(x)
        x = self.conv_block_4(x)
        x = self.cov1d(x)
        x = x.transpose(1, 2)

        return x


class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''

    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        # encode
        embedding = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize

        # decode
        predicted = self.dec(embedding)
        
        return predicted



# modified entropy loss
class Entropy_Loss(nn.Module):
    def __init__(self, entroy_coe):
        super(Entropy_Loss, self).__init__()
        self.coe = entroy_coe

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, x):
        en_loss = self.entropy_loss(x)
        return en_loss

    def entropy_loss(self, embedding):
        N = embedding.shape[1]
        N = torch.tensor(N).type(torch.float32)
        mask = embedding != 0
        mask1 = torch.sum(mask, axis=0)
        mask2 = mask1 != 0
        n = torch.sum(mask2).type(torch.float32)
        loss_min = (N // 2 + 1).lgamma().exp() ** 2 / (N + 1).lgamma().exp()
        loss = (N - n + 1).lgamma().exp() * (n + 1).lgamma().exp() / (N + 1).lgamma().exp()
        loss = loss - loss_min

        return self.coe * loss


class Regularization_1(nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization_1, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)

    #       self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'dec' in name and 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)

