import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallModelWrapper(nn.Module):
    def __init__(self, input_dim, output_dim, model, objective, **kwargs):
        super(SmallModelWrapper, self).__init__()
        self.model = eval(model['name'])(input_dim, output_dim, **model)
        self.objective = eval(objective['name'])(**objective)

    def forward(self, feats_inp, linears_inp, linears_tar, mask_label):
        predicted, _ = self.model(features=feats_inp, linears=linears_inp)
        loss, _ = self.objective(predicted, linears_tar, mask_label[:, :, 0])
        return loss, predicted


class L1(nn.Module):
    def __init__(self, log=False, eps=1e-10, **kwargs):
        super().__init__()
        self.log = log
        self.eps = eps
        self.fn = torch.nn.L1Loss()

    def forward(self, predicted, linear_tar, stft_length_masks, **kwargs):
        # stft_length_masks: (batch_size, max_time)
        # predicted, linear_tar: (batch_size, max_time, feat_dim)

        src = predicted * stft_length_masks.unsqueeze(-1)
        tar = linear_tar * stft_length_masks.unsqueeze(-1)

        if self.log:
            l1 = self.fn((src + self.eps).log(), (tar + self.eps).log())
        else:
            l1 = self.fn(src, tar)
        
        return l1, {}


class SISDR(nn.Module):
    def __init__(self, eps=1e-10, **kwargs):
        super().__init__()
        self.eps = eps

    def forward(self, predicted, linear_tar, stft_length_masks, **kwargs):
        # stft_length_masks: (batch_size, max_time)
        # predicted, linear_tar: (batch_size, max_time, feat_dim)
        src = F.relu(predicted).pow(0.5) * stft_length_masks.unsqueeze(-1)
        tar = F.relu(linear_tar).pow(0.5) * stft_length_masks.unsqueeze(-1)

        src = src.flatten(start_dim=1).contiguous()
        tar = tar.flatten(start_dim=1).contiguous()

        alpha = torch.sum(src * tar, dim=1) / (torch.sum(tar * tar, dim=1) + self.eps)
        ay = alpha.unsqueeze(1) * tar
        norm = torch.sum((ay - src) * (ay - src), dim=1) + self.eps
        loss = -10 * torch.log10(torch.sum(ay * ay, dim=1) / norm + self.eps)
        
        return loss.mean(), {}


class Residual(nn.Module):
    def __init__(self, input_size=201, output_size=201, hidden_size=201, num_layers=3, bidirectional=False,
                 activation='Sigmoid', cmvn=False, eps=1e-6, **kwargs):
        super(Residual, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.scaling_layer = nn.Sequential(
            nn.Linear(max(1, int(bidirectional) * 2) * hidden_size, output_size), eval(f'nn.{activation}()'))
        self.init_weights()
        self.bidirectional = bidirectional
        self.cmvn = cmvn
        self.eps = eps

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'scaling_layer.0.weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, features, linears, **kwargs):
        offset, _ = self.lstm(features)
        if self.cmvn:
            offset = (offset - offset.mean(dim=1, keepdim=True)) / (offset.std(dim=1, keepdim=True) + self.eps)
        offset = self.scaling_layer(offset)
        predicted = linears * offset
        return predicted, {'offset': offset}


class LSTM(nn.Module):
    def __init__(self, input_size=201, output_size=201, hidden_size=201, num_layers=3, bidirectional=False,
                 activation='ReLU', **kwargs):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.scaling_layer = nn.Sequential(
            nn.Linear(max(1, int(bidirectional) * 2) * hidden_size, output_size), eval(f'nn.{activation}()'))
        self.init_weights()
        self.bidirectional = bidirectional

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'scaling_layer.0.weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, features, **kwargs):
        predicted, _ = self.lstm(features)
        predicted = self.scaling_layer(predicted)
        return predicted, {}