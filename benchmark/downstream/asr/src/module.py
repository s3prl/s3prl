import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.autograd import Function

# +

FBANK_SIZE = 40
# -







''' one layer of liGRU using torchscript to accelrate training speed'''
class liGRU_layer(torch.jit.ScriptModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        dropout=0.0,
        nonlinearity="relu",
        bidirectional=True,
        device="cuda",
        do_fusion=False,
        fusion_layer_size=64,
        number_of_mic=1,
        act="relu",
        reduce="mean",
    ):

        super(liGRU_layer, self).__init__()

        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device
        self.do_fusion = do_fusion
        self.fusion_layer_size = fusion_layer_size
        self.number_of_mic = number_of_mic
        self.act = act
        self.reduce = reduce

        if self.do_fusion:
            self.hidden_size = self.fusion_layer_size //  self.number_of_mic

        if self.do_fusion:
            self.wz = FusionLinearConv(
                self.input_size, self.hidden_size, bias=True, number_of_mic = self.number_of_mic, act=self.act, reduce=self.reduce
            ).to(device)

            self.wh = FusionLinearConv(
                self.input_size, self.hidden_size, bias=True, number_of_mic = self.number_of_mic, act=self.act, reduce=self.reduce
            ).to(device)
        else:
            self.wz = nn.Linear(
                self.input_size, self.hidden_size, bias=True
            ).to(device)

            self.wh = nn.Linear(
                self.input_size, self.hidden_size, bias=True
            ).to(device)

            self.wz.bias.data.fill_(0)
            torch.nn.init.xavier_normal_(self.wz.weight.data)
            self.wh.bias.data.fill_(0)
            torch.nn.init.xavier_normal_(self.wh.weight.data)

        self.u = nn.Linear(
            self.hidden_size, 2 * self.hidden_size, bias=False
        ).to(device)

        # Adding orthogonal initialization for recurrent connection
        nn.init.orthogonal_(self.u.weight)

        self.bn_wh = nn.BatchNorm1d(self.hidden_size, momentum=0.05).to(
            device
        )

        self.bn_wz = nn.BatchNorm1d(self.hidden_size, momentum=0.05).to(
            device
        )


        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False).to(device)
        self.drop_mask_te = torch.tensor([1.0], device=device).float()
        self.N_drop_masks = 100
        self.drop_mask_cnt = 0
        # Setting the activation function
        self.act = torch.nn.ReLU().to(device)

    @torch.jit.script_method
    def forward(self, x):
        # type: (Tensor) -> Tensor



        if self.bidirectional:
            x_flip = x.flip(0)
            x = torch.cat([x, x_flip], dim=1)

        # Feed-forward affine transformations (all steps in parallel)
        wz = self.wz(x)
        wh = self.wh(x)

        # Apply batch normalization
        wz_bn = self.bn_wz(wz.view(wz.shape[0] * wz.shape[1], wz.shape[2]))
        wh_bn = self.bn_wh(wh.view(wh.shape[0] * wh.shape[1], wh.shape[2]))

        wz = wz_bn.view(wz.shape[0], wz.shape[1], wz.shape[2])
        wh = wh_bn.view(wh.shape[0], wh.shape[1], wh.shape[2])

        # Processing time steps
        h = self.ligru_cell(wz, wh)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=1)
            h_b = h_b.flip(0)
            h = torch.cat([h_f, h_b], dim=2)



        return h

    @torch.jit.script_method
    def ligru_cell(self, wz, wh):
        # type: (Tensor, Tensor) -> Tensor

        self.batch_size = wh.shape[0]//2
        if self.batch_size % 2 == 0:
            self.b_even = True
        else:
            self.b_even = False

        if self.b_even:
            if self.bidirectional:
                h_init = torch.zeros(
                    2 * self.batch_size,
                    self.hidden_size,
                    device="cuda",
                )
                drop_masks_i = self.drop(
                    torch.ones(
                        self.N_drop_masks,
                        2 * self.batch_size,
                        self.hidden_size,
                        device="cuda",
                    )
                ).data

            else:
                h_init = torch.zeros(
                    self.batch_size,
                    self.hidden_size,
                    device="cuda",
                )
                drop_masks_i = self.drop(
                    torch.ones(
                        self.N_drop_masks,
                        self.batch_size,
                        self.hidden_size,
                        device="cuda",
                    )
                ).data

            hiddens = []
            ht = h_init

            if self.training:

                drop_mask = drop_masks_i[self.drop_mask_cnt]
                self.drop_mask_cnt = self.drop_mask_cnt + 1

                if self.drop_mask_cnt >= self.N_drop_masks:
                    self.drop_mask_cnt = 0
                    if self.bidirectional:
                        drop_masks_i = (
                            self.drop(
                                torch.ones(
                                    self.N_drop_masks,
                                    2 * self.batch_size+1,
                                    self.hidden_size,
                                )
                            )
                            .to(self.device)
                            .data
                        )
                    else:
                        drop_masks_i = (
                            self.drop(
                                torch.ones(
                                    self.N_drop_masks,
                                    self.batch_size,
                                    self.hidden_size,
                                )
                            )
                            .to(self.device)
                            .data
                        )

            else:
                drop_mask = self.drop_mask_te
        else:
            if self.bidirectional:
                h_init = torch.zeros(
                    2 * self.batch_size+1,
                    self.hidden_size,
                    device="cuda",
                )
                drop_masks_i = self.drop(
                    torch.ones(
                        self.N_drop_masks,
                        2 * self.batch_size+1,
                        self.hidden_size,
                        device="cuda",
                    )
                ).data

            else:
                h_init = torch.zeros(
                    self.batch_size,
                    self.hidden_size,
                    device="cuda",
                )
                drop_masks_i = self.drop(
                    torch.ones(
                        self.N_drop_masks,
                        self.batch_size,
                        self.hidden_size,
                        device="cuda",
                    )
                ).data

            hiddens = []
            ht = h_init

            if self.training:

                drop_mask = drop_masks_i[self.drop_mask_cnt]
                self.drop_mask_cnt = self.drop_mask_cnt + 1

                if self.drop_mask_cnt >= self.N_drop_masks:
                    self.drop_mask_cnt = 0
                    if self.bidirectional:
                        drop_masks_i = (
                            self.drop(
                                torch.ones(
                                    self.N_drop_masks,
                                    2 * self.batch_size+1,
                                    self.hidden_size,
                                )
                            )
                            .to(self.device)
                            .data
                        )
                    else:
                        drop_masks_i = (
                            self.drop(
                                torch.ones(
                                    self.N_drop_masks,
                                    self.batch_size,
                                    self.hidden_size,
                                )
                            )
                            .to(self.device)
                            .data
                        )

            else:
                drop_mask = self.drop_mask_te
        #print('wh', wh.shape)
        #print('ht', ht.shape)
        for k in range(wh.shape[1]):

            uz, uh = self.u(ht).chunk(2, 1)
            '''bug fixing'''
            at = wh[:, k, :] + uh # B, T, D
            zt = wz[:, k, :] + uz
            # ligru equation
            zt = torch.sigmoid(zt)

            hcand = self.act(at) * drop_mask
            ht = zt * ht + (1 - zt) * hcand
            hiddens.append(ht)

        # Stacking hidden states
        h = torch.stack(hiddens)


        h = h.permute(1, 0, 2)
        return h


# +

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :, getattr(torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda])().long(), :
    ]
    return x.view(xsize)


# -

def act_fun(act_type):

    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


''' new liGRU '''
class liGRU(nn.Module):
    def __init__(self, inp_dim, ligru_lay, bidirection, dropout, layer_norm, \
    proj=[False, False, False, False], to_do='train'):
        super(liGRU, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.ligru_lay = ligru_lay
        self.ligru_drop = dropout
        self.ligru_use_batchnorm = [True,True,True,True]
        self.ligru_use_laynorm = layer_norm
        self.ligru_use_laynorm_inp = False
        self.ligru_use_batchnorm_inp = False
        self.ligru_orthinit = True
        self.ligru_act = ["relu","relu","relu","relu"]
        self.bidir = bidirection
        self.use_cuda =True
        self.to_do = to_do
        self.proj = proj


        if isinstance(self.ligru_lay, list):
            self.N_ligru_lay = len(self.ligru_lay)
        else:
            self.N_ligru_lay = 1
            self.ligru_use_batchnorm = [False] #[True]
            self.ligru_act = ["relu"]
            self.ligru_lay = [self.ligru_lay]
            self.proj = [False] # for decoder
        


        if self.to_do == "train":
            self.test_flag = False
        else:
            self.test_flag = True

        # List initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        self.wz = nn.ModuleList([])  # Update Gate
        self.uz = nn.ModuleList([])  # Update Gate

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wh = nn.ModuleList([])  # Batch Norm
        self.bn_wz = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.ligru_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.ligru_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)
        
 

        
        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_ligru_lay):

            # Activations
            self.act.append(act_fun(self.ligru_act[i]))

            add_bias = True

            if self.ligru_use_laynorm[i] or self.ligru_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wh.append(nn.Linear(current_input, self.ligru_lay[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.ligru_lay[i], bias=add_bias))

            # Recurrent connections
            self.uh.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i], bias=False))

            if self.ligru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)
            # Glorot init for feedforward weight
            nn.init.xavier_normal_(self.wh[i].weight)
            nn.init.xavier_normal_(self.wz[i].weight)
            
            # batch norm initialization
            self.bn_wh.append(nn.BatchNorm1d(self.ligru_lay[i], momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.ligru_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.ligru_lay[i]))

            if self.bidir:
                current_input = 2 * self.ligru_lay[i]
            else:
                current_input = self.ligru_lay[i]

        self.out_dim = self.ligru_lay[i] + self.bidir * self.ligru_lay[i]
        # for encoder
        self.pj = None
        if self.proj[0]:
            self.pj = nn.Linear(self.out_dim, self.out_dim)


    def forward(self, x, x_len): 
        #print('decoder input shape:', x.shape)
        # Applying Layer/Batch Norm
        if bool(self.ligru_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.ligru_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.N_ligru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.ligru_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.ligru_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == False:
                drop_mask = torch.bernoulli(
                    torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.ligru_drop[i])
                )
            else:
                drop_mask = torch.FloatTensor([1 - self.ligru_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.ligru_use_batchnorm[i]:

                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])

                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] * wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1], wz_out.shape[2])

            # Processing time steps
            hiddens = []
            ht = h_init

            for k in range(x.shape[0]):

                # ligru equation
                zt = torch.sigmoid(wz_out[k] + self.uz[i](ht))
                at = wh_out[k] + self.uh[i](ht)
                hcand = self.act[i](at) * drop_mask
                ht = zt * ht + (1 - zt) * hcand
                #print('ht:', ht)
                #print(ht.shape)
                if self.ligru_use_laynorm[i]:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0 : int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2) : x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h
            if self.proj[0]:
                x = torch.tanh(self.pj(x))
            

        return x, x_len

'''new function for layer normalization'''

class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = self.layer_norm(x)
        return x 

class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)




class VGGExtractor_LN(nn.Module):
    def __init__(self,input_dim):
        super(VGGExtractor_LN, self).__init__()
        self.init_dim = 64
        self.hide_dim = 128
        #print(input_dim)
        in_channel,freq_dim,out_dim = self.check_dim(input_dim)
       
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        LN_dim = FBANK_SIZE
        
        self.extractor = nn.Sequential(
                                nn.Conv2d( in_channel, self.init_dim, 3, stride=1, padding=1),
                                CNNLayerNorm(LN_dim),                              
                                nn.ReLU(),                             
                                nn.Conv2d( self.init_dim, self.init_dim, 3, stride=1, padding=1),
                                CNNLayerNorm(LN_dim),
                                nn.ReLU(),                              
                                nn.MaxPool2d(2, stride=2),  # Half-time dimension      
                                              
                                nn.Conv2d( self.init_dim, self.hide_dim, 3, stride=1, padding=1),
                                CNNLayerNorm(LN_dim//2),
                                nn.ReLU(),                                
                                nn.Conv2d( self.hide_dim, self.hide_dim, 3, stride=1, padding=1),
                                CNNLayerNorm(LN_dim//2),   
                                nn.ReLU(),                               
                                nn.MaxPool2d(2, stride=2), 
                            )

    def check_dim(self,input_dim):
        # Check input dimension, delta feature should be stack over channel. 
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim // 13),13,(13 // 4)*self.hide_dim
        elif input_dim % FBANK_SIZE == 0:
            # Fbank feature
            return int(input_dim // FBANK_SIZE),FBANK_SIZE,(FBANK_SIZE//4)*self.hide_dim
        else:
            raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+d)

    def view_input(self,feature,feat_len):
        # downsample time
        feat_len = feat_len//4
        # crop sequence s.t. t%4==0
        if feature.shape[1]%4 != 0:
            feature = feature[:,:-(feature.shape[1]%4),:].contiguous()
        bs,ts,ds = feature.shape # 8, 1960, 120
        #print('f', feature.shape)
        # stack feature according to result of check_dim
        feature = feature.view(bs,ts,self.in_channel,self.freq_dim)
        feature = feature.transpose(1,2)

        return feature,feat_len

    def forward(self,feature,feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature,feat_len)
        #print(feature.shape)
        # Foward
        feature = self.extractor(feature)
        # BSx128xT/4xD/4 -> BSxT/4x128xD/4
        feature = feature.transpose(1,2)
        #  BS x T/4 x 128 x D/4 -> BS x T/4 x 32D
        feature = feature.contiguous().view(feature.shape[0],feature.shape[1],self.out_dim)
        return feature,feat_len

class VGGExtractor(nn.Module):
    ''' VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf'''
    def __init__(self,input_dim):
        super(VGGExtractor, self).__init__()
        self.init_dim = 128
        self.hide_dim = 256
        in_channel,freq_dim,out_dim = self.check_dim(input_dim)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim

        self.extractor = nn.Sequential(
                                nn.Conv2d( in_channel, self.init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.init_dim, self.init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2, ceil_mode=True), # Half-time dimension
                                nn.Conv2d( self.init_dim, self.hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.hide_dim, self.hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2, ceil_mode=True) # Half-time dimension
                            )

    def check_dim(self,input_dim):
        # Check input dimension, delta feature should be stack over channel. 
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim // 13),13,(13 // 4)*self.hide_dim
        elif input_dim % FBANK_SIZE == 0:
            # Fbank feature
            return int(input_dim // FBANK_SIZE),FBANK_SIZE,(FBANK_SIZE//4)*self.hide_dim
        else:
            raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+d)

    def view_input(self,feature,feat_len):
        # downsample time
        feat_len = feat_len//4 # ?
        # crop sequence s.t. t%4==0
        if feature.shape[1]%4 != 0:
            feature = feature[:,:-(feature.shape[1]%4),:].contiguous()
        bs,ts,ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs,ts,self.in_channel,self.freq_dim)
        feature = feature.transpose(1,2)

        return feature,feat_len

    def forward(self,feature,feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature,feat_len)
        # Foward
        feature = self.extractor(feature)
        # BSx128xT/4xD/4 -> BSxT/4x128xD/4
        feature = feature.transpose(1,2)
        #  BS x T/4 x 128 x D/4 -> BS x T/4 x 32D
        feature = feature.contiguous().view(feature.shape[0],feature.shape[1],self.out_dim)
        return feature,feat_len


class FreqVGGExtractor(nn.Module):
    ''' Frequency Modification VGG extractor for ASR '''
    def __init__(self,input_dim, split_freq, low_dim=4):
        super(FreqVGGExtractor, self).__init__()
        self.split_freq =    split_freq
        self.low_init_dim  = low_dim
        self.low_hide_dim  = low_dim * 2
        self.high_init_dim = 64 - low_dim
        self.high_hide_dim = 128 - low_dim * 2

        in_channel,freq_dim = self.check_dim(input_dim)
        self.in_channel     = in_channel
        self.freq_dim       = freq_dim
        self.low_out_dim    = split_freq // 4 * self.low_hide_dim
        self.high_out_dim   = (freq_dim - split_freq) // 4 * self.high_hide_dim
        self.out_dim        = self.low_out_dim + self.high_out_dim

        self.low_extractor = nn.Sequential(
                                nn.Conv2d( in_channel, self.low_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.low_init_dim, self.low_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2), # Half-time dimension
                                nn.Conv2d( self.low_init_dim, self.low_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.low_hide_dim, self.low_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2) # Half-time dimension
                            )
        self.high_extractor = nn.Sequential(
                                nn.Conv2d( in_channel, self.high_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.high_init_dim, self.high_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2), # Half-time dimension
                                nn.Conv2d( self.high_init_dim, self.high_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.high_hide_dim, self.high_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2) # Half-time dimension
                            )
        
        assert(self.split_freq % 4 == 0)
        assert(self.split_freq > 0 and self.split_freq < self.freq_dim)

    def check_dim(self,input_dim):
        # Check input dimension, delta feature should be stack over channel. 
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim // 13),13
        elif input_dim % FBANK_SIZE == 0:
            # Fbank feature
            return int(input_dim // FBANK_SIZE),FBANK_SIZE
        else:
            raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+d)

    def view_input(self,feature,feat_len):
        # downsample time
        feat_len = feat_len//4
        # crop sequence s.t. t%4==0
        if feature.shape[1]%4 != 0:
            feature = feature[:,:-(feature.shape[1]%4),:].contiguous()
        bs,ts,ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs,ts,self.in_channel,self.freq_dim)
        feature = feature.transpose(1,2)

        return feature,feat_len

    def forward(self,feature,feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature,feat_len)
        # Foward
        low_feature = self.low_extractor(feature[:,:,:,:self.split_freq])
        high_feature = self.high_extractor(feature[:,:,:,self.split_freq:])
        # features : BS x 4 x T/4 x D/4 , BS x 124 x T/4 x D/4
        # BS x H x T/4 x D/4 -> BS x T/4 x H x D/4
        low_feature = low_feature.transpose(1,2)
        high_feature = high_feature.transpose(1,2)
        #  BS x T/4 x H x D/4 -> BS x T/4 x HD/4
        low_feature = low_feature.contiguous().view(low_feature.shape[0],low_feature.shape[1],self.low_out_dim)
        high_feature = high_feature.contiguous().view(high_feature.shape[0],high_feature.shape[1],self.high_out_dim)
        feature = torch.cat((low_feature, high_feature), dim=-1)
        return feature, feat_len

class VGGExtractor2(nn.Module):
    ''' VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf'''
    ''' Only downsample once '''
    def __init__(self,input_dim):
        super(VGGExtractor2, self).__init__()
        self.init_dim = 64
        self.hide_dim = 128
        in_channel,freq_dim,out_dim = self.check_dim(input_dim)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim

        self.extractor = nn.Sequential(
                                nn.Conv2d( in_channel, self.init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.init_dim, self.init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2), # Half-time dimension
                                nn.Conv2d( self.init_dim, self.hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.hide_dim, self.hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d((1, 2), stride=(1, 2)) # 
                            )

    def check_dim(self,input_dim):
        # Check input dimension, delta feature should be stack over channel. 
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim // 13),13,(13 // 4)*self.hide_dim
        elif input_dim % FBANK_SIZE == 0:
            # Fbank feature
            return int(input_dim // FBANK_SIZE),FBANK_SIZE,(FBANK_SIZE//4)*self.hide_dim
        else:
            raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+d)

    def view_input(self,feature,feat_len):
        # downsample time
        feat_len = feat_len//2
        # crop sequence s.t. t%4==0
        if feature.shape[1]%2 != 0:
            feature = feature[:,:-(feature.shape[1]%2),:].contiguous()
        bs,ts,ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs,ts,self.in_channel,self.freq_dim)
        feature = feature.transpose(1,2)

        return feature,feat_len

    def forward(self,feature,feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature,feat_len)
        # Foward
        feature = self.extractor(feature)
        # BSx128xT/2xD/4 -> BSxT/2x128xD/4
        feature = feature.transpose(1,2)
        #  BS x T/2 x 128 x D/4 -> BS x T/2 x 32D
        feature = feature.contiguous().view(feature.shape[0],feature.shape[1],self.out_dim)
        return feature,feat_len

class FreqVGGExtractor2(nn.Module):
    ''' Frequency Modification VGG extractor for ASR '''
    def __init__(self,input_dim, split_freq, low_dim=4):
        super(FreqVGGExtractor2, self).__init__()
        self.split_freq =    split_freq
        self.low_init_dim  = low_dim
        self.low_hide_dim  = low_dim * 2
        self.high_init_dim = 64 - low_dim
        self.high_hide_dim = 128 - low_dim * 2
        # self.init_dim      =  64
        # self.low_hide_dim  =   8
        # self.high_hide_dim = 120

        in_channel,freq_dim = self.check_dim(input_dim)
        self.in_channel     = in_channel
        self.freq_dim       = freq_dim
        self.low_out_dim    = split_freq // 4 * self.low_hide_dim
        self.high_out_dim   = (freq_dim - split_freq) // 4 * self.high_hide_dim
        self.out_dim        = self.low_out_dim + self.high_out_dim

        # self.first_extractor = nn.Sequential(
        #                         nn.Conv2d( in_channel, self.init_dim, 3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.Conv2d( self.init_dim, self.init_dim, 3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.MaxPool2d(2, stride=2), # Half-time dimension
        #                     )
        # self.low_extractor = nn.Sequential(
        #                         nn.Conv2d( self.init_dim, self.low_hide_dim, 3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.Conv2d( self.low_hide_dim, self.low_hide_dim, 3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.MaxPool2d((1, 2), stride=(1, 2)) # 
        #                     )
        # self.high_extractor = nn.Sequential(
        #                         nn.Conv2d( self.init_dim, self.high_hide_dim, 3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.Conv2d( self.high_hide_dim, self.high_hide_dim, 3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.MaxPool2d((1, 2), stride=(1, 2)) # 
        #                     )
        self.low_extractor = nn.Sequential(
                                nn.Conv2d( in_channel, self.low_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.low_init_dim, self.low_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2), # Half-time dimension
                                nn.Conv2d( self.low_init_dim, self.low_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.low_hide_dim, self.low_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d((1, 2), stride=(1, 2)) # 
                            )
        self.high_extractor = nn.Sequential(
                                nn.Conv2d( in_channel, self.high_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.high_init_dim, self.high_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2), # Half-time dimension
                                nn.Conv2d( self.high_init_dim, self.high_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.high_hide_dim, self.high_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d((1, 2), stride=(1, 2)) # 
                            )
        
        assert(self.split_freq % 4 == 0)
        assert(self.split_freq > 0 and self.split_freq < self.freq_dim)

    def check_dim(self,input_dim):
        # Check input dimension, delta feature should be stack over channel. 
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim // 13),13
        elif input_dim % FBANK_SIZE == 0:
            # Fbank feature
            return int(input_dim // FBANK_SIZE),FBANK_SIZE
        else:
            raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+d)

    def view_input(self,feature,feat_len):
        # downsample time
        feat_len = feat_len//2
        # crop sequence s.t. t%4==0
        if feature.shape[1]%2 != 0:
            feature = feature[:,:-(feature.shape[1]%2),:].contiguous()
        bs,ts,ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs,ts,self.in_channel,self.freq_dim)
        feature = feature.transpose(1,2)

        return feature,feat_len

    def forward(self,feature,feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature,feat_len)
        # feature   = self.first_extractor(feature) # new 
        # Foward
        low_feature  = self.low_extractor(feature[:,:,:,:self.split_freq])
        high_feature = self.high_extractor(feature[:,:,:,self.split_freq:])
        # low_feature  = self.low_extractor(feature[:,:,:,:self.split_freq//2])
        # high_feature = self.high_extractor(feature[:,:,:,self.split_freq//2:])
        # features : BS x 4 x T/4 x D/4 , BS x 124 x T/4 x D/4
        # BS x H x T/4 x D/4 -> BS x T/4 x H x D/4
        low_feature = low_feature.transpose(1,2)
        high_feature = high_feature.transpose(1,2)
        #  BS x T/4 x H x D/4 -> BS x T/4 x HD/4
        low_feature = low_feature.contiguous().view(low_feature.shape[0],low_feature.shape[1],self.low_out_dim)
        high_feature = high_feature.contiguous().view(high_feature.shape[0],high_feature.shape[1],self.high_out_dim)
        feature = torch.cat((low_feature, high_feature), dim=-1)
        return feature, feat_len

class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling'''
    def __init__(self, input_dim, module, dim, bidirection, dropout, layer_norm, sample_rate, sample_style, proj, batch_size):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2*dim if bidirection else dim
        self.out_dim = sample_rate*rnn_out_dim if sample_rate>1 and sample_style=='concat' else rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.sample_style = sample_style
        self.proj = proj

        if self.sample_style not in ['drop','concat']:
            raise ValueError('Unsupported Sample Style: '+self.sample_style)
        #print(input_dim) = 160
        #print(dim) = 320

        # Recurrent layer
        if module in ['LSTM','GRU']:
            self.layer = getattr(nn,module.upper())(input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True)
            self.gru = True
        ## get LSTM or GRU
        else: # liGRU
            self.layer = liGRU_layer(input_dim, dim, batch_size, bidirectional=bidirection)
            self.gru = False

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout>0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim,rnn_out_dim)

    def forward(self, input_x , x_len):
        # Forward RNN
        '''before using rnn to acclerate?'''
        #if not self.training:
            #self.layer.flatten_parameters()
        
        # ToDo: check time efficiency of pack/pad
        #input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        if self.gru:
            output,_ = self.layer(input_x)
        else:
            output = self.layer(input_x)
        #print('input:', input_x.shape)
        #print('output:', output.shape)
        #output,x_len = pad_packed_sequence(output,batch_first=True)

        # Normalizations
        if self.layer_norm:
            output = self.ln(output)
        if self.dropout>0:
            output = self.dp(output)

        # Perform Downsampling
        if self.sample_rate > 1:
            batch_size,timestep,feature_dim = output.shape
            '''output is not 1d'''

            x_len = x_len//self.sample_rate

            if self.sample_style =='drop':
                # Drop the unselected timesteps
                output = output[:,::self.sample_rate,:].contiguous()
            else:
                # Drop the redundant frames and concat the rest according to sample rate
                if timestep%self.sample_rate != 0:
                    output = output[:,:-(timestep%self.sample_rate),:]
                output = output.contiguous().view(batch_size,int(timestep/self.sample_rate),feature_dim*self.sample_rate)

        if self.proj:
            output = torch.tanh(self.pj(output)) 

        return output,x_len


class BaseAttention(nn.Module):
    ''' Base module for attentions '''
    def __init__(self, temperature, num_head):
        super().__init__()
        self.temperature = temperature
        self.num_head = num_head
        self.softmax = nn.Softmax(dim=-1)
        self.reset_mem()

    def reset_mem(self):
        # Reset mask
        self.mask = None
        self.k_len = None

    def set_mem(self):
        pass

    def compute_mask(self,k,k_len):
        # Make the mask for padded states
        self.k_len = k_len
        bs,ts,_ = k.shape
        self.mask = np.zeros((bs,self.num_head,ts))
        for idx,sl in enumerate(k_len): # there are "batch" enc_len
            self.mask[idx,:,sl:] = 1 # ToDo: more elegant way? padding spare in the end of the sentence
        self.mask = torch.from_numpy(self.mask).to(k_len.device, dtype=torch.bool).view(-1,ts)# BNxT
    ### important
    def _attend(self, energy, value):
        attn = energy / self.temperature
        attn = attn.masked_fill(self.mask, -np.inf) 
        attn = self.softmax(attn) # BNxT
        output = torch.bmm(attn.unsqueeze(1), value).squeeze(1) # BNxT x BNxTxD-> BNxD
        ## we don't use v in LAS case, v is enc_feature
        ## output is g, to decoder
        return output, attn


class ScaleDotAttention(BaseAttention):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, num_head):
        super().__init__(temperature, num_head)

    def forward(self, q, k, v):
        ts = k.shape[1]
        energy = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1) # BNxD * BNxDxT = BNxT
        output, attn = self._attend(energy,v)
        
        attn = attn.view(-1,self.num_head,ts) # BNxT -> BxNxT

        return output, attn


class LocationAwareAttention(BaseAttention):
    ''' Location-Awared Attention '''
    def __init__(self, kernel_size, kernel_num, dim, num_head, temperature):
        super().__init__(temperature, num_head)
        self.prev_att  = None
        self.loc_conv = nn.Conv1d(num_head, kernel_num, kernel_size=2*kernel_size+1, padding=kernel_size, bias=False)
        self.loc_proj = nn.Linear(kernel_num, dim,bias=False)
        self.gen_energy = nn.Linear(dim, 1) # why output dim is 1?
        self.dim = dim

    def reset_mem(self):
        super().reset_mem()
        self.prev_att = None

    def set_mem(self, prev_att):
        self.prev_att = prev_att

    def forward(self, q, k, v):
        bs_nh,ts,_ = k.shape
        bs = bs_nh//self.num_head

        # Uniformly init prev_att
        if self.prev_att is None:
            self.prev_att = torch.zeros((bs,self.num_head,ts)).to(k.device)
            for idx,sl in enumerate(self.k_len):
                self.prev_att[idx,:,:sl] = 1.0/sl

        # Calculate location context
        loc_context = torch.tanh(self.loc_proj(self.loc_conv(self.prev_att).transpose(1,2))) # BxNxT->BxTxD
        loc_context = loc_context.unsqueeze(1).repeat(1,self.num_head,1,1).view(-1,ts,self.dim)   # BxNxTxD -> BNxTxD
        q = q.unsqueeze(1) # BNx1xD
        
        # Compute energy and context
        energy = self.gen_energy(torch.tanh( k+q+loc_context )).squeeze(2) # BNxTxD -> BNxT 
        output, attn = self._attend(energy,v) # including softmax
        attn = attn.view(bs,self.num_head,ts) # BNxT -> BxNxT
        self.prev_att = attn

        return output, attn
