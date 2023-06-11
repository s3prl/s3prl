# coding=utf-8
# Copyright 2022 project FaST-VGS.
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ResDAVEnet code modified from https://github.com/wnhsu/ResDAVEnet-VQ/blob/master/models/AudioModels.py
import math
import numpy as np
import torch
from torch import nn
from .w2v2_model import  Wav2Vec2Model_cls
from .utils import w2v2_loss, Margin_InfoNCE_loss, VisualFeatEncoder, BertLayer, LXMERTXLayer
import logging
logger = logging.getLogger(__name__)

def flatten_tensor(inputs):
    """
    convert [b,t,c] into [b*t,c]
    """
    btc = inputs.shape
    return inputs.reshape(-1, btc[-1]), btc

def unflatten_tensor(inputs, btc):
    """
    Inverse function for flatten_tensor()
    """
    if inputs is None:
        return inputs
    return inputs.view(btc)

def get_flattened_indices(nframes, padded_len):
    indices = []
    for i, nframe in enumerate(nframes):
        indices.append(torch.arange(nframe) + i * padded_len)
    return torch.cat(indices).to(nframes.device)

def get_flattened_indices_clusters(nframes, padded_len):
    indices = []
    for i, nframe in enumerate(nframes):
        indices.append(torch.arange(nframe) + i * padded_len)
    return indices

def conv1d(in_planes, out_planes, width=9, stride=1, bias=False):
    """1xd convolution with padding"""
    if width % 2 == 0:
        pad_amt = int(width / 2)
    else:
        pad_amt = int((width - 1) / 2)
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,width), 
                     stride=stride, padding=(0,pad_amt), bias=bias)
class SpeechBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, width=9, stride=1, downsample=None):
        super(SpeechBasicBlock, self).__init__()
        self.conv1 = conv1d(inplanes, planes, width=width, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d(planes, planes, width=width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResDavenet(nn.Module):
    def __init__(self):
        super(ResDavenet, self).__init__()
        feat_dim=768
        block=SpeechBasicBlock
        layers=[2, 2, 2, 2]
        layer_widths=[128, 128, 256, 512, 768]
        convsize=9
        self.feat_dim = feat_dim
        self.inplanes = layer_widths[0]
        self.linear = nn.Linear(feat_dim, self.inplanes)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, layer_widths[1], layers[0], 
                                       width=convsize, stride=2)
        self.layer2 = self._make_layer(block, layer_widths[2], layers[1], 
                                       width=convsize, stride=2)
        self.layer3 = self._make_layer(block, layer_widths[3], layers[2], 
                                       width=convsize, stride=2)
        self.layer4 = self._make_layer(block, layer_widths[4], layers[3], 
                                       width=convsize, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, width=9, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )       
        layers = []
        layers.append(block(self.inplanes, planes, width=width, stride=stride, 
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, width=width, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x) # [b, t, 768] -> [b, t, 128]
        x = x.transpose(1,2) # [b,t,128]-> [b,128,t]
        x = x.unsqueeze(2).contiguous() # [b, 128, 1, t]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.squeeze(2).transpose(1,2).contiguous()
        return x
    def get_inter(self, x, inter):
        x = self.linear(x) # [b, t, 768] -> [b, t, 128]
        x = x.transpose(1,2) # [b,t,128]-> [b,128,t]
        x = x.unsqueeze(2).contiguous() # [b, 128, 1, t]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        if inter == 0:
            return x.squeeze(2).transpose(1,2).contiguous()
        x = self.layer2(x)
        if inter == 1:
            return x.squeeze(2).transpose(1,2).contiguous()
        x = self.layer3(x)
        if inter == 2:
            return x.squeeze(2).transpose(1,2).contiguous()
        x = self.layer4(x)
        x = x.squeeze(2).transpose(1,2).contiguous()
        return x

class DualEncoder(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--num_attention_heads", type=int, help="number of attention heads for visn transformer and cross-modal transformer", default=12)
        parser.add_argument("--intermediate_size", type=int, help="size of feed forward net dimension in visn transformer and cross-modal transformer", default=3072)
        parser.add_argument("--hidden_size", type=int, help="dimension of transformer feature in visn transformer and cross-modal transformer", default=768)
        parser.add_argument("--hidden_act", type=str, help="activation function of visn transformer and cross-modal transformer", default="gelu")
        parser.add_argument("--hidden_dropout_prob", type=float, help="dropout prob for visn transformer and cross-modal transformer", default=0.1)
        parser.add_argument("--attention_probs_dropout_prob", type=float, help="attention dropout prob for visn transformer and cross-modal transformer", default=0.1)
        parser.add_argument("--max_position_embeddings", type=int, default=512) # not used
        parser.add_argument("--initializer_range", type=float, help="range of linear layers (QKV layers) of visn transformer and cross-modal transformer", default=0.02)
        parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
        parser.add_argument("--xtrm_layers", type=int, help="number of cross-modal layer", default=5)
        parser.add_argument("--trm_layers", type=int, help="number of visn layer (relational transformer)", default=5)
        parser.add_argument("--visual_feat_dim", type=int, help="input visual feature dim (from faster rcnn)", default=2048)
        parser.add_argument("--visual_pos_dim", type=int, help="input visual positional embedding dim (originally the bounding boxes from faster rcnn)", default=4)
        parser.add_argument("--return_attention_weight", action="store_true", default=False, help="return the attention weight of the first layer of the first x_layer, i.e. audio attends to image feats [b,heads,T_audio,T_image]")
        parser.add_argument("--fine_matching_weight", type=float, default=1.0)
        parser.add_argument("--coarse_matching_weight", type=float, default=0.1)
        parser.add_argument("--caption_w2v2_weight", type=float, default=None, help="the weight on w2v2 loss on audios of spokencoco/places/flickr8k")
        parser.add_argument("--libri_w2v2_weight", type=float, default=None)
        parser.add_argument("--coarse_to_fine_retrieve", action="store_true", default=False)
        parser.add_argument("--fb_w2v2_weights_fn", type=str, help="the path of w2v2 small model trained by FAIR", default=None)
        parser.add_argument("--margin", type=float, default=1.0)
        parser.add_argument("--topk", type=float, default=100)
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.caption_w2v2_weight == None:
            self.no_caption_audio_loss = True
        else:
            self.no_caption_audio_loss = False
        
        if self.args.caption_w2v2_weight == None and self.args.libri_w2v2_weight == None:
            args.encoder_layers = args.layer_use + 1
        
        self.conv1_trm1_trm3 = Wav2Vec2Model_cls(args)
        self.conv2 = ResDavenet()
        self.audio_cls_token_proj_coarse = nn.Sequential(nn.Linear(self.args.hidden_size, self.args.hidden_size*2), nn.GELU(), nn.Linear(self.args.hidden_size*2, self.args.hidden_size))
        self.audio_cls_token_proj_pre = nn.Sequential(nn.Linear(self.args.hidden_size, self.args.hidden_size*2), nn.GELU(), nn.Linear(self.args.hidden_size*2, self.args.hidden_size))
        self.trm2 = BertLayer(args)
        self.trm2_proj = nn.Linear(self.args.hidden_size, self.args.hidden_size)

        self.visn_fc = VisualFeatEncoder(args)
        self.visual_cls_token = torch.nn.Parameter(torch.randn((1, 1, args.hidden_size)), requires_grad=True)
        self.trm = nn.ModuleList([BertLayer(args) for _ in range(args.trm_layers)])
        self.visual_cls_token_proj_coarse = nn.Sequential(nn.Linear(self.args.hidden_size, self.args.hidden_size*2), nn.GELU(), nn.Linear(self.args.hidden_size*2, self.args.hidden_size))

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    

    def forward_image(self, visual_feats, visual_pos, visual_attention_mask):
        visual_feats = (visual_feats, visual_pos)
        visual_feats = self.visn_fc(visual_feats)
        visual_feats = torch.cat([self.visual_cls_token.repeat(visual_feats.shape[0],1,1), visual_feats], dim=1)
        for layer_module in self.trm:
            visual_feats = layer_module(visual_feats, None)
        cls_token_coarse = self.visual_cls_token_proj_coarse(visual_feats[:,0])
        return visual_feats, cls_token_coarse

    def forward_audio(self, audio_feats, audio_attention_mask, test=False):
        if test:
            self.conv1_trm1_trm3.eval()
            trm13_out = self.conv1_trm1_trm3(audio_feats, padding_mask=audio_attention_mask, mask=False, features_only=True, tgt_layer=self.args.layer_use)
            losses = {}
        else:
            self.conv1_trm1_trm3.train()
            if self.no_caption_audio_loss:
                trm13_out = self.conv1_trm1_trm3(audio_feats, padding_mask=audio_attention_mask, mask=False, features_only=True, tgt_layer=self.args.layer_use)
                losses = {}
            else:
                trm13_out = self.conv1_trm1_trm3(audio_feats, padding_mask=audio_attention_mask, mask=True)
                losses = w2v2_loss(self.conv1_trm1_trm3, trm13_out, self.args, suffix="caption")
        non_padding_mask = ~trm13_out['padding_mask']
        w2v2_nframes = non_padding_mask.int().sum(-1)
        audio_feats = self.conv2(trm13_out['layer_feats'])
        pooling_ratio = round(trm13_out['layer_feats'].shape[1] / audio_feats.shape[1])
        nframes = torch.div(w2v2_nframes, pooling_ratio).to(w2v2_nframes.dtype)
        attention_mask = torch.arange(len(audio_feats[0])).unsqueeze(0).to(audio_feats.device) >= nframes.unsqueeze(1)
        
        cls_token_coarse = self.audio_cls_token_proj_coarse(trm13_out['cls_token'])
        cls_token = self.audio_cls_token_proj_pre(trm13_out['cls_token'])
        audio_feats = torch.cat([cls_token.unsqueeze(1), audio_feats],dim=1)
        cls_token_padding_mask = torch.zeros((attention_mask.shape[0],1)).to(attention_mask)
        attention_mask = torch.cat([cls_token_padding_mask, attention_mask], dim=1)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) #[batch_size, 1, 1, to_seq_length]
        # this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        extended_audio_attention_mask = extended_attention_mask * -10000.0 # 0.0 is what we want to attend, -10000. is what we don't want to attend
        audio_feats = self.trm2(audio_feats, extended_audio_attention_mask)
        cls_token_coarse = self.trm2_proj(audio_feats[:,0])
        return audio_feats, cls_token_coarse, extended_audio_attention_mask, losses

    def forward_libri(self, audio_feats, audio_attention_mask):
        trm13_out = self.conv1_trm1_trm3(audio_feats, padding_mask=audio_attention_mask, mask=True)
        losses = w2v2_loss(self.conv1_trm1_trm3, trm13_out, self.args, suffix="libri")
        return losses

    # @torch.cuda.amp.autocast()
    def forward(
        self,
        audio_feats,
        attention_mask=None,
        visual_feats=None,
        visual_pos=None,
        visual_attention_mask=None, # this is not used, cause we always use all 36 features
        test = False,
        inter = -1,
        forward_libri = False
    ):
        if forward_libri:
            libri_loss = self.forward_libri(audio_feats, attention_mask)
            return libri_loss
        elif test:
            visual_feats, visual_cls = self.forward_image(visual_feats, visual_pos, visual_attention_mask)
            audio_feats, audio_cls, extended_audio_attention_mask, _ = self.forward_audio(audio_feats, attention_mask, test)
            return audio_feats, audio_cls, extended_audio_attention_mask, visual_feats, visual_cls
        else:
            visual_feats, visual_cls= self.forward_image(visual_feats, visual_pos, visual_attention_mask)
            audio_feats, audio_cls, extended_audio_attention_mask, losses = self.forward_audio(audio_feats, attention_mask)
            return audio_feats, audio_cls, extended_audio_attention_mask, visual_feats, visual_cls, losses

    def carefully_load_state_dict(self, states):
        """
        1) Take care of DataParallel/nn.Module state_dict
        2) Show keys that are not loaded due to size mismatch or not found in model
        """
        new_states = self.state_dict()
        loaded_keys = []
        for k, v in states.items():
            k = k[7:] if k.startswith('module') else k
            # if "audio_convnet" in k:
            #     print(f"skip audio convnet weights {k}")
            #     continue
            if k in new_states and new_states[k].size() == v.size():
                new_states[k] = v
                loaded_keys.append(k)
            else:
                print('Ignoring %s due to not existing or size mismatch' % k)

        non_loaded_keys = set(new_states.keys()) - set(loaded_keys)
        if non_loaded_keys:
            print('\nDual Encoder states that do not exist in the seed_dir:')
            for k in sorted(non_loaded_keys):
                print('  %s' % k)
        
        self.load_state_dict(new_states)

class CrossEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.xtrm = nn.ModuleList(
        [LXMERTXLayer(args) for _ in range(args.xtrm_layers)]
    )
        self.fc = nn.Sequential(nn.Linear(self.args.hidden_size*2, self.args.hidden_size), nn.GELU(), nn.Linear(self.args.hidden_size,1))        
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, audio_feats_square, extended_audio_attention_mask_square, visual_feats_square, extended_visual_attention_mask_square=None, return_attention_weight=False):
        for layer_module in self.xtrm:
            audio_feats_square, visual_feats_square = layer_module(
                audio_feats_square, extended_audio_attention_mask_square, visual_feats_square, extended_visual_attention_mask_square
            )
        cls_token = torch.cat([audio_feats_square[:,0],visual_feats_square[:,0]],dim=-1)
        cross_relationship_score_square = self.fc(cls_token)
        return cross_relationship_score_square
    def carefully_load_state_dict(self, states):
        """
        1) Take care of DataParallel/nn.Module state_dict
        2) Show keys that are not loaded due to size mismatch or not found in model
        """
        new_states = self.state_dict()
        loaded_keys = []
        for k, v in states.items():
            k = k[7:] if k.startswith('module') else k
            if k in new_states and new_states[k].size() == v.size():
                new_states[k] = v
                loaded_keys.append(k)
            else:
                print('Ignoring %s due to not existing or size mismatch' % k)

        non_loaded_keys = set(new_states.keys()) - set(loaded_keys)
        if non_loaded_keys:
            print('\nCross Encoder states that do not exist in the seed_dir:')
            for k in sorted(non_loaded_keys):
                print('  %s' % k)
        
        self.load_state_dict(new_states)