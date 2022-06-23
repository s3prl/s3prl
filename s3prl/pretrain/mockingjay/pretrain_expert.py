# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ pretrain/mockingjay/pretrain_expert.py ]
#   Synopsis     [ the mockingjay pretrain expert ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import yaml
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#-------------#
from pretrain.mockingjay.dataset import KaldiAcousticDataset, OnlineAcousticDataset
from upstream.mockingjay.model import TransformerConfig, TransformerInitModel
from upstream.mockingjay.model import TransformerSpecPredictionHead, TransformerModel
from upstream.baseline.extracter import get_extracter
from upstream.baseline.preprocessor import get_preprocessor
from utility.audio import plot_spectrogram_to_numpy


####################
# UPSTREAM WRAPPER #
####################
class UpstreamPretrainExpert(nn.Module):
    """
    The Mockingjay pretrain expert
    """

    def __init__(self, datarc, upstream_config, device='cuda', multi_gpu=False, **kwargs):
        super(UpstreamPretrainExpert, self).__init__()

        self.datarc = datarc
        self.device = device
        self.multi_gpu = multi_gpu

        if type(upstream_config) == str:
            self.upstream_config = yaml.load(open(upstream_config, 'r'), Loader=yaml.FullLoader)
            print('[UpstreamPretrainExpert] - Using upstream config from:', upstream_config)
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            print('[UpstreamPretrainExpert] - Using upstream config from the previous experiment.')
        else:
            raise ValueError
        
        if 'libri_root' in self.datarc and 'kaldi' in self.upstream_config['audio']:
            print('[UpstreamPretrainExpert] - Using kaldi feature extracter, on-the-fly feature extraction')
            extracter, input_dim, _ = get_extracter(self.upstream_config['audio'])
            output_dim = None
        elif 'libri_root' in self.datarc:
            print('[UpstreamPretrainExpert] - Using online preprocessor, on-the-fly feature extraction')
            extracter, input_dim, output_dim = get_preprocessor(self.upstream_config['audio'])
        else:
            print('[UpstreamPretrainExpert] - Using features pre-extracted and saved')
            extracter, input_dim = None, self.upstream_config['transformer']['input_dim']
            output_dim = None
        print('[UpstreamPretrainExpert] - Input dim:', input_dim)

        self._get_train_dataloader(extracter)

        print('[UpstreamPretrainExpert] - Initializing model...')
        model_config = TransformerConfig(self.upstream_config['transformer'])
        setattr(model_config, 'loss', self.upstream_config['task']['loss'])
        self.model = TransformerForMaskedAcousticModel(model_config, input_dim, output_dim=output_dim)

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print('[UpstreamPretrainExpert] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[UpstreamPretrainExpert] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def _get_train_dataloader(self, extracter):
        if 'libri_root' in self.datarc and 'kaldi' not in self.upstream_config['audio']:
            dataset = OnlineAcousticDataset(extracter,
                                            self.upstream_config['task'],
                                            self.datarc['train_batch_size'],
                                            target_level=self.upstream_config['audio']['target_level'],
                                            **self.datarc)
        else:
            dataset = KaldiAcousticDataset(extracter,
                                           self.upstream_config['task'],
                                           self.datarc['train_batch_size'],
                                           **self.datarc)
        self.dataloader = DataLoader(dataset, batch_size=1, # for bucketing
                                     shuffle=True, num_workers=self.datarc['num_workers'],
                                     drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn)

    # Interface
    def load_model(self, init_ckpt):
        assert 'Transformer' in init_ckpt and 'SpecHead' in init_ckpt
        if self.multi_gpu:
            self.model.module.Transformer.load_state_dict(init_ckpt['Transformer'])
            self.model.module.SpecHead.load_state_dict(init_ckpt['SpecHead'])
        else:
            self.model.Transformer.load_state_dict(init_ckpt['Transformer'])
            self.model.SpecHead.load_state_dict(init_ckpt['SpecHead'])

    # Interface
    def loss_to_device(self):
        self.model.loss.to(self.device) if not self.multi_gpu else self.model.module.loss.to(self.device)

    # Interface
    def add_state_to_save(self, all_states):
        all_states['SpecHead'] = self.model.SpecHead.state_dict() if not self.multi_gpu else \
                                 self.model.module.SpecHead.state_dict()
        all_states['Transformer'] = self.model.Transformer.state_dict() if not self.multi_gpu else \
                                    self.model.module.Transformer.state_dict()
        all_states['Upstream_Config'] = self.upstream_config
        return all_states

    # Interface
    def get_train_dataloader(self):
        return self.dataloader

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs):
        """
        Args:
            data:
                [spec_masked, pos_enc, mask_label, attn_mask, spec_target]
            
            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss        
        """

        spec_masked, pos_enc, mask_label, attn_mask, spec_target = data[0], data[1], data[2], data[3], data[4]
        spec_masked = spec_masked.to(self.device)
        
        if pos_enc.dim() == 3:
            # pos_enc: (batch_size, seq_len, hidden_size)
            # GPU memory need (batch_size * seq_len * hidden_size)
            pos_enc = pos_enc.to(self.device)
        elif pos_enc.dim() == 2:
            # pos_enc: (seq_len, hidden_size)
            # GPU memory only need (seq_len * hidden_size) even after expanded
            pos_enc = pos_enc.to(self.device).expand(spec_masked.size(0), *pos_enc.size())

        mask_label = mask_label.to(self.device)
        attn_mask = attn_mask.to(self.device)
        spec_target = spec_target.to(self.device)
        
        loss, pred_spec = self.model(spec_masked, pos_enc, mask_label, attn_mask, spec_target)

        if global_step % log_step == 0:
            spec_list = [spec_masked, pred_spec, spec_target]
            name_list = ['mask_spec', 'pred_spec', 'true_spec']
            
            for i in range(len(spec_list)):
                spec = plot_spectrogram_to_numpy(spec_list[i][0].data.cpu().numpy())
                records[name_list[i]] = spec
            
        return loss, records

    # interface
    def on_before_zero_grad(self):
        pass
    
    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        for key, values in records.items():
            logger.add_image(
                f'{prefix}{key}',
                values,
                global_step=global_step
            )


class TransformerForMaskedAcousticModel(TransformerInitModel):
    """
    Transformer model with the masked acoustic modeling head.
    This module comprises the Transformer model followed by the masked acoustic modeling head.

        Params:
            `config`: a TransformerConfig class instance with the configuration to build a new model
            `intput_dim`: int,  input dimension of model
            `output_dim`: int,  output dimension of model
            `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
            `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
                This can be used to compute head importance metrics. Default: False

        Inputs:
            `spec_input`: a torch.LongTensor of shape [batch_size, sequence_length, feature_dimension]
                with the selected frames processed as masked frames during training,
                generated by the `process_train_MAM_data()` function in `transformer/mam.py`.
            `pos_enc`: a torch.LongTensor of shape [batch_size, sequence_length, hidden_size],
                generated by the `fast_position_encoding()` function in `transformer/mam.py`.
            `masked_label`: masked acoustic modeling labels - torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [1, 0]. All labels set to -1 are ignored, the loss
                is only computed for the labels set to 1.
            `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
                selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
                input sequence length in the current batch. It's the mask that we typically use for attention when
                a batch has varying length sentences.
            `spce_label`: a torch.LongTensor of shape [batch_size, sequence_length, feature_dimension]
                which are the ground truth spectrogram used as reconstruction labels.
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.

        Outputs:
            if `spec_label` and `mask_label` is not `None`:
                Outputs the masked acoustic modeling loss and predicted spectrogram.
            if `spec_label` and `mask_label` is `None`:
                Outputs the masked acoustic modeling predicted spectrogram of shape [batch_size, sequence_length, output_dim].

    Example usage:

        ```python
        spec_input = torch.LongTensor(spec_frames)
        pos_enc = torch.LongTensor(position_encoding(seq_len=len(spec_frames)))

        config = TransformerConfig(config)

        model = TransformerForMaskedAcousticModel(config)
        masked_spec_logits = model(spec_input, pos_enc)
        ```
    """
    def __init__(self, config, input_dim, output_dim, output_attentions=False, keep_multihead_output=False):
        super(TransformerForMaskedAcousticModel, self).__init__(config, output_attentions)
        self.Transformer = TransformerModel(config, input_dim, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.SpecHead = TransformerSpecPredictionHead(config, output_dim if output_dim is not None else input_dim)
        self.apply(self.init_Transformer_weights)
        loss = { 'L1' : nn.L1Loss(),
                 'MSE' : nn.MSELoss() }
        self.loss = loss[config.loss] if hasattr(config, 'loss') else loss['L1']

    def forward(self, spec_input, pos_enc, mask_label=None, attention_mask=None, spec_label=None, head_mask=None):
        outputs = self.Transformer(spec_input, pos_enc, attention_mask,
                            output_all_encoded_layers=False,
                            head_mask=head_mask)
        if self.output_attentions:
            all_attentions, sequence_output = outputs
        else:
            sequence_output = outputs
        pred_spec, pred_state = self.SpecHead(sequence_output)

        if spec_label is not None and mask_label is not None:
            assert mask_label.sum() > 0, 'Without any masking, loss might go NaN. Check your pretrain data processing (s3prl/pretrain/mockingjay/task.py)'
            masked_spec_loss = self.loss(pred_spec.masked_select(mask_label), spec_label.masked_select(mask_label))
            return masked_spec_loss, pred_spec
        elif self.output_attentions:
            return all_attentions, pred_spec
        return pred_spec, pred_state
