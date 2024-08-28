# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mlm/expert.py ]
#   Synopsis     [ the MLM wrapper ]
#   Author       [ Andy T. Liu (andi611) ]
"""*********************************************************************************************"""

import os
from typing import Dict, List, Union
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedTokenizerFast
from tokenizers import processors
from transformers import RobertaModel

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NUMEXPR_MAX_THREADS'] = '32'
class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str, **kwargs):
        super().__init__()
        self.name = "[MLM UpstreamExpert]"

        # Load the custom tokenizer
        self.tokenizer = self.get_speech_tokenizer(os.path.join(ckpt, "tokenizer-dinosr.json"))
        self.padding_idx = self.tokenizer.pad_token_id

        # Load the model
        self.model = RobertaModel.from_pretrained(ckpt,
                                                  use_safetensors=True,
                                                  add_pooling_layer=False)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}['bfloat16']
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=ptdtype)
        
    def get_speech_tokenizer(self, tokenizer_path):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token ="[CLS]",
            cls_token ="[CLS]",
            pad_token="[PAD]",
            sep_token="[SEP]",
            unk_token="[UNK]",
            mask_token="[MASK]",
            clean_up_tokenization_spaces=True,
        )
        tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.cls_token_id),
                ("[SEP]", tokenizer.sep_token_id),
            ],
        )
        return tokenizer

    def get_downsample_rates(self, key: str) -> int:
        """
        Since the downsample rate is not a fixed constant,
        we set the downsample rate as None,
        and hack the upstream interface to accomodate padding.
        """
        return None

    def forward(self, wavs: List[str]) -> Dict[str, Union[Tensor, List[Tensor]]]:

        batch_encoded = self.tokenizer(
            wavs,
            add_special_tokens=True,
            padding='longest',
            truncation=False,
            return_tensors='pt',
            return_attention_mask=True,
            return_length=False,
            )
        input_ids = batch_encoded['input_ids'].to(self.device)
        attention_mask = batch_encoded['attention_mask'].to(self.device)

        with self.ctx:
            hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
                ).hidden_states

            # Expand the attention mask to match the hidden states dimensions
            expanded_attention_mask = attention_mask.unsqueeze(-1)
            # Apply the mask to each layer's hidden states
            for i in range(len(hidden_states)):
                hidden_states[i].mul_(expanded_attention_mask)

        # The "hidden_states" key will be used as default in many cases
        return {
            "hidden_states": hidden_states,
            "last_hidden_state": hidden_states[-1],
        }
