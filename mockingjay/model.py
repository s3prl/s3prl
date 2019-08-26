# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ mockingjay/solver.py ]
#   Synopsis     [ solver for the mockingjay model]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference 1  [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
"""PyTorch Mockingjay model."""
import copy
import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
"""
1. should loss be masked? only predict the masked token rather than reconstructing the entire input.
2. does spec matter? linear or mel

TODO:
spec_mask
attention_mask
masked input
"""

class MockingjayConfig(object):
	"""Configuration class to store the configuration of a `MockingjayModel`.
	"""
	def __init__(self, config):
		self.downsample_rate = config['mockingjay']['downsample_rate']
		self.hidden_size = config['mockingjay']['hidden_size']
		self.num_hidden_layers = config['mockingjay']['num_hidden_layers']
		self.num_attention_heads = config['mockingjay']['num_attention_heads']
		self.hidden_act = config['mockingjay']['hidden_act']
		self.intermediate_size = config['mockingjay']['intermediate_size']
		self.hidden_dropout_prob = config['mockingjay']['hidden_dropout_prob']
		self.attention_probs_dropout_prob = config['mockingjay']['attention_probs_dropout_prob']
		self.initializer_range = config['mockingjay']['initializer_range']
		self.layer_norm_eps = float(config['mockingjay']['layer_norm_eps'])


def prune_linear_layer(layer, index, dim=0):
	""" Prune a linear layer (a model parameters) to keep only entries in index.
		Return the pruned layer as a new layer with requires_grad=True.
		Used to remove heads.
	"""
	index = index.to(layer.weight.device)
	W = layer.weight.index_select(dim, index).clone().detach()
	if layer.bias is not None:
		if dim == 1:
			b = layer.bias.clone().detach()
		else:
			b = layer.bias[index].clone().detach()
	new_size = list(layer.weight.size())
	new_size[dim] = len(index)
	new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
	new_layer.weight.requires_grad = False
	new_layer.weight.copy_(W.contiguous())
	new_layer.weight.requires_grad = True
	if layer.bias is not None:
		new_layer.bias.requires_grad = False
		new_layer.bias.copy_(b.contiguous())
		new_layer.bias.requires_grad = True
	return new_layer


def gelu(x):
	"""Implementation of the gelu activation function.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
		Also see https://arxiv.org/abs/1606.08415
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
	return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


try:
	from apex.normalization.fused_layer_norm import FusedLayerNorm as MockingjayLayerNorm
except ImportError:
	print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
	class MockingjayLayerNorm(nn.Module):
		def __init__(self, hidden_size, eps=1e-12):
			"""Construct a layernorm module in the TF style (epsilon inside the square root).
			"""
			super(MockingjayLayerNorm, self).__init__()
			self.weight = nn.Parameter(torch.ones(hidden_size))
			self.bias = nn.Parameter(torch.zeros(hidden_size))
			self.variance_epsilon = eps

		def forward(self, x):
			u = x.mean(-1, keepdim=True)
			s = (x - u).pow(2).mean(-1, keepdim=True)
			x = (x - u) / torch.sqrt(s + self.variance_epsilon)
			return self.weight * x + self.bias


def position_encoding(seq_len, hidden_size, padding_idx=None):
	''' Sinusoid position encoding table '''
 
	def cal_angle(position, hid_idx):
		return position / np.power(10000, 2 * (hid_idx // 2) / hidden_size)
 
	def get_posi_angle_vec(position):
		return [cal_angle(position, hid_j) for hid_j in range(hidden_size)]
 
	sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(seq_len)])
 
	sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
	sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
 
	if padding_idx is not None:
		sinusoid_table[padding_idx] = 0. # zero vector for padding dimension
 
	return torch.FloatTensor(sinusoid_table)  # seq_len × hidden_size


class MockingjayInputRepresentations(nn.Module):
	"""Construct the input representation from spectrogram, and position encodings.
	"""
	def __init__(self, config, input_dim):
		super(MockingjayInputRepresentations, self).__init__()
		self.hidden_size = config.hidden_size
		self.spec_transform = nn.Linear(input_dim*config.downsample_rate, config.hidden_size)

		# self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
		# any TensorFlow checkpoint file
		self.LayerNorm = MockingjayLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, spec):
		seq_length = spec.size(1)
		spec_transformed = self.spec_transform(spec)
		position_encoded = position_encoding(seq_len, self.hidden_size)

		input_representations = spec_transformed + position_encoded
		input_representations = self.LayerNorm(input_representations)
		input_representations = self.dropout(input_representations)
		return input_representations


class MockingjaySelfAttention(nn.Module):
	def __init__(self, config, output_attentions=False, keep_multihead_output=False):
		super(MockingjaySelfAttention, self).__init__()
		if config.hidden_size % config.num_attention_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (config.hidden_size, config.num_attention_heads))
		self.output_attentions = output_attentions
		self.keep_multihead_output = keep_multihead_output
		self.multihead_output = None

		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, hidden_states, attention_mask, head_mask=None):
		mixed_query_layer = self.query(hidden_states)
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		# Apply the attention mask is (precomputed for all layers in MockingjayModel forward() function)
		attention_scores = attention_scores + attention_mask

		# Normalize the attention scores to probabilities.
		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		# Mask heads if we want to
		if head_mask is not None:
			attention_probs = attention_probs * head_mask

		context_layer = torch.matmul(attention_probs, value_layer)
		if self.keep_multihead_output:
			self.multihead_output = context_layer
			self.multihead_output.retain_grad()

		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		if self.output_attentions:
			return attention_probs, context_layer
		return context_layer


class MockingjaySelfOutput(nn.Module):
	def __init__(self, config):
		super(MockingjaySelfOutput, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.LayerNorm = MockingjayLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class MockingjayAttention(nn.Module):
	def __init__(self, config, output_attentions=False, keep_multihead_output=False):
		super(MockingjayAttention, self).__init__()
		self.output_attentions = output_attentions
		self.self = MockingjaySelfAttention(config, output_attentions=output_attentions,
											  keep_multihead_output=keep_multihead_output)
		self.output = MockingjaySelfOutput(config)

	def prune_heads(self, heads):
		if len(heads) == 0:
			return
		mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
		for head in heads:
			mask[head] = 0
		mask = mask.view(-1).contiguous().eq(1)
		index = torch.arange(len(mask))[mask].long()
		# Prune linear layers
		self.self.query = prune_linear_layer(self.self.query, index)
		self.self.key = prune_linear_layer(self.self.key, index)
		self.self.value = prune_linear_layer(self.self.value, index)
		self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
		# Update hyper params
		self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
		self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

	def forward(self, input_tensor, attention_mask, head_mask=None):
		self_output = self.self(input_tensor, attention_mask, head_mask)
		if self.output_attentions:
			attentions, self_output = self_output
		attention_output = self.output(self_output, input_tensor)
		if self.output_attentions:
			return attentions, attention_output
		return attention_output


class MockingjayIntermediate(nn.Module):
	def __init__(self, config):
		super(MockingjayIntermediate, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
		if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
			self.intermediate_act_fn = ACT2FN[config.hidden_act]
		else:
			self.intermediate_act_fn = config.hidden_act

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states


class MockingjayOutput(nn.Module):
	def __init__(self, config):
		super(MockingjayOutput, self).__init__()
		self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
		self.LayerNorm = MockingjayLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class MockingjayLayer(nn.Module):
	def __init__(self, config, output_attentions=False, keep_multihead_output=False):
		super(MockingjayLayer, self).__init__()
		self.output_attentions = output_attentions
		self.attention = MockingjayAttention(config, output_attentions=output_attentions,
											   keep_multihead_output=keep_multihead_output)
		self.intermediate = MockingjayIntermediate(config)
		self.output = MockingjayOutput(config)

	def forward(self, hidden_states, attention_mask, head_mask=None):
		attention_output = self.attention(hidden_states, attention_mask, head_mask)
		if self.output_attentions:
			attentions, attention_output = attention_output
		intermediate_output = self.intermediate(attention_output)
		layer_output = self.output(intermediate_output, attention_output)
		if self.output_attentions:
			return attentions, layer_output
		return layer_output


class MockingjayEncoder(nn.Module):
	def __init__(self, config, output_attentions=False, keep_multihead_output=False):
		super(MockingjayEncoder, self).__init__()
		self.output_attentions = output_attentions
		layer = MockingjayLayer(config, output_attentions=output_attentions,
								  keep_multihead_output=keep_multihead_output)
		self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

	def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, head_mask=None):
		all_encoder_layers = []
		all_attentions = []
		for i, layer_module in enumerate(self.layer):
			hidden_states = layer_module(hidden_states, attention_mask, head_mask[i])
			if self.output_attentions:
				attentions, hidden_states = hidden_states
				all_attentions.append(attentions)
			if output_all_encoded_layers:
				all_encoder_layers.append(hidden_states)
		if not output_all_encoded_layers:
			all_encoder_layers.append(hidden_states)
		if self.output_attentions:
			return all_attentions, all_encoder_layers
		return all_encoder_layers


class MockingjaySpecPredictionHead(nn.Module):
	def __init__(self, config, x_sample):
		super(MockingjaySpecPredictionHead, self).__init__()
		self.input_dim = x_sample.shape[-1]
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
			self.transform_act_fn = ACT2FN[config.hidden_act]
		else:
			self.transform_act_fn = config.hidden_act
		self.LayerNorm = MockingjayLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.output = nn.Linear(config.hidden_size, self.input_dim * config.downsample_rate)

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.transform_act_fn(hidden_states)
		hidden_states = self.LayerNorm(hidden_states)
		linear_output = self.output(hidden_states)
		return linear_output


class MockingjayPreTrainedModel(nn.Module):
	""" An abstract class to handle weights initialization and
		a simple interface for dowloading and loading pretrained models.
	"""
	def __init__(self, config, *inputs, **kwargs):
		super(MockingjayPreTrainedModel, self).__init__()
		self.config = config

	def init_Mockingjay_weights(self, module):
		""" Initialize the weights.
		"""
		if isinstance(module, (nn.Linear, nn.Embedding)):
			# Slightly different from the TF version which uses truncated_normal for initialization
			# cf https://github.com/pytorch/pytorch/pull/5617
			module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
		elif isinstance(module, MockingjayLayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()


class MockingjayModel(MockingjayPreTrainedModel):
	"""Mockingjay model ("Bidirectional Embedding Representations from a Transformer").

	Params:
		`config`: a MockingjayConfig class instance with the configuration to build a new model
		`output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
		`keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
			This can be used to compute head importance metrics. Default: False

	Inputs:
		`input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
			with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
			`extract_features.py`, `run_classifier.py` and `run_squad.py`)
		`token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
			types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
			a `sentence B` token (see Mockingjay paper for more details).
		`attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
			selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
			input sequence length in the current batch. It's the mask that we typically use for attention when
			a batch has varying length sentences.
		`output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
		`head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
			It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.


	Outputs: Tuple of (encoded_layers, pooled_output)
		`encoded_layers`: controled by `output_all_encoded_layers` argument:
			- `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
				of each attention block (i.e. 12 full sequences for Mockingjay-base, 24 for Mockingjay-large), each
				encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
			- `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
				to the last attention block of shape [batch_size, sequence_length, hidden_size],
		`pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
			classifier pretrained on top of the hidden state associated to the first character of the
			input (`CLS`) to train on the Next-Sentence task (see Mockingjay's paper).

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

	config = modeling.MockingjayConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

	model = modeling.MockingjayModel(config=config)
	all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config, x_sample, output_attentions=False, keep_multihead_output=False):
		super(MockingjayModel, self).__init__(config)
		self.input_dim = x_sample.shape[-1]
		self.output_attentions = output_attentions
		self.input_representations = MockingjayInputRepresentations(config, self.input_dim)
		self.encoder = MockingjayEncoder(config, output_attentions=output_attentions,
										   keep_multihead_output=keep_multihead_output)
		self.apply(self.init_Mockingjay_weights)

	def prune_heads(self, heads_to_prune):
		""" Prunes heads of the model.
			heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
		"""
		for layer, heads in heads_to_prune.items():
			self.encoder.layer[layer].attention.prune_heads(heads)

	def get_multihead_outputs(self):
		""" Gather all multi-head outputs.
			Return: list (layers) of multihead module outputs with gradients
		"""
		return [layer.attention.self.multihead_output for layer in self.encoder.layer]

	def forward(self, input_ids, attention_mask=None, output_all_encoded_layers=True, head_mask=None):
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids)

		# We create a 3D attention mask from a 2D tensor mask.
		# Sizes are [batch_size, 1, 1, to_seq_length]
		# So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
		# this attention mask is more simple than the triangular masking of causal attention
		# used in OpenAI GPT, we just need to prepare the broadcast dimension here.
		extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

		# Since attention_mask is 1.0 for positions we want to attend and 0.0 for
		# masked positions, this operation will create a tensor which is 0.0 for
		# positions we want to attend and -10000.0 for masked positions.
		# Since we are adding it to the raw scores before the softmax, this is
		# effectively the same as removing these entirely.
		extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

		# Prepare head mask if needed
		# 1.0 in head_mask indicate we keep the head
		# attention_probs has shape bsz x n_heads x N x N
		# input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
		# and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
		if head_mask is not None:
			if head_mask.dim() == 1:
				head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
				head_mask = head_mask.expand_as(self.config.num_hidden_layers, -1, -1, -1, -1)
			elif head_mask.dim() == 2:
				head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
			head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
		else:
			head_mask = [None] * self.config.num_hidden_layers

		input_representations = self.input_representations(input_ids)
		encoded_layers = self.encoder(input_representations,
									  extended_attention_mask,
									  output_all_encoded_layers=output_all_encoded_layers,
									  head_mask=head_mask)
		if self.output_attentions:
			all_attentions, encoded_layers = encoded_layers
		sequence_output = encoded_layers[-1]
		if not output_all_encoded_layers:
			encoded_layers = encoded_layers[-1]
		if self.output_attentions:
			return all_attentions, encoded_layers
		return encoded_layers


class MockingjayForMaskedSpeechModel(MockingjayPreTrainedModel):
	"""Mockingjay model with the masked language modeling head.
	This module comprises the Mockingjay model followed by the masked language modeling head.

	Params:
		`config`: a MockingjayConfig class instance with the configuration to build a new model
		`output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
		`keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
			This can be used to compute head importance metrics. Default: False

	Inputs:
		`input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
			with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
			`extract_features.py`, `run_classifier.py` and `run_squad.py`)
		`token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
			types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
			a `sentence B` token (see Mockingjay paper for more details).
		`attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
			selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
			input sequence length in the current batch. It's the mask that we typically use for attention when
			a batch has varying length sentences.
		`masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
			with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
			is only computed for the labels set in [0, ..., vocab_size]
		`head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
			It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.

	Outputs:
		if `masked_lm_labels` is  not `None`:
			Outputs the masked language modeling loss.
		if `masked_lm_labels` is `None`:
			Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

	config = MockingjayConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
		num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

	model = MockingjayForMaskedLM(config)
	masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config, x_sample, output_attentions=False, keep_multihead_output=False):
		super(MockingjayForMaskedSpeechModel, self).__init__(config)
		self.output_attentions = output_attentions
		self.Mockingjay = MockingjayModel(config, x_sample, output_attentions=output_attentions,
									  keep_multihead_output=keep_multihead_output)
		self.project_spec = MockingjaySpecPredictionHead(config, x_sample)
		self.apply(self.init_Mockingjay_weights)

	def forward(self, spec_input, attention_mask=None, spec_label=None, mask_label=None, head_mask=None):
		outputs = self.Mockingjay(spec_input, attention_mask,
							output_all_encoded_layers=False,
							head_mask=head_mask)
		if self.output_attentions:
			all_attentions, sequence_output = outputs
		else:
			sequence_output = outputs
		pred_spec = self.project_spec(sequence_output)

		if spec_label is not None and mask_label is not None:
			loss = nn.SmoothL1Loss() 
			masked_spec_loss = loss(pred_spec.masked_select(mask_label), spec_label.masked_select(mask_label))
			return masked_spec_loss
		elif self.output_attentions:
			return all_attentions, pred_specs
		return pred_spec


