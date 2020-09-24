# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model_quantize.py ]
#   Synopsis     [ Implementation of the VQ Layer and GST Layer ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


###################
# VQ LAYER GUMBEL #
###################

# Reference: https://github.com/pytorch/fairseq/blob/master/fairseq/modules/gumbel_vector_quantizer.py

class VectorQuantizeLayer_GB(nn.Module):
    def __init__(
        self,
        input_dim,
        vq_size,
        vq_dim,
        temp=(1.0, 0.1, 0.99),
        groups=1,
        combine_groups=True,
        time_first=True,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
    ):
        """Vector quantization using gumbel softmax
        Args:
            input_dim: input dimension (channels)
            vq_size: number of quantized vectors per group
            vq_dim: dimensionality of the resulting quantized vector
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.input_dim = input_dim
        self.vq_size = vq_size
        self.groups = groups
        self.combine_groups = combine_groups
        self.time_first = time_first
        self.out_dim = vq_dim

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * vq_size, var_dim))
        nn.init.uniform_(self.vars)

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, groups * vq_size),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * vq_size)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        assert len(temp) == 3, temp

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.vq_size)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(
                    self.vq_size ** self.groups, -1
                )
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.vq_size * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            self.vars.squeeze(0)
                .index_select(0, indices)
                .view(self.vq_size ** self.groups, -1)
        )

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert (
                n < cb_size
        ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * (self.vq_size ** exponent)
        return res

    def forward(self, x, produce_targets=False):

        result = {"vq_size": self.vq_size * self.groups}

        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        # hard_probs = torch.mean(hard_x.float(), dim=0)
        # result["code_perplexity"] = torch.exp(
        #     -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        # ).sum()

        # avg_probs = torch.softmax(
        #     x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        # ).mean(dim=0)
        # result["prob_perplexity"] = torch.exp(
        #     -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        # ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                .argmax(dim=-1)
                .view(bsz, tsz, self.groups)
                .detach()
            )

        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.vq_size, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)

        if not self.time_first:
            x = x.transpose(1, 2)  # BTC -> BCT

        # result["x"] = x
        # return result
        return x


###################
# VQ LAYER GUMBEL #
###################

# Reference: https://github.com/iamyuanchung/VQ-APC/blob/master/vqapc_model.py

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """From https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
    logits: a tensor of shape (*, n_class)
    returns an one-hot vector of shape (*, n_class)
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


class VectorQuantizeLayer_GB_deprecated(nn.Module):
    '''
    inputs --- [batch_size, time_step, input_size]
    outputs --- [batch_size, time_step, vq_dim]
    '''
    def __init__(self, input_size, vocab_size, vq_dim, hidden_size=0,
                gumbel_temperature=0.5):
        """Defines a VQ layer that follows an RNN layer.
        input_size: an int indicating the pre-quantized input feature size,
            usually the hidden size of RNN.
        hidden_size: an int indicating the hidden size of the 1-layer MLP applied
            before gumbel-softmax. If equals to 0 then no MLP is applied.
        vocab_size: an int indicating the number of codes.
        vq_dim: an int indicating the size of each code. If not the last layer,
            then must equal to the RNN hidden size.
        gumbel_temperature: a float indicating the temperature for gumbel-softmax.
        """
        super(VectorQuantizeLayer_GB_deprecated, self).__init__()

        self.out_dim = vq_dim
        self.with_hiddens = hidden_size > 0

        # RNN hiddens to VQ hiddens.
        if self.with_hiddens:
            # Apply a linear layer, followed by a ReLU and another linear. Following
            # https://arxiv.org/abs/1910.05453
            self.vq_hiddens = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.vq_logits = nn.Linear(hidden_size, vocab_size)
        else:
            # Directly map to logits without any transformation.
            self.vq_logits = nn.Linear(input_size, vocab_size)

        self.gumbel_temperature = gumbel_temperature
        self.codebook_CxE = nn.Linear(vocab_size, vq_dim, bias=False)


    def forward(self, inputs_BxLxI):
        if self.with_hiddens:
            hiddens_BxLxH = self.relu(self.vq_hiddens(inputs_BxLxI))
            logits_BxLxC = self.vq_logits(hiddens_BxLxH)
        else:
            logits_BxLxC = self.vq_logits(inputs_BxLxI)

        if not self.training:
            # During inference, just take the max index.
            shape = logits_BxLxC.size()
            _, ind = logits_BxLxC.max(dim=-1)
            onehot_BxLxC = torch.zeros_like(logits_BxLxC).view(-1, shape[-1])
            onehot_BxLxC.scatter_(1, ind.view(-1, 1), 1)
            onehot_BxLxC = onehot_BxLxC.view(*shape)
        else:
            onehot_BxLxC = gumbel_softmax(logits_BxLxC, temperature=self.gumbel_temperature)
        
        codes_BxLxE = self.codebook_CxE(onehot_BxLxC)

        #return logits_BxLxC, codes_BxLxE
        return codes_BxLxE


###############
# VQ LAYER L2 #
###############

# Reference: https://github.com/ttaoREtw/semi-tts/blob/master/src/embed.py

class VectorQuantizeLayer_L2(nn.Module):
    def __init__(self, input_dim, vocab_size, vq_dim, temp=1, skip_prob=0, stop_grad=False):
        super(VectorQuantizeLayer_L2, self).__init__()
        
        # Required attributes
        self.out_dim = vq_dim
        
        # Latent embedding
        self.onehot = nn.Embedding.from_pretrained(torch.eye(vocab_size), freeze=True)
        
        # Scaling factor
        if temp < 0:
            self.temp = nn.Parameter(torch.FloatTensor([1]))
        else:
            self.register_buffer('temp', torch.FloatTensor([temp]))
        
        # Criterion for deriving distribution
        self.measurement = neg_batch_l2

        # Skip connection of enc/dec
        self.skip_prob = skip_prob

        # Speech2speech gradient applied on embedding
        self.stop_grad = stop_grad
        
        # Random init. learnable embedding
        self.linear = nn.Linear(input_dim, vq_dim)
        self.learnable_table = nn.Parameter(torch.randn((vocab_size, vq_dim)))
    
    @property
    def embedding(self):
        return nn.Embedding.from_pretrained(self.learnable_table)

    def forward(self, x):
        B, S, _ = x.shape

        x = self.linear(x)
        similarity = F.relu(self.temp)*self.measurement(x, self.learnable_table, B, S)
        
        # Compute enc. output distribution over codebook (based on L2)
        p_code = similarity.softmax(dim=-1)
        
        # Select nearest neighbor in codebook
        picked_idx = p_code.argmax(dim=-1)
        
        if self.stop_grad:
            # Stop-grad version
            picked_code = F.embedding(picked_idx, self.learnable_table)
        else:
            # Straight-through Gradient Estimation onehot version
            p_hard = p_code + (self.onehot(picked_idx) - p_code).detach()
            picked_code = F.linear(p_hard, self.learnable_table.T)
        
        if self.training and self.skip_prob > 0 and np.random.rand() < self.skip_prob:
            # skip connection (only when training)
            vq_code = x
        else:
            # Quantize
            vq_code = x + picked_code - x.detach()

        #return picked_code, vq_code
        return vq_code


def neg_batch_l2(x, y, B, S):
    flat_x = x.reshape(B*S, -1) 
    l2_distance = torch.sum(flat_x.pow(2), dim=-1, keepdim=True) \
                  + torch.sum(y.pow(2), dim=-1) \
                  - 2 * torch.matmul(flat_x, y.t())
    return - l2_distance.view(B, S, -1)


################
# Other LAYERS #
################
class LinearLayer(nn.Module):
    def __init__(self, input_dim, out_dim, use_activation=True):
        super(LinearLayer, self).__init__()
        self.out_dim = out_dim # Required attributes
        self.linear = nn.Linear(input_dim, out_dim)
        self.use_activation = use_activation
        self.activation = nn.GELU()

    def forward(self, x, sequence_data=True):
        if not sequence_data: # [N, hidden_size]
            x = x.unsqueeze(1) # [N, 1, hidden_size]
        x = self.linear(x)
        if self.use_activation:
            x = self.activation(x)
        return x


#############
# GST LAYER #
#############

# Reference: https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py

class GlobalStyleTokenLayer(nn.Module):
    '''
    inputs --- [N, 1, hidden_size] or [N, T, hidden_size]
    outputs --- [N, 1, hidden_size] or [N, T, hidden_size]
    '''

    def __init__(self, input_dim, token_num, hidden_size, num_heads=8):

        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(token_num, hidden_size))
        init.normal_(self.embed, mean=0, std=0.5)

        self.attention = MultiHeadAttention(query_dim=input_dim, key_dim=hidden_size, num_units=hidden_size, num_heads=num_heads)
        self.out_dim = hidden_size


    def forward(self, query):
        N = query.size(0)
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, hidden_size]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out