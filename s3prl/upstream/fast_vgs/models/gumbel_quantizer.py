# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelVectorQuantizer_trim_mask(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        temp,
        groups,
        combine_groups,
        vq_dim,
        time_first,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
    ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
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
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        if isinstance(temp, str):
            import ast
            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"

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

            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(
                    self.num_vars ** self.groups, -1
                )
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            self.vars.squeeze(0)
            .index_select(0, indices)
            .view(self.num_vars ** self.groups, -1)
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
            res += indices[..., i] * (self.num_vars ** exponent)
        return res

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False, mask_indices=None):

        result = {"num_vars": self.num_vars * self.groups}
        # mask_indices = torch.ones(x.shape[:2]).to(x) if mask_indices is None else mask_indices
        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        # mask_indices = mask_indices.reshape(-1) # [b,t] -> [b*t,]
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        ).mean(dim=0)
        # when produce quantize full x (not just masked part), the prob_perp is also calculated on full quantized sequence
        # might need to change to only masked part to follow w2v2, but I currently think this is a good idea
        # as I will model the codes as language tokens
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x) # x is onehots 
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        if produce_targets: # produce indices
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                .argmax(dim=-1)
                .view(bsz, tsz, self.groups)
                .detach()
            )
            # print(f"shape of code indices: {result['targets'].shape}")

        x = x.unsqueeze(-1) * vars # onehots times vars, like index selection
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)
        c = x.shape[-1]
        
        if mask_indices is not None:
            # print(f"what's the result of directly take the x[mask_indices] but not view (reshape) it {x[mask_indices].shape}")
            x = x[mask_indices].view(bsz, -1, c)
        if not self.time_first:
            x = x.transpose(1, 2)  # BTC -> BCT

        result["x"] = x

        return result



class GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        temp,
        groups,
        combine_groups,
        vq_dim,
        time_first,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
    ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
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
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        if isinstance(temp, str):
            import ast
            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"

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

            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(
                    self.num_vars ** self.groups, -1
                )
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            self.vars.squeeze(0)
            .index_select(0, indices)
            .view(self.num_vars ** self.groups, -1)
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
            res += indices[..., i] * (self.num_vars ** exponent)
        return res

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False, mask_indices=None):

        result = {"num_vars": self.num_vars * self.groups}
        # mask_indices = torch.ones(x.shape[:2]).to(x) if mask_indices is None else mask_indices
        # if not self.time_first:
        #     x = x.transpose(1, 2)
        bt, fsz = x.shape
        # mask_indices = mask_indices.reshape(-1) # [b,t] -> [b*t,]
        x = self.weight_proj(x)
        x = x.view(bt * self.groups, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bt, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        avg_probs = torch.softmax(
            x.view(bt, self.groups, -1).float(), dim=-1
        ).mean(dim=0)
        # when produce quantize full x (not just masked part), the prob_perp is also calculated on full quantized sequence
        # might need to change to only masked part to follow w2v2, but I currently think this is a good idea
        # as I will model the codes as language tokens
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x) # x is onehots 
        else:
            x = hard_x

        x = x.view(bt, -1)

        vars = self.vars
        if self.combine_groups:
            vars = vars.repeat(1, self.groups, 1)

        if produce_targets: # produce indices
            raise NotImplementedError
            # result["targets"] = (
            #     x.view(bt * self.groups, -1)
            #     .argmax(dim=-1)
            #     .view(bsz, tsz, self.groups)
            #     .detach()
            # )
            # # print(f"shape of code indices: {result['targets'].shape}")

        x = x.unsqueeze(-1) * vars # onehots times vars, like index selection
        x = x.view(bt, self.groups, self.num_vars, -1)
        x = x.sum(-2)
        # x = x.view(bsz, tsz, -1)
        x = x.view(bt,-1)
        # c = x.shape[-1]
        
        # if mask_indices is not None:
        #     # print(f"what's the result of directly take the x[mask_indices] but not view (reshape) it {x[mask_indices].shape}")
        #     x = x[mask_indices].view(bsz, -1, c)
        # if not self.time_first:
        #     x = x.transpose(1, 2)  # BTC -> BCT

        result["x"] = x

        return result
