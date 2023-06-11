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

import torch
import torch.nn.functional as F
import math
from torch import nn


def w2v2_loss(model, w2v2_out, args, suffix):
    if args.trim_mask:
        x = w2v2_out['features'][w2v2_out['mask_indices']].view(w2v2_out['features'].size(0), -1, w2v2_out['features'].size(-1))
    else:
        x = w2v2_out['features'][w2v2_out['mask_indices']].view(-1, args.encoder_embed_dim)
    y = w2v2_out["masked_target"]
    negs = w2v2_out['negs']
    if model.target_glu:
        y = model.target_glu(w2v2_out['masked_target'])
        negs = model.target_glu(w2v2_out['negs'])

    x = model.final_proj(x)
    if args.trim_mask:
        logits = model.compute_preds_trim_mask(x, y, negs) # [num_negs+1, B, T_pred], which is consineSim between prediction and target
        num_negs_p1, B, T_pred = logits.shape
        logits = logits.view(num_negs_p1, -1) # [num_negs+1, B + T_pred]
        logits = logits.transpose(0,1) # [B + T_pred, num_negs+1]
        log_prob = F.log_softmax(logits, dim=1)
        target = torch.zeros(B*T_pred).type(torch.LongTensor).to(logits.device)
    else:
        logits = model.compute_preds(x, y, negs, w2v2_out['mask_indices']) # # [n_negatives+1, T_pred1+T_pred2+...+T_predB], which is consineSim between prediction and target
        logits = logits.transpose(0,1) # [T_pred1+T_pred2+...+T_predB, n_negatives+1]
        log_prob = F.log_softmax(logits, dim=1)
        target = torch.zeros(logits.shape[0]).type(torch.LongTensor).to(logits.device)
    contrastive_loss = F.nll_loss(log_prob, target)
    losses = {}
    if args.feature_grad_mult > 0.:
        diversity_loss, convnet_penalty = model.get_extra_losses(w2v2_out)
        losses[f'{suffix}_w2v2_contrastive_loss'] = contrastive_loss
        losses[f'{suffix}_w2v2_diversity_loss'] = diversity_loss
        losses[f'{suffix}_w2v2_convnet_penalty'] = convnet_penalty
    else:
        diversity_loss = model.get_extra_losses(w2v2_out)[0]
        losses[f'{suffix}_w2v2_contrastive_loss'] = contrastive_loss
        losses[f'{suffix}_w2v2_diversity_loss'] = diversity_loss
    w2v2_loss = 0
    for key in losses:
        w2v2_loss += losses[key]
    losses[f'{suffix}_w2v2_loss'] = w2v2_loss
    return losses

def Margin_InfoNCE_loss(S, margin, img_id=None):
    target = torch.LongTensor(list(range(S.size(0)))).to(S.device)
    deltas = margin * torch.eye(S.size(0)).to(S.device)
    S = S - deltas
    very_neg = torch.tensor(-10000.).to(S)
    if img_id is not None:
        img_id_equal_matrix = torch.from_numpy((img_id[:,None] == img_id[None,:])).to(S)
        diag_mask = (-1. * torch.eye(S.size(0)).to(S.device)) + torch.ones_like(S)
        mask = img_id_equal_matrix * diag_mask * very_neg # diag are all 0; for non-diag entries, -10000 if corresponding img_ids' are equal, otherwise 0
        # print(f"mask that is to be added to S: {mask}")
        # print(f"sum of entries of mask for each audio: {mask.sum(1)}") # actually -10000 rarely happened
        S = S + mask

    I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)            
    C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)        
    loss = I2C_loss + C2I_loss
    return loss


# model modules and functions

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

BertLayerNorm = torch.nn.LayerNorm

class VisualFeatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = config.visual_feat_dim
        pos_dim = config.visual_pos_dim
        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        feats, boxes = visn_input
        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        if boxes is not None:
            y = self.box_fc(boxes)
            y = self.box_layer_norm(y)
            output = (x + y) / 2
        else:
            output = x

        output = self.dropout(output)
        return output
class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None, return_attention_weight=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) #[b, heads, T_q, T_k]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) #[b, heads, T_q, T_k]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        #[b,heads,T_q,T_k]@[b,heads,T_k,C] -> [b,heads,T_q,C]
        context_layer = torch.matmul(attention_probs, value_layer) 
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() #[b,T_q,heads,C]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) 
        context_layer = context_layer.view(*new_context_layer_shape) #[b,T_q, heads*C]
        if return_attention_weight:
            assert not self.training
            return context_layer, attention_probs[:,:,0]
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None, return_attention_weight=False):
        if return_attention_weight:
            output, attention_weight = self.att(input_tensor, ctx_tensor, ctx_att_mask, return_attention_weight=True)
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        if return_attention_weight:
            return attention_output, attention_weight
        return attention_output


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertClassificationHead(nn.Module):
    def __init__(self, num_labels, hid_dim):
        super().__init__()


        in_dim = hid_dim
        out_dim = num_labels

        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim * 2),
            GeLU(),
            nn.LayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, out_dim),
        )

    def forward(self, x):
        logit = self.logit_fc(x)
        return logit


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertVisualObjHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        self.visual_losses = config.visual_losses

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict(
            {
                key: nn.Linear(config.hidden_size, config.visual_loss_config[key][0])
                for key in self.visual_losses
            }
        )

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class VisualFeatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = config.visual_feat_dim
        pos_dim = config.visual_pos_dim
        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        feats, boxes = visn_input
        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        if boxes is not None:
            y = self.box_fc(boxes)
            y = self.box_layer_norm(y)
            output = (x + y) / 2
        else:
            output = x

        output = self.dropout(output)
        return output
        
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.activation = nn.Tanh()
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class LXMERTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.audio_self_att = BertAttention(config)
        self.visn_self_att = BertAttention(config)

        # Intermediate and Output Layers (FFNs)
        self.audio_inter = BertIntermediate(config)
        self.audio_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(
        self, audio_input, audio_attention_mask, visn_input, visn_attention_mask, return_attention_weight=False
    ):
        # Cross Attention
        if return_attention_weight:
            audio_att_output, a2v_attention_weight = self.visual_attention(
            audio_input, visn_input, ctx_att_mask=visn_attention_mask, return_attention_weight=True
        )
        else:
            audio_att_output = self.visual_attention(
                audio_input, visn_input, ctx_att_mask=visn_attention_mask
            )
        visn_att_output = self.visual_attention(
            visn_input, audio_input, ctx_att_mask=audio_attention_mask
        )
        if return_attention_weight:
            return audio_att_output, visn_att_output, a2v_attention_weight
        return audio_att_output, visn_att_output

    def self_att(
        self, audio_input, audio_attention_mask, visn_input, visn_attention_mask
    ):
        # Self Attention
        #audio_att_output = self.audio_self_att(audio_input, audio_attention_mask)[0]
        #visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)[0]
        # not [0], maybe this is only when we use transformer's bert modules
        audio_att_output = self.audio_self_att(audio_input, audio_input, audio_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_input, visn_attention_mask)
        return audio_att_output, visn_att_output

    def output_fc(self, audio_input, visn_input):
        # FC layers
        audio_inter_output = self.audio_inter(audio_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        audio_output = self.audio_output(audio_inter_output, audio_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return audio_output, visn_output

    def forward(self, audio_feats, audio_attention_mask, visn_feats, visn_attention_mask, return_attention_weight=False):
        audio_att_output = audio_feats
        visn_att_output = visn_feats
        if return_attention_weight:
            audio_att_output, visn_att_output, a2v_attention_weight = self.cross_att(
            audio_att_output, audio_attention_mask, visn_att_output, visn_attention_mask, True
        )
        else:
            audio_att_output, visn_att_output = self.cross_att(
                audio_att_output, audio_attention_mask, visn_att_output, visn_attention_mask
            )
        audio_att_output, visn_att_output = self.self_att(
            audio_att_output, audio_attention_mask, visn_att_output, visn_attention_mask
        )
        audio_output, visn_output = self.output_fc(audio_att_output, visn_att_output)
        if return_attention_weight:
            return audio_output, visn_output, a2v_attention_weight

        return audio_output, visn_output

class Last_LXMERTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Self-attention Layers
        # self.audio_self_att = BertAttention(config)
        self.visn_self_att = BertAttention(config)

        # Intermediate and Output Layers (FFNs)
        # self.audio_inter = BertIntermediate(config)
        # self.audio_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(
        self, audio_input, audio_attention_mask, visn_input, visn_attention_mask, return_attention_weight=False
    ):
        # Cross Attention
        if return_attention_weight:
            audio_att_output, a2v_attention_weight = self.visual_attention(
            audio_input, visn_input, ctx_att_mask=visn_attention_mask, return_attention_weight=True
        )
        else:
            audio_att_output = self.visual_attention(
                audio_input, visn_input, ctx_att_mask=visn_attention_mask
            )
        visn_att_output = self.visual_attention(
            visn_input, audio_input, ctx_att_mask=audio_attention_mask
        )
        if return_attention_weight:
            return audio_att_output, visn_att_output, a2v_attention_weight
        return audio_att_output, visn_att_output

    def self_att(
        self, visn_input, visn_attention_mask, return_attention_weight=False
    ):
        # Self Attention
        #audio_att_output = self.audio_self_att(audio_input, audio_attention_mask)[0]
        #visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)[0]
        # not [0], maybe this is only when we use transformer's bert modules
        # audio_att_output = self.audio_self_att(audio_input, audio_input, audio_attention_mask)
        # return audio_att_output, visn_att_output
        if return_attention_weight:
            visn_att_output, attention_weight = self.visn_self_att(visn_input, visn_input, visn_attention_mask, return_attention_weight=True)
            return visn_att_output, attention_weight
        else:
            visn_att_output = self.visn_self_att(visn_input, visn_input, visn_attention_mask)
            return visn_att_output

    def output_fc(self, visn_input):
        # FC layers
        # audio_inter_output = self.audio_inter(audio_input)
        # visn_inter_output = self.visn_inter(visn_input)

        # # Layer output
        # audio_output = self.audio_output(audio_inter_output, audio_input)
        # visn_output = self.visn_output(visn_inter_output, visn_input)
        # return audio_output, visn_output

        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return visn_output


    def forward(self, audio_feats, audio_attention_mask, visn_feats, visn_attention_mask, return_attention_weight=False):
        audio_att_output = audio_feats
        visn_att_output = visn_feats
        # if return_attention_weight:
        #     audio_att_output, visn_att_output, a2v_attention_weight = self.cross_att(
        #     audio_att_output, audio_attention_mask, visn_att_output, visn_attention_mask, True
        # )
        # else:
        #     audio_att_output, visn_att_output = self.cross_att(
        #         audio_att_output, audio_attention_mask, visn_att_output, visn_attention_mask
        #     )
        audio_att_output, visn_att_output = self.cross_att(
                audio_att_output, audio_attention_mask, visn_att_output, visn_attention_mask
            )
        if return_attention_weight:
            visn_att_output, attention_weight = self.self_att(
                visn_att_output, visn_attention_mask, return_attention_weight=True
            )
            visn_output = self.output_fc(visn_att_output)
            return visn_output, attention_weight
        else:
            visn_att_output = self.self_att(
                visn_att_output, visn_attention_mask
            )
            visn_output = self.output_fc(visn_att_output)
            return visn_output
