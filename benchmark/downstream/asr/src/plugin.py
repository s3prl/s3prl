import torch
from torch import nn
from src.util import load_embedding
from src.bert_embedding import BertEmbeddingPredictor


class EmbeddingRegularizer(nn.Module):
    ''' Perform word embedding regularization training for ASR'''

    def __init__(self, tokenizer, dec_dim, enable, src, distance, weight, fuse, temperature,
                 freeze=True, fuse_normalize=False, dropout=0.0, bert=None):
        super(EmbeddingRegularizer, self).__init__()
        self.enable = enable
        if enable:
            if bert is not None:
                self.use_bert = True
                if not isinstance(bert, str):
                    raise ValueError(
                        "`bert` should be a str specifying bert config such as \"bert-base-uncased\".")
                self.emb_table = BertEmbeddingPredictor(bert, tokenizer, src)
                vocab_size, emb_dim = self.emb_table.model.bert.embeddings.word_embeddings.weight.shape
                vocab_size = vocab_size-3  # cls,sep,mask not used
                self.dim = emb_dim
            else:
                self.use_bert = False
                pretrained_emb = torch.FloatTensor(
                    load_embedding(tokenizer, src))
                # pretrained_emb = nn.functional.normalize(pretrained_emb,dim=-1) # ToDo : Check impact on old version
                vocab_size, emb_dim = pretrained_emb.shape
                self.dim = emb_dim

                self.emb_table = nn.Embedding.from_pretrained(
                    pretrained_emb, freeze=freeze, padding_idx=0)

            self.emb_net = nn.Sequential(nn.Linear(dec_dim, (emb_dim+dec_dim)//2),
                                         nn.ReLU(),
                                         nn.Linear((emb_dim+dec_dim)//2, emb_dim))
            self.weight = weight
            self.distance = distance
            self.fuse_normalize = fuse_normalize
            if distance == 'CosEmb':
                # This maybe somewhat reduandant since cos emb loss includes ||x||
                self.measurement = nn.CosineEmbeddingLoss(reduction='none')
            elif distance == 'MSE':
                self.measurement = nn.MSELoss(reduction='none')
            else:
                raise NotImplementedError

            self.apply_dropout = dropout > 0
            if self.apply_dropout:
                self.dropout = nn.Dropout(dropout)

            self.apply_fuse = fuse != 0
            if self.apply_fuse:
                # Weight for mixing emb/dec prob
                if fuse == -1:
                    # Learnable fusion
                    self.fuse_type = "learnable"
                    self.fuse_learnable = True
                    self.fuse_lambda = nn.Parameter(
                        data=torch.FloatTensor([0.5]))
                elif fuse == -2:
                    # Learnable vocab-wise fusion
                    self.fuse_type = "vocab-wise learnable"
                    self.fuse_learnable = True
                    self.fuse_lambda = nn.Parameter(
                        torch.ones((vocab_size))*0.5)
                else:
                    self.fuse_type = str(fuse)
                    self.fuse_learnable = False
                    self.register_buffer(
                        'fuse_lambda', torch.FloatTensor([fuse]))
                # Temperature of emb prob.
                if temperature == -1:
                    self.temperature = 'learnable'
                    self.temp = nn.Parameter(data=torch.FloatTensor([1]))
                elif temperature == -2:
                    self.temperature = 'elementwise'
                    self.temp = nn.Parameter(torch.ones((vocab_size)))
                else:
                    self.temperature = str(temperature)
                    self.register_buffer(
                        'temp', torch.FloatTensor([temperature]))
                self.eps = 1e-8

    def create_msg(self):
        msg = ['Plugin.    | Word embedding regularization enabled (type:{}, weight:{})'.format(
            self.distance, self.weight)]
        if self.apply_fuse:
            msg.append('           | Embedding-fusion decoder enabled ( temp. = {}, lambda = {} )'.
                       format(self.temperature, self.fuse_type))
        return msg

    def get_weight(self):
        if self.fuse_learnable:
            return torch.sigmoid(self.fuse_lambda).mean().cpu().data
        else:
            return self.fuse_lambda

    def get_temp(self):
        return nn.functional.relu(self.temp).mean()

    def fuse_prob(self, x_emb, dec_logit):
        ''' Takes context and decoder logit to perform word embedding fusion '''
        # Compute distribution for dec/emb
        if self.fuse_normalize:
            emb_logit = nn.functional.linear(nn.functional.normalize(x_emb, dim=-1),
                                             nn.functional.normalize(self.emb_table.weight, dim=-1))
        else:
            emb_logit = nn.functional.linear(x_emb, self.emb_table.weight)
        emb_prob = (nn.functional.relu(self.temp)*emb_logit).softmax(dim=-1)
        dec_prob = dec_logit.softmax(dim=-1)
        # Mix distribution
        if self.fuse_learnable:
            fused_prob = (1-torch.sigmoid(self.fuse_lambda))*dec_prob +\
                torch.sigmoid(self.fuse_lambda)*emb_prob
        else:
            fused_prob = (1-self.fuse_lambda)*dec_prob + \
                self.fuse_lambda*emb_prob
        # Log-prob
        log_fused_prob = (fused_prob+self.eps).log()

        return log_fused_prob

    def forward(self, dec_state, dec_logit, label=None, return_loss=True):
        # Match embedding dim.
        log_fused_prob = None
        loss = None

        #x_emb = nn.functional.normalize(self.emb_net(dec_state),dim=-1)
        if self.apply_dropout:
            dec_state = self.dropout(dec_state)
        x_emb = self.emb_net(dec_state)

        if return_loss:
            # Compute embedding loss
            b, t = label.shape
            # Retrieve embedding
            if self.use_bert:
                with torch.no_grad():
                    y_emb = self.emb_table(label).contiguous()
            else:
                y_emb = self.emb_table(label)
            # Regression loss on embedding
            if self.distance == 'CosEmb':
                loss = self.measurement(
                    x_emb.view(-1, self.dim), y_emb.view(-1, self.dim), torch.ones(1).to(dec_state.device))
            else:
                loss = self.measurement(
                    x_emb.view(-1, self.dim), y_emb.view(-1, self.dim))
            loss = loss.view(b, t)
            # Mask out padding
            loss = torch.where(label != 0, loss, torch.zeros_like(loss))
            loss = torch.mean(loss.sum(dim=-1) /
                              (label != 0).sum(dim=-1).float())

        if self.apply_fuse:
            log_fused_prob = self.fuse_prob(x_emb, dec_logit)

        return loss, log_fused_prob
