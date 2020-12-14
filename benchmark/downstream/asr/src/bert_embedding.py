import torch
import torch.nn as nn

from src import text
from pytorch_pretrained_bert import BertForMaskedLM
from pytorch_pretrained_bert.modeling import BertOnlyMLMHead


class BertLikeSentencePieceTextEncoder(object):
    def __init__(self, text_encoder):
        if not isinstance(text_encoder, text.SubwordTextEncoder):
            raise TypeError(
                "`text_encoder` must be an instance of `src.text.SubwordTextEncoder`.")
        self.text_encoder = text_encoder

    @property
    def vocab_size(self):
        # +3 accounts for [CLS], [SEP] and [MASK]
        return self.text_encoder.vocab_size + 3

    @property
    def cls_idx(self):
        return self.vocab_size - 3

    @property
    def sep_idx(self):
        return self.vocab_size - 2

    @property
    def mask_idx(self):
        return self.vocab_size - 1

    @property
    def eos_idx(self):
        return self.text_encoder.eos_idx


def generate_embedding(bert_model, labels):
    """Generate bert's embedding from fine-tuned model."""
    batch_size, time = labels.shape

    cls_ids = torch.full(
        (batch_size, 1), bert_model.bert_text_encoder.cls_idx, dtype=labels.dtype, device=labels.device)
    bert_labels = torch.cat([cls_ids, labels], 1)
    # replace eos with sep
    eos_idx = bert_model.bert_text_encoder.eos_idx
    sep_idx = bert_model.bert_text_encoder.sep_idx
    bert_labels[bert_labels == eos_idx] = sep_idx

    embedding, _ = bert_model.bert(bert_labels, output_all_encoded_layers=True)
    # sum over all layers embedding
    embedding = torch.stack(embedding).sum(0)
    # get rid of cls
    embedding = embedding[:, 1:]

    assert labels.shape == embedding.shape[:-1]

    return embedding


def load_fine_tuned_model(bert_model, text_encoder, path):
    """Load fine-tuned bert model given text encoder and checkpoint path."""
    bert_text_encoder = BertLikeSentencePieceTextEncoder(text_encoder)

    model = BertForMaskedLM.from_pretrained(bert_model)
    model.bert_text_encoder = bert_text_encoder
    model.bert.embeddings.word_embeddings = nn.Embedding(
        bert_text_encoder.vocab_size, model.bert.embeddings.word_embeddings.weight.shape[1])
    model.config.vocab_size = bert_text_encoder.vocab_size
    model.cls = BertOnlyMLMHead(
        model.config, model.bert.embeddings.word_embeddings.weight)

    model.load_state_dict(torch.load(path))

    return model


class BertEmbeddingPredictor(nn.Module):
    def __init__(self, bert_model, text_encoder, path):
        super(BertEmbeddingPredictor, self).__init__()
        self.model = load_fine_tuned_model(bert_model, text_encoder, path)

    def forward(self, labels):
        # do not modify this
        self.eval()
        return generate_embedding(self.model, labels)
