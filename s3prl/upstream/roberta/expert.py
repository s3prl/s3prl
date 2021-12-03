import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.models.roberta import RobertaModel
from ..interfaces import UpstreamBase

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


class UpstreamExpert(UpstreamBase):
    """
    The expert of RoBERTa
    """

    def __init__(
        self,
        frontend_model,
        model_name_or_path="./bert_kmeans/",
        checkpoint_file="bert_kmeans.pt",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.frontend_model = frontend_model
        self.roberta = RobertaModel.from_pretrained(model_name_or_path, checkpoint_file)
        self.max_positions = self.roberta.cfg.model.max_positions
    
    def get_downsample_rates(self, key: str):
        return 160

    def forward(self, wavs):
        with torch.no_grad():
            self.frontend_model.eval()
            strings = self.frontend_model(wavs)

        tokens = [
            self.roberta.task.source_dictionary.encode_line(
                string, append_eos=False, add_if_not_exist=False
            ).long()
            for string in strings
        ]
        tokens = pad_sequence(
            tokens,
            batch_first=True,
            padding_value=self.roberta.task.source_dictionary.pad(),
        ).to(wavs[0].device)

        bucket_layer_results = []
        for start in range(0, tokens.size(-1), self.max_positions):
            bucket_layer_result = self.roberta.extract_features(tokens[:, start : start + self.max_positions], return_all_hiddens=True)
            bucket_layer_results.append(bucket_layer_result)

        layer_bucket_results = [
            [bucket_layer_results[bucket_id][layer_id] for bucket_id in range(len(bucket_layer_results))]
            for layer_id in range(len(bucket_layer_results[0]))
        ]
        layer_results = [torch.cat(layer_buckets, dim=1) for layer_buckets in layer_bucket_results]

        return {
            "hidden_states": layer_results,
            "last_hidden_state": layer_results[-1],
        }
