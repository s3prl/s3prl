import torch
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .convert import load_converted_model

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


class UpstreamExpert(UpstreamBase):
    """
    The expert of RoBERTa
    """

    def __init__(
        self,
        ckpt: str,
        frontend_model: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.frontend_model = frontend_model
        self.roberta, task_cfg = load_converted_model(ckpt)
        self.dictionary = self.roberta.encoder.dictionary
        self.max_positions = self.roberta.max_positions()

    def get_downsample_rates(self, key: str):
        return 160

    def extract_features(
        self, tokens: torch.LongTensor, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        device = tokens.device
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.roberta.max_positions():
            raise ValueError(
                "tokens exceeds maximum length: {} > {}".format(
                    tokens.size(-1), self.roberta.max_positions()
                )
            )
        features, extra = self.roberta(
            tokens.to(device=device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def forward(self, wavs):
        with torch.no_grad():
            self.frontend_model.eval()
            strings = self.frontend_model(wavs)

        tokens = [
            self.dictionary.encode_line(
                string, append_eos=False, add_if_not_exist=False
            ).long()
            for string in strings
        ]
        tokens = pad_sequence(
            tokens,
            batch_first=True,
            padding_value=self.dictionary.pad(),
        ).to(wavs[0].device)

        bucket_layer_results = []
        for start in range(0, tokens.size(-1), self.max_positions):
            bucket_layer_result = self.extract_features(
                tokens[:, start : start + self.max_positions], return_all_hiddens=True
            )
            bucket_layer_results.append(bucket_layer_result)

        layer_bucket_results = [
            [
                bucket_layer_results[bucket_id][layer_id]
                for bucket_id in range(len(bucket_layer_results))
            ]
            for layer_id in range(len(bucket_layer_results[0]))
        ]
        layer_results = [
            torch.cat(layer_buckets, dim=1) for layer_buckets in layer_bucket_results
        ]

        return {
            "hidden_states": layer_results,
            "last_hidden_state": layer_results[-1],
        }


class LegacyUpstreamExpert(UpstreamBase):
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
        from fairseq.models.roberta import RobertaModel

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
            bucket_layer_result = self.roberta.extract_features(
                tokens[:, start : start + self.max_positions], return_all_hiddens=True
            )
            bucket_layer_results.append(bucket_layer_result)

        layer_bucket_results = [
            [
                bucket_layer_results[bucket_id][layer_id]
                for bucket_id in range(len(bucket_layer_results))
            ]
            for layer_id in range(len(bucket_layer_results[0]))
        ]
        layer_results = [
            torch.cat(layer_buckets, dim=1) for layer_buckets in layer_bucket_results
        ]

        return {
            "hidden_states": layer_results,
            "last_hidden_state": layer_results[-1],
        }
