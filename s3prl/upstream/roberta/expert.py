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

        if len(self.hooks) == 0:
            modules = "self.roberta.model.encoder.sentence_encoder.layers"
            for module_id, _ in enumerate(eval(modules)):
                self.add_hook(
                    f"{modules}[{module_id}]",
                    lambda input, output: output.transpose(0, 1),
                )

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
        features = self.roberta.extract_features(tokens)

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
