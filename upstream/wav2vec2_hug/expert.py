import torch
import torch.nn as nn

from transformers import Wav2Vec2Processor, Wav2Vec2Model

SAMPLE_RATE = 16000


class UpstreamExpert(nn.Module):
    def __init__(
        self,
        ckpt: str = None,
        model_config: str = None,
        feature_selection: str = None,
        **kwargs
    ):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.

            feature_selection:
                The string for you to control the different behavior of the
                same pretrained model, like extracting different layers as
                the representations.
        """
        super().__init__()

        self.processor = Wav2Vec2Processor.from_pretrained(ckpt)
        self.model = Wav2Vec2Model.from_pretrained(ckpt)

        pseudo_input = [torch.randn(SAMPLE_RATE)]
        pseudo_output = self.forward(pseudo_input)
        self._output_dim = pseudo_output[0].size(-1)

    # Interface
    def get_downsample_rate(self):
        # 160 means 10 ms per frame for 16000 Hz waveforms
        return 320

    # Interface
    def get_output_dim(self):
        return self._output_dim

    # Interface
    def forward(self, wavs: [torch.FloatTensor]):
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs])
        processor_outputs = self.processor(
            [wav.cpu().numpy() for wav in wavs],
            return_tensors="pt",
            sampling_rate=SAMPLE_RATE,
            padding="longest",
        )
        model_outputs = self.model(
            processor_outputs.input_values.to(device),
            attention_mask=processor_outputs.get("attention_mask", None),
            output_hidden_states=True,
        )
        return model_outputs.last_hidden_state
