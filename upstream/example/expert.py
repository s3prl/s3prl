import torch
import torch.nn as nn

HIDDEN_DIM = 512
FEATURE_DIM = 256
FEATURE_SEQ_LEN = 100


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

        # The model needs to be a nn.Module for finetuning
        # not required for representation extraction
        self.model = nn.Linear(HIDDEN_DIM, FEATURE_DIM)

    # Interface
    def get_downsample_rate(self):
        # 160 means 10 ms per frame for 16000 Hz waveforms
        return 160

    # Interface
    def get_output_dim(self):
        return FEATURE_DIM

    # Interface
    def forward(self, wavs: [torch.FloatTensor]):
        def get_feature(wav):
            # This is get an example code so random hidden states is generated
            hidden = torch.rand(FEATURE_SEQ_LEN, HIDDEN_DIM).to(wav.device)
            feature = self.model(hidden)
            return feature

        return [get_feature(wav) for wav in wavs]
