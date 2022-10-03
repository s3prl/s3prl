import logging

from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase

logger = logging.getLogger(__name__)


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, model_config, **kwargs):
        super().__init__(**kwargs)

        try:
            from pase.models.frontend import wf_builder

        except ModuleNotFoundError:
            logger.error(
                "Please check https://github.com/s3prl/s3prl/blob/master/s3prl/upstream/pase/README.md "
                "for how to install the dependencies of PASE+."
            )
            raise

        def build_pase(ckpt, model_config):
            pase = wf_builder(model_config)
            pase.load_pretrained(ckpt, load_last=True, verbose=False)
            return pase

        self.model = build_pase(ckpt, model_config)

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        wavs = pad_sequence(wavs, batch_first=True)
        wavs = wavs.unsqueeze(1)

        features = self.model(wavs)  # (batch_size, feature_dim, extracted_seqlen)
        features = features.transpose(
            1, 2
        ).contiguous()  # (batch_size, extracted_seqlen, feature_dim)

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
