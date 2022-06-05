from .base import SequentialDataPipe
from .common_pipes import ApplySoxEffectOnFile, SetOutputKeys


class DumpFeaturePipe(SequentialDataPipe):
    def __init__(
        self,
        output_keys: dict = None,
        effects: list = None,
    ):
        output_keys = output_keys or dict(
            x="wav",
            x_len="wav_len",
            unique_name="id",
        )

        super().__init__(
            ApplySoxEffectOnFile(effects=effects),
            SetOutputKeys(output_keys=output_keys),
        )
