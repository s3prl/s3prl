import logging

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from s3prl.utility.helper import zero_mean_unit_var_norm

from ..interfaces import UpstreamBase
from .convert import load_converted_model

logger = logging.getLogger(__name__)

from ..interfaces import UpstreamBase
from .convert import load_converted_model

logger = logging.getLogger(__name__)


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, feature_selection: str = None, **kwargs):
        super().__init__(**kwargs)
        model, task_cfg = load_converted_model(ckpt)
        self.model = model
        self.wav_normalize = task_cfg.normalize

        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

        # These options are only used for aligning representations between s3prl and huggingface
        # See utility/compare_wav2vec2.py
        self.apply_padding_mask = True
        self.numpy_wav_normalize = False

        assert feature_selection is None or feature_selection in [
            "fairseq_layers",
            "fairseq_layers_before_residual",
        ]
        self.feature_selection = feature_selection

        if feature_selection is None:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        if self.wav_normalize:
            if self.numpy_wav_normalize:
                wavs = zero_mean_unit_var_norm([wav.cpu().numpy() for wav in wavs])
                wavs = [torch.from_numpy(wav).to(device) for wav in wavs]
            else:
                wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        results = self.model.extract_features(
            padded_wav, wav_padding_mask if self.apply_padding_mask else None
        )

        if self.feature_selection is not None:
            if self.feature_selection == "fairseq_layers":
                return {
                    "hidden_states": [
                        h[0].transpose(0, 1) for h in results["layer_results"]
                    ],
                }
            elif self.feature_selection == "fairseq_layers_before_residual":
                return {
                    "hidden_states": [
                        h[2].transpose(0, 1) for h in results["layer_results"]
                    ],
                }
        else:
            pass
            # This forward function only does the model forward
            # The return dict is then handled by UpstreamBase's hooks


class LegacyUpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        logger.warning("Use the legacy expert for wav2vec 2.0 which depends on fairseq")

        super().__init__(**kwargs)
        model, cfg, task = self.load_model(ckpt)
        self.model = model[0]
        self.wav_normalize = cfg.task.normalize

        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

        # These options are only used for aligning representations between s3prl and huggingface
        # See utility/compare_wav2vec2.py
        self.apply_padding_mask = True
        self.numpy_wav_normalize = False

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess

    @staticmethod
    def load_model(ckpt_path: str):
        """
        Sanitize the config in the checkpoint as there are some irrelevant fields
        in the released checkpoint which can cause the model loading to fail
        """
        import dataclasses

        import fairseq
        import omegaconf
        from fairseq.tasks.audio_pretraining import AudioPretrainingConfig

        ckpt_state = torch.load(ckpt_path, map_location="cpu")

        def fix_cfg(cfg):
            for key in list(cfg.keys()):
                if key not in ["task", "model"]:
                    cfg.pop(key)

            fields_pretraining = [
                field.name for field in dataclasses.fields(AudioPretrainingConfig)
            ]
            for key in list(cfg["task"].keys()):
                if key not in fields_pretraining:
                    cfg["task"].pop(key)

        if "cfg" in ckpt_state:
            cfg = ckpt_state["cfg"]
            if isinstance(cfg, omegaconf.DictConfig):
                with omegaconf.open_dict(cfg):
                    fix_cfg(cfg)
            else:
                fix_cfg(cfg)

            if not isinstance(cfg, omegaconf.DictConfig):
                cfg = omegaconf.OmegaConf.create(cfg)
            ckpt_state["cfg"] = cfg

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path], state=ckpt_state
        )
        return model, cfg, task

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        if self.wav_normalize:
            if self.numpy_wav_normalize:
                wavs = zero_mean_unit_var_norm([wav.cpu().numpy() for wav in wavs])
                wavs = [torch.from_numpy(wav).to(device) for wav in wavs]
            else:
                wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        results = self.model.extract_features(
            padded_wav, wav_padding_mask if self.apply_padding_mask else None
        )

        # This forward function only does the model forward
        # The return dict is then handled by UpstreamBase's hooks
