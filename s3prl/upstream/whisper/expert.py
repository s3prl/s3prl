import logging
from functools import partial

import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WhisperModel

from s3prl.hub import wavlm_base_plus

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DOWNSAMPLE_RATE = 320
WAVLM_BASE_NUM_LAYER = 13
WAVLM_BASE_HIDDEN_SIZE = 768
WHISPER_BASE_NUM_LAYER = 7
WHISPER_HIDDEN_SIZE = 512


def build_mlp(input_size: int, hidden_size: int, output_size: int):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size),
    )


class UpstreamExpert(torch.nn.Module):
    def __init__(
        self,
        project_model: str,
        project_size: int,
        use_wavlm: bool,
        use_whisper: bool,
        extract_mels: bool,
        use_noisy_mels: bool,
        use_clean_mels: bool,
        whisper_mel: str = "noisy",
        noise_disentanglement: float = 1.0,
        **kwds,
    ):
        super().__init__()
        assert use_wavlm or use_whisper
        self.whisper_mel = whisper_mel
        self.extract_mels = extract_mels
        self.use_noisy_mels = use_noisy_mels
        self.use_clean_mels = use_clean_mels
        logger.info("Setup Whisper Mel extractor")
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(
            "openai/whisper-base"
        )

        if project_model == "MLP":
            build_model = partial(
                build_mlp, hidden_size=project_size, output_size=project_size
            )
        else:
            raise ValueError

        self.use_wavlm = use_wavlm
        if use_wavlm:
            logger.info("Setup WavLM")
            self.wavlm = wavlm_base_plus()
            self.wavlm_projects = torch.nn.ModuleList(
                [
                    build_model(WAVLM_BASE_HIDDEN_SIZE)
                    for _ in range(WAVLM_BASE_NUM_LAYER)
                ]
            )

        self.use_whisper = use_whisper
        if use_whisper:
            logger.info("Setup Whisper")
            self.whisper_model = WhisperModel.from_pretrained("openai/whisper-base")
            self.whisper_projects = torch.nn.ModuleList(
                [
                    build_model(WHISPER_HIDDEN_SIZE)
                    for _ in range(WHISPER_BASE_NUM_LAYER)
                ]
            )

        self.noise_disentanglement = noise_disentanglement
        if noise_disentanglement > 0:
            self.clean_whisper_projects = torch.nn.ModuleList(
                [
                    build_model(WHISPER_HIDDEN_SIZE)
                    for _ in range(WHISPER_BASE_NUM_LAYER)
                ]
            )
            self.noise_disentanglement_l1 = torch.nn.L1Loss()

        self.register_buffer("device_detector", torch.zeros(1))

    def get_downsample_rates(self, key: str) -> int:
        return DOWNSAMPLE_RATE

    def forward(self, all_upstream_inputs):
        device = self.device_detector.device

        if isinstance(all_upstream_inputs, (list, tuple)):
            wavs = [wav.to(device) for wav in all_upstream_inputs]
            noisy_wavs = wavs
            clean_wavs = wavs
            noisy_mels = self.whisper_feature_extractor(
                [wav.cpu().numpy() for wav in noisy_wavs],
                return_tensors="pt",
                sampling_rate=SAMPLE_RATE,
                do_normalize=True,
            ).input_features.to(device)
            clean_mels = self.whisper_feature_extractor(
                [wav.cpu().numpy() for wav in clean_wavs],
                return_tensors="pt",
                sampling_rate=SAMPLE_RATE,
                do_normalize=True,
            ).input_features.to(device)
        else:

            def to_device(target, device):
                if target is None:
                    return None

                if isinstance(target, (list, tuple)):
                    return [torch.FloatTensor(t).to(device) for t in target]
                else:
                    return torch.FloatTensor(target).to(device)

            noisy_wavs = to_device(all_upstream_inputs["noisy_wavs"], device)
            clean_wavs = to_device(all_upstream_inputs["clean_wavs"], device)
            noisy_mels = to_device(all_upstream_inputs["noisy_mels"], device)
            clean_mels = to_device(all_upstream_inputs["clean_mels"], device)

            if self.extract_mels:
                if self.use_noisy_mels:
                    wavs_for_whisper = all_upstream_inputs["noisy_wavs"]
                    noisy_mels = self.whisper_feature_extractor(
                        wavs_for_whisper,
                        return_tensors="pt",
                        sampling_rate=SAMPLE_RATE,
                        do_normalize=True,
                    ).input_features

                if self.use_clean_mels:
                    wavs_for_whisper = all_upstream_inputs["clean_wavs"]
                    clean_mels = self.whisper_feature_extractor(
                        wavs_for_whisper,
                        return_tensors="pt",
                        sampling_rate=SAMPLE_RATE,
                        do_normalize=True,
                    ).input_features

        wavs_len = [len(wav) for wav in noisy_wavs]
        max_seq_len = len(list(range(0, max(wavs_len), DOWNSAMPLE_RATE)))

        wavlm_projected_hs = []
        if self.use_wavlm:
            with torch.no_grad():
                self.wavlm.eval()
                wavlm_hs = self.wavlm(noisy_wavs)["hidden_states"]

            wavlm_hs = [
                F.layer_norm(hs[:, :max_seq_len, :], hs.shape[-1:]) for hs in wavlm_hs
            ]
            for hs, project in zip(wavlm_hs, self.wavlm_projects):
                projected_hs = project(hs)
                wavlm_projected_hs.append(projected_hs)

        whisper_projected_hs = []
        if self.use_whisper:
            if self.whisper_mel == "noisy":
                whisper_mels = noisy_mels
            elif self.whisper_mel == "clean":
                whisper_mels = clean_mels

            decoder_input_ids = (
                torch.tensor([[1]]) * self.whisper_model.config.decoder_start_token_id
            )
            with torch.no_grad():
                self.whisper_model.eval()
                result = self.whisper_model(
                    whisper_mels.to(device),
                    decoder_input_ids=decoder_input_ids.to(device),
                    output_hidden_states=True,
                )
            whisper_hs = result.encoder_hidden_states
            whisper_hs = [
                F.layer_norm(hs[:, :max_seq_len, :], hs.shape[-1:]) for hs in whisper_hs
            ]
            for hs, project in zip(whisper_hs, self.whisper_projects):
                projected_hs = project(hs)
                whisper_projected_hs.append(projected_hs)

        if self.noise_disentanglement > 0:
            clean_whisper_projected_hs = []
            with torch.no_grad():
                self.whisper_model.eval()
                result = self.whisper_model(
                    clean_mels.to(device),
                    decoder_input_ids=decoder_input_ids.to(device),
                    output_hidden_states=True,
                )
            clean_whisper_hs = result.encoder_hidden_states
            clean_whisper_hs = [
                F.layer_norm(hs[:, :max_seq_len, :], hs.shape[-1:])
                for hs in clean_whisper_hs
            ]
            for hs, project in zip(clean_whisper_hs, self.clean_whisper_projects):
                projected_hs = project(hs)
                clean_whisper_projected_hs.append(projected_hs)

            noisy_whisper_hs = torch.stack(whisper_projected_hs, dim=0).mean(dim=0)
            clean_whisper_hs = torch.stack(clean_whisper_projected_hs, dim=0).mean(
                dim=0
            )
            noise_disentangle_loss = self.noise_disentanglement_l1(
                noisy_whisper_hs, clean_whisper_hs
            )
            scaled_noise_disentangle_loss = (
                self.noise_disentanglement * noise_disentangle_loss
            )

        if self.use_wavlm and self.use_whisper:
            hidden_states = []
            for wavlm_hs, whisper_hs in zip(wavlm_projected_hs, whisper_projected_hs):
                min_len = min(wavlm_hs.size(1), whisper_hs.size(1))
                wavlm_hs = wavlm_hs[:, :min_len, :]
                whisper_hs = whisper_hs[:, :min_len, :]
                hidden_states = [*hidden_states, wavlm_hs, whisper_hs]
                hidden_state = torch.stack(hidden_states, dim=0).mean(dim=0)

        elif self.use_wavlm:
            hidden_state = torch.stack(wavlm_projected_hs, dim=0).mean(dim=0)

        elif self.use_whisper:
            hidden_state = torch.stack(whisper_projected_hs, dim=0).mean(dim=0)

        results = {
            "hidden_states": [hidden_state],
        }
        if self.noise_disentanglement > 0:
            results.update(
                {
                    "noise_disentangle_loss": noise_disentangle_loss,
                    "scaled_noise_disentangle_loss": scaled_noise_disentangle_loss,
                }
            )

        return results
