import sys
from collections import OrderedDict
from typing import Callable, List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.tensor import Tensor

from utility.helper import show

SAMPLE_RATE = 16000


class UpstreamBase(nn.Module):
    def __init__(
        self,
        wav_normalize: bool = False,
        hooks: Dict[str, Callable] = None,
        hook_postprocess: Callable = None,
        **kwargs,
    ):
        super().__init__()
        self.wav_normalize = wav_normalize
        self.eps = 1.0e-12

        self.hooks = hooks or {}
        self.hook_postprocess = hook_postprocess
        self._hook_handlers = {}
        self._hook_hiddens = OrderedDict()

    @staticmethod
    def tolist(paired_wavs: List[Tensor], paired_feature: Tensor):
        assert paired_feature.dim() == 3
        # (batch_size, max_seq_len, feat_dim)

        ratio = max([len(wav) for wav in paired_wavs]) / paired_feature.size(1)
        feature_len = [round(len(wav) / ratio) for wav in paired_wavs]
        feature = [f[:l] for f, l in zip(paired_feature, feature_len)]
        return feature

    def _sync_hook_handlers(self):
        for hook_handler in self._hook_handlers.values():
            hook_handler.remove()
        self._hook_handlers.clear()

        for hook_module_name, transform in self.hooks.items():
            module = eval(hook_module_name)
            if not isinstance(module, nn.Module):
                show(
                    f"[UpstreamBase] - {hook_module_name} is not a valid nn.Module. Skip.",
                    file=sys.stderr,
                )
            assert callable(transform)

            def hook_handler_template(hook_hiddens, hook_module_name, transform):
                def hook_handler(self, input, output):
                    hidden = transform(input, output)
                    hook_hiddens[hook_module_name] = hidden

                return hook_handler

            self._hook_handlers[hook_module_name] = module.register_forward_hook(
                hook_handler_template(self._hook_hiddens, hook_module_name, transform)
            )

    def __call__(self, wavs: List[Tensor], *args, **kwargs):
        if self.wav_normalize:
            wavs = [(x - x.mean()) / (x.std() + self.eps) for x in wavs]

        if self.hooks.keys() != self._hook_handlers.keys():
            self._sync_hook_handlers()

        result = super().__call__(wavs, *args, **kwargs) or {}
        assert isinstance(result, dict)

        if len(self._hook_hiddens) > 0:
            if (
                result.get("hidden_states") is not None
                or result.get("last_hidden_state") is not None
            ):
                show(
                    f"[UpstreamBase] - If there are registered hooks, 'hidden_states' and 'last_hidden_state'\
                    are reserved and should not be used in child class.",
                    file=sys.stderr,
                )
                raise ValueError

            hook_hiddens = self._hook_hiddens.copy()
            self._hook_hiddens.clear()

            if callable(self.hook_postprocess):
                hook_hiddens = self.hook_postprocess(hook_hiddens)

            result["hidden_states"] = hook_hiddens
            result["last_hidden_state"] = hook_hiddens[next(reversed(hook_hiddens))]

            default = result.get("default")
            if isinstance(default, Tensor):
                torch.allclose(default, result["last_hidden_state"])

        return result


class Featurizer(nn.Module):
    def __init__(
        self,
        upstream: UpstreamBase,
        feature_selection: str = "hidden_states",
        upstream_device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.feature_selection = feature_selection
        self.name = f"Featurizer for {upstream.__class__}"

        show(
            f"[{self.name}] - The input upstream is only for initialization and not saved in this nn.Module"
        )

        paired_wavs = [torch.randn(SAMPLE_RATE).to(upstream_device)]
        paired_features = upstream(paired_wavs)

        feature = self._select_feature(paired_features)
        if isinstance(feature, (list, tuple)):
            self.layer_num = len(feature)
            show(
                f"[{self.name}] - Take a list of {self.layer_num} features and weighted sum them."
            )
            self.weights = nn.Parameter(torch.ones(self.layer_num))
            feature = self._weighted_sum([f.cpu() for f in feature])
        else:
            feature = feature.cpu()

        self.output_dim = feature.size(-1)
        ratio = round(max(len(wav) for wav in paired_wavs) / feature.size(1))
        possible_rate = torch.LongTensor([160, 320])
        self.downsample_rate = int(
            possible_rate[(possible_rate - ratio).abs().argmin(dim=-1)]
        )

    def _select_feature(self, features):
        feature = features.get(self.feature_selection)

        if isinstance(feature, dict):
            feature = list(feature.values())

        if feature is None:
            show(
                f"[{self.name}] - feature_selection = {self.feature_selection} is not supported for this upstream.",
                file=sys.stderr,
            )
            show(
                f"[{self.name}] - Supported options: {features.keys()}",
                file=sys.stderr,
            )
            raise ValueError
        return feature

    def _weighted_sum(self, feature):
        assert self.layer_num == len(feature), f"{self.layer_num} != {len(feature)}"
        stacked_feature = torch.stack(feature, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

    def forward(
        self,
        paired_wavs: List[Tensor],
        paired_features: Dict[str, Union[Tensor, List[Tensor], Dict[str, Tensor]]],
    ):
        feature = self._select_feature(paired_features)
        if isinstance(feature, list) or isinstance(feature, tuple):
            feature = self._weighted_sum(feature)

        return UpstreamBase.tolist(paired_wavs, feature)
