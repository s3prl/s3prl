import sys
from typing import Callable, List, Dict, Tuple, Union

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from s3prl.utility.helper import show

SAMPLE_RATE = 16000
TOLERABLE_SEQLEN_DIFF = 5


class Hook:
    def __init__(self, module_path, transform, unique_identifier=None):
        self.module_path = module_path
        self.transform = transform
        self.unique_identifier = unique_identifier or module_path
        self.handler = None

        assert isinstance(self.module_path, str)
        assert callable(self.transform)
        assert isinstance(self.unique_identifier, str)


class initHook(type):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        for hook in instance.hooks:
            if hook.handler is None:
                instance._register_hook_handler(hook)
        return instance


class UpstreamBase(nn.Module, metaclass=initHook):
    def __init__(
        self,
        hooks: List[Tuple] = None,
        hook_postprocess: Callable[
            [List[Tuple[str, Tensor]]], List[Tuple[str, Tensor]]
        ] = None,
        **kwargs,
    ):
        """
        Args:
            hooks: each Tuple is an argument list for the Hook initializer
        """
        super().__init__()
        self.hooks: List[Hook] = [Hook(*hook) for hook in hooks] if hooks else []
        self.hook_postprocess = hook_postprocess
        self._hook_hiddens: List[Tuple(str, Tensor)] = []

    def remove_all_hooks(self):
        for hook in self.hooks:
            hook.handler.remove()
        self.hooks.clear()

    def remove_hook(self, unique_identifier: str):
        updated_hooks = []
        for hook in self.hooks:
            if hook.unique_identifier == unique_identifier:
                hook.handler.remove()
            else:
                updated_hooks.append(hook)
        self.hooks = updated_hooks

    def add_hook(self, *args, **kwargs):
        hook = Hook(*args, **kwargs)
        self._register_hook_handler(hook)
        self.hooks.append(hook)

    def _register_hook_handler(self, hook: Hook):
        module = eval(hook.module_path)
        if not isinstance(module, nn.Module):
            show(
                f"[UpstreamBase] - {hook.module_path} is not a valid nn.Module. Skip.",
                file=sys.stderr,
            )
            return

        if callable(hook.handler):
            show(
                f"[UpstreamBase] - Existing hook handler for {hook.unique_identifier} is found. Remove the existing one.",
                file=sys.stderr,
            )
            hook.handler.remove()

        def generate_hook_handler(hiddens: List, hook: Hook):
            def hook_handler(self, input, output):
                hiddens.append((hook.unique_identifier, hook.transform(input, output)))

            return hook_handler

        hook.handler = module.register_forward_hook(
            generate_hook_handler(self._hook_hiddens, hook)
        )

    def __call__(self, wavs: List[Tensor], *args, **kwargs):
        self._hook_hiddens.clear()

        result = super().__call__(wavs, *args, **kwargs) or {}
        assert isinstance(result, dict)

        if len(self._hook_hiddens) > 0:
            if (
                result.get("_hidden_states_info") is not None
                or result.get("hidden_states") is not None
                or result.get("last_hidden_state") is not None
            ):
                show(
                    "[UpstreamBase] - If there are registered hooks, '_hidden_states_info', 'hidden_states', and "
                    "'last_hidden_state' are reserved and should not be included in child class's return dict.",
                    file=sys.stderr,
                )
                raise ValueError

            hook_hiddens = self._hook_hiddens.copy()
            self._hook_hiddens.clear()

            if callable(self.hook_postprocess):
                hook_hiddens = self.hook_postprocess(hook_hiddens)

            result["_hidden_states_info"], result["hidden_states"] = zip(*hook_hiddens)
            result["last_hidden_state"] = result["hidden_states"][-1]

            for layer_id, hidden_state in enumerate(result["hidden_states"]):
                result[f"hidden_state_{layer_id}"] = hidden_state

        return result


class Featurizer(nn.Module):
    def __init__(
        self,
        upstream: UpstreamBase,
        feature_selection: str = "hidden_states",
        upstream_device: str = "cuda",
        layer_selection: int = None,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.name = "Featurizer"

        upstream.eval()
        paired_wavs = [torch.randn(SAMPLE_RATE).to(upstream_device)]
        with torch.no_grad():
            paired_features = upstream(paired_wavs)

        if feature_selection not in paired_features:
            if "hidden_states" in paired_features:
                show(
                    f"[{self.name}] - Warning: {feature_selection} is not a supported args.upstream_feature_selection."
                    f" Using \"hidden_states\" as the default key.",
                    file=sys.stderr
                )
                feature_selection = "hidden_states"
            else:
                show(
                    f"[{self.name}] - Error: {feature_selection} is not a supported args.upstream_feature_selection."
                    f" The default key \"hidden_states\" is also not supported."
                    f" Please specify -s with the following options: {list(paired_wavs.keys())}",
                    file=sys.stderr
                )
                raise ValueError
        self.feature_selection = feature_selection
        self.layer_selection = layer_selection
        self.normalize = normalize

        feature = self._select_feature(paired_features)
        if isinstance(feature, (list, tuple)):
            self.layer_num = len(feature)
            show(
                f"[{self.name}] - Take a list of {self.layer_num} features and weighted sum them.",
                file=sys.stderr
            )
            self.weights = nn.Parameter(torch.zeros(self.layer_num))
            feature = self._weighted_sum([f.cpu() for f in feature])
        else:
            feature = feature.cpu()

        self.output_dim = feature.size(-1)
        if hasattr(upstream, "get_downsample_rates"):
            self.downsample_rate = upstream.get_downsample_rates(feature_selection)
            show(
                f"[{self.name}] - The selected feature {feature_selection}'s downsample rate is {self.downsample_rate}",
                file=sys.stderr
            )
        else:
            self.downsample_rate = round(max(len(wav) for wav in paired_wavs) / feature.size(1))
            show(
                f"[{self.name}] - Warning: The provided upstream does not give statis downsample rate"
                " by the \"get_downsample_rates\" interface (see upstream/example/expert.py)."
                " The downsample rate is calculated dynamically basing on the shape of the"
                f" input waveforms v.s. the output features: {self.downsample_rate}",
                file=sys.stderr
            )

    def _select_feature(self, features):
        feature = features.get(self.feature_selection)

        if isinstance(feature, dict):
            feature = list(feature.values())

        if isinstance(feature, (list, tuple)) and len(feature) == 1:
            feature = feature[0]
        
        if isinstance(feature, (list, tuple)) and isinstance(self.layer_selection, int):
            feature = feature[self.layer_selection]

        return feature

    def _weighted_sum(self, feature):
        assert self.layer_num == len(feature), (
            "If you run into this error, there is a great chance"
            " you are finetuning the upstream with wav2vec2's transformer blocks"
            " in weighted-sum mode (default), including wav2vec2, hubert, and decoar2."
            " These models use the layerdrop technique which causes the different number"
            " of layer forwards between different model forwards, resulting in different"
            " number of hidden states for different model forwards. Hence, finetuning"
            " these upstreams is essentially incompatible with weight-sum mode unless"
            " you turn off the layerdrop option in fairseq. See:"
            " https://github.com/pytorch/fairseq/blob/f6abcc2a67328bee8b15c596bb626ce2d720aae6/fairseq/models/wav2vec/wav2vec2.py#L857"
            " However, since finetuning upstreams will backward the gradient through all layers"
            " which serves the same functionality as weighted-sum: all layers can be used for different"
            " downstream tasks. Hence instead of finetuning upstream with weighted-sum, we suggest to"
            " follow the more common setting: finetuning upstream with the last layer. Please use the"
            " following options: --upstream_trainable --upstream_feature_selection last_hidden_state."
            " Or: -f -s last_hidden_state"
        )
        stacked_feature = torch.stack(feature, dim=0)

        if self.normalize:
            stacked_feature = F.layer_norm(
                stacked_feature, (stacked_feature.shape[-1],))

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature

    def tolist(self, paired_wavs: List[Tensor], paired_feature: Tensor):
        assert paired_feature.dim() == 3, "(batch_size, max_seq_len, feat_dim)"
        feature_len = [round(len(wav) / self.downsample_rate) for wav in paired_wavs]
        assert abs(paired_feature.size(1) - round(max([len(wav) for wav in paired_wavs]) / self.downsample_rate)) < TOLERABLE_SEQLEN_DIFF
        feature = [f[:l] for f, l in zip(paired_feature, feature_len)]
        return feature

    def forward(
        self,
        paired_wavs: List[Tensor],
        paired_features: Dict[str, Union[Tensor, List[Tensor], Dict[str, Tensor]]],
    ):
        feature = self._select_feature(paired_features)
        if isinstance(feature, (list, tuple)):
            feature = self._weighted_sum(feature)

        return self.tolist(paired_wavs, feature)
