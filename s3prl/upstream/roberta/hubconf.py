import glob
import os

import torch

import s3prl
from s3prl.util.download import _urls_to_filepaths

from ..interfaces import Featurizer as _Featurizer
from .expert import LegacyUpstreamExpert as _LegacyUpstreamExpert
from .expert import UpstreamExpert as _UpstreamExpert


class _vq_wav2vec_codeids_wrapper(torch.nn.Module):
    def __init__(self, vq_wav2vec):
        super().__init__()
        self.vq_wav2vec = vq_wav2vec
        self.featurizer = _Featurizer(vq_wav2vec, "codeids", upstream_device="cpu")

    def _indices_to_string(self, sentence_idxs):
        return (
            "<s> "
            + " ".join("-".join(map(str, idx.tolist())) for idx in sentence_idxs)
            + " </s>"
        )

    def forward(self, wavs):
        batch_idxs = self.featurizer(wavs, self.vq_wav2vec(wavs))
        strings = [
            self._indices_to_string(sentence_idxs) for sentence_idxs in batch_idxs
        ]
        return strings


def _roberta_local(frontend_model, model_name_or_path, checkpoint_file, **kwargs):
    assert isinstance(frontend_model, torch.nn.Module)
    assert os.path.exists(model_name_or_path)
    return _LegacyUpstreamExpert(
        frontend_model, model_name_or_path, checkpoint_file, **kwargs
    )


def _vq_wav2vec_roberta(vq_wav2vec, **kwargs):
    frontend_model = _vq_wav2vec_codeids_wrapper(vq_wav2vec)
    return _roberta_local(frontend_model, **kwargs)


def vq_wav2vec_kmeans_roberta(refresh=False, legacy=False, **kwargs):
    if legacy:
        vq_wav2vec = getattr(s3prl.hub, f"vq_wav2vec_kmeans")(refresh=refresh)

        tar_file = _urls_to_filepaths(
            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/bert_kmeans.tar",
            refresh=refresh,
        )
        tar_dir = os.path.join(os.path.dirname(tar_file), "vq_wav2vec_kmeans_roberta/")
        os.makedirs(tar_dir, exist_ok=True)
        os.system(f"tar -xf {tar_file} -C {tar_dir}")

        pt_files = glob.glob(os.path.join(tar_dir, "*.pt"))
        assert len(pt_files) == 1
        pt_file = pt_files[0]

        kwargs["model_name_or_path"] = tar_dir
        kwargs["checkpoint_file"] = pt_file
        return _vq_wav2vec_roberta(vq_wav2vec, **kwargs)
    else:
        vq_wav2vec = getattr(s3prl.hub, f"vq_wav2vec_kmeans")()
        return _UpstreamExpert(
            _urls_to_filepaths(
                "https://huggingface.co/s3prl/converted_ckpts/resolve/main/vq_wav2vec_kmeans_roberta.pt",
                refresh=refresh,
            ),
            _vq_wav2vec_codeids_wrapper(vq_wav2vec),
        )


def discretebert(*args, legacy=False, **kwargs):
    return vq_wav2vec_kmeans_roberta(*args, legacy=legacy, **kwargs)
