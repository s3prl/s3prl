from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def espnet_hubert_custom(ckpt, *args, config=None, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)


def espnet_hubert_local(*args, **kwargs):
    return espnet_hubert_custom(*args, **kwargs)


def cvhubert(*args, refresh=False, **kwargs):
    url = "https://huggingface.co/espnet/espnet_cvhubert/resolve/main/exp/hubert_iter2_train_ssl_torchaudiohubert_base_960h_pretrain_it2_raw/latest.pth"
    config_url = "https://huggingface.co/espnet/espnet_cvhubert/raw/main/exp/hubert_iter2_train_ssl_torchaudiohubert_base_960h_pretrain_it2_raw/config.yaml"
    ckpt, config = _urls_to_filepaths(url, config_url, refresh=refresh)
    return espnet_hubert_custom(ckpt, config)


def wavlablm_ek_40k(*args, refresh=False, **kwargs):
    url = "https://huggingface.co/espnet/WavLabLM-EK-40k/resolve/main/exp_li/hubert_iter2_train_ssl_torchaudiohubert_large_960h_pretrain_it2_cont_raw_layer_9/5epoch.pth"
    config_url = "https://huggingface.co/espnet/WavLabLM-EK-40k/raw/main/exp_li/hubert_iter2_train_ssl_torchaudiohubert_large_960h_pretrain_it2_cont_raw_layer_9/config.yaml"
    ckpt, config = _urls_to_filepaths(url, config_url, refresh=refresh)
    return espnet_hubert_custom(ckpt, config)


def wavlablm_ms_40k(*args, refresh=False, **kwargs):
    url = "https://huggingface.co/espnet/WavLabLM-MS-40k/resolve/main/exp_babel/hubert_iter2_train_ssl_torchaudiohubert_large_960h_pretrain_it2_wavlm_babel_light_raw_layer_9/5epoch.pth"
    config_url = "https://huggingface.co/espnet/WavLabLM-MS-40k/raw/main/exp_babel/hubert_iter2_train_ssl_torchaudiohubert_large_960h_pretrain_it2_wavlm_babel_light_raw_layer_9/config.yaml"
    ckpt, config = _urls_to_filepaths(url, config_url, refresh=refresh)
    return espnet_hubert_custom(ckpt, config)


def wavlablm_mk_40k(*args, refresh=False, **kwargs):
    url = "https://huggingface.co/espnet/WavLabLM-MK-40k/resolve/main/exp_li/hubert_iter2_train_ssl_torchaudiohubert_large_960h_pretrain_it2_wavlm_raw_layer_9/valid.acc_m.ave_10best.pth"
    config_url = "https://huggingface.co/espnet/WavLabLM-MK-40k/raw/main/exp_li/hubert_iter2_train_ssl_torchaudiohubert_large_960h_pretrain_it2_wavlm_raw_layer_9/config.yaml"
    ckpt, config = _urls_to_filepaths(url, config_url, refresh=refresh)
    return espnet_hubert_custom(ckpt, config)


def espnet_hubert_base_iter1(*args, refresh=False, **kwargs):
    url = "https://huggingface.co/espnet/simpleoier_librispeech_hubert_iter1_train_ssl_torchaudiohubert_base_960h_pretrain_it1_raw/resolve/main/exp/hubert_iter1_train_ssl_torchaudiohubert_base_960h_pretrain_it1_raw/valid.loss.ave.pth"
    config_url = "https://huggingface.co/espnet/simpleoier_librispeech_hubert_iter1_train_ssl_torchaudiohubert_base_960h_pretrain_it1_raw/raw/main/exp/hubert_iter1_train_ssl_torchaudiohubert_base_960h_pretrain_it1_raw/config.yaml"
    ckpt, config = _urls_to_filepaths(url, config_url, refresh=refresh)
    return espnet_hubert_custom(ckpt, config)


def espnet_hubert_base_iter0(*args, refresh=False, **kwargs):
    url = "https://huggingface.co/espnet/simpleoier_librispeech_hubert_iter0_train_ssl_torchaudiohubert_base_960h_pretrain_it0_raw/resolve/main/exp/hubert_iter0_train_ssl_torchaudiohubert_base_960h_pretrain_it0_raw/valid.loss.ave.pth"
    config_url = "https://huggingface.co/espnet/simpleoier_librispeech_hubert_iter0_train_ssl_torchaudiohubert_base_960h_pretrain_it0_raw/raw/main/exp/hubert_iter0_train_ssl_torchaudiohubert_base_960h_pretrain_it0_raw/config.yaml"
    ckpt, config = _urls_to_filepaths(url, config_url, refresh=refresh)
    return espnet_hubert_custom(ckpt, config)


def espnet_hubert_large_gs_ll60k(*args, refresh=False, **kwargs):
    url = "https://huggingface.co/espnet/hubert_large_gs_16_librilight60k/resolve/main/mnt/datastore/exp/hubert_iter1_train_ssl_torchaudiohubert_large_960h_pretrain_it2_bins_raw/valid.loss.ave_10best.pth"
    config_url = "https://huggingface.co/espnet/hubert_large_gs_16_librilight60k/blob/main/mnt/datastore/exp/hubert_iter1_train_ssl_torchaudiohubert_large_960h_pretrain_it2_bins_raw/config.yaml"
    ckpt, config = _urls_to_filepaths(url, config_url, refresh=refresh)
    return espnet_hubert_custom(ckpt, config)
