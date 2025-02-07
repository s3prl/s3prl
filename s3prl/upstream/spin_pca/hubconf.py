from .expert import UpstreamExpert


def spin_hubert_2048_pca_384(refresh=False, **kwargs):
    kwargs["spin_name"] = "spin_hubert_2048"
    kwargs["pca_path"] = (
        "/home/leo1994122701/cslm/cst/result/pca_models/spin_hubert_2048_pca_384"  # TODO: to url
    )
    return UpstreamExpert(**kwargs)


def spin_hubert_2048_pca_16(refresh=False, **kwargs):
    kwargs["spin_name"] = "spin_hubert_2048"
    kwargs["pca_path"] = (
        "/home/leo1994122701/cslm/cst/result/pca_models/spin_hubert_2048_pca_16"  # TODO: to url
    )
    return UpstreamExpert(**kwargs)
