from .expert import UpstreamExpert


def spin_hubert_2048_kmeans_500(refresh=False, **kwargs):
    kwargs["spin_name"] = "spin_hubert_2048"
    kwargs["km_path"] = (
        "/home/leo1994122701/cslm/cst/result/km_models/spin_hubert_2048_km500"  # TODO: to url
    )
    return UpstreamExpert(**kwargs)


def spin_hubert_2048_kmeans_1000(refresh=False, **kwargs):
    kwargs["spin_name"] = "spin_hubert_2048"
    kwargs["km_path"] = (
        "/home/leo1994122701/cslm/cst/result/km_models/spin_hubert_2048_km1000"  # TODO: to url
    )
    return UpstreamExpert(**kwargs)


def spin_hubert_2048_kmeans_1000_stack2(refresh=False, **kwargs):
    kwargs["spin_name"] = "spin_hubert_2048"
    kwargs["km_path"] = (
        "/home/leo1994122701/cslm/cst/result/km_models/spin_hubert_2048_km1000"  # TODO: to url
    )
    kwargs["stack_num"] = 2
    return UpstreamExpert(**kwargs)


def spin_hubert_2048_kmeans_1000_stack4(refresh=False, **kwargs):
    kwargs["spin_name"] = "spin_hubert_2048"
    kwargs["km_path"] = (
        "/home/leo1994122701/cslm/cst/result/km_models/spin_hubert_2048_km1000"  # TODO: to url
    )
    kwargs["stack_num"] = 4
    return UpstreamExpert(**kwargs)


def spin_hubert_2048_kmeans_1000_stack8(refresh=False, **kwargs):
    kwargs["spin_name"] = "spin_hubert_2048"
    kwargs["km_path"] = (
        "/home/leo1994122701/cslm/cst/result/km_models/spin_hubert_2048_km1000"  # TODO: to url
    )
    kwargs["stack_num"] = 8
    return UpstreamExpert(**kwargs)
