from .expert import UpstreamExpert


def ae_lr5e_4_mlp_latent(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_mlp/best_valid_loss--0.675.ckpt",
        "latent",
    )


def ae_lr5e_4_mlp_reconstruct(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_mlp/best_valid_loss--0.675.ckpt",
        "reconstruct",
    )


def ae_lr5e_4_1transformer_latent128(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer_latent128/best_valid_loss--1.379.ckpt",
        "latent",
    )


def ae_lr5e_4_1transformer_latent64(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer_latent64/best_valid_loss--1.176.ckpt",
        "latent",
    )


def ae_lr5e_4_1transformer_latent32(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer_latent32/best_valid_loss--1.049.ckpt",
        "latent",
    )


def ae_lr5e_4_1transformer_latent16(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer_latent16/best_valid_loss--0.922.ckpt",
        "latent",
    )
