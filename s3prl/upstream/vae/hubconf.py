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


def ae_lr5e_4_1transformer_latent384(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer/best_valid_loss--2.156.ckpt",
        "latent",
    )


def ae_lr5e_4_1transformer_latent384_stack4(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer/best_valid_loss--2.156.ckpt",
        "latent",
        stack=4,
    )


def ae_lr5e_4_1transformer_latent128(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer_latent128/best_valid_loss--1.379.ckpt",
        "latent",
    )


def ae_lr5e_4_1transformer_latent128_stack8(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer_latent128/best_valid_loss--1.379.ckpt",
        "latent",
        stack=8,
    )


def ae_lr5e_4_1transformer_latent128_reconstruct(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer_latent128/best_valid_loss--1.379.ckpt",
        "reconstruct",
    )


def ae_lr5e_4_1transformer_latent64(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer_latent64/best_valid_loss--1.176.ckpt",
        "latent",
    )


def ae_lr5e_4_1transformer_latent64_reconstruct(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer_latent64/best_valid_loss--1.176.ckpt",
        "reconstruct",
    )


def ae_lr5e_4_1transformer_latent32(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer_latent32/best_valid_loss--1.049.ckpt",
        "latent",
    )


def ae_lr5e_4_1transformer_latent32_reconstruct(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer_latent32/best_valid_loss--1.049.ckpt",
        "reconstruct",
    )


def ae_lr5e_4_1transformer_latent16(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer_latent16/best_valid_loss--0.922.ckpt",
        "latent",
    )


def ae_lr5e_4_1transformer_reconstruct(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_1transformer_latent16/best_valid_loss--0.922.ckpt",
        "reconstruct",
    )


def ae_lr5e_4_2transformer_latent16(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_2transformer_latent16/best_valid_loss--1.185.ckpt",
        "latent",
    )


def ae_lr5e_4_2transformer_reconstruct(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_2transformer_latent16/best_valid_loss--1.185.ckpt",
        "reconstruct",
    )


def ae_lr5e_4_4transformer_latent16(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_4transformer_latent16/best_valid_loss--1.383.ckpt",
        "latent",
    )


def ae_lr5e_4_4transformer_latent16_stack8(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_4transformer_latent16/best_valid_loss--1.383.ckpt",
        "latent",
        stack=8,
    )


def ae_lr5e_4_4transformer_reconstruct(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_4transformer_latent16/best_valid_loss--1.383.ckpt",
        "reconstruct",
    )


def ae_lr5e_4_6transformer_latent16(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_6transformer_latent16/best_valid_loss--1.462.ckpt",
        "latent",
    )


def ae_lr5e_4_6transformer_reconstruct(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/vae/ae_lr5e-4_6transformer_latent16/best_valid_loss--1.462.ckpt",
        "reconstruct",
    )


def ae_lr5e_4_spatial128_stack8_4transformer_latent128(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/temporal/ae_lr5e-4_spatial128_stack8_4transformer_latent128/best_valid_loss-0.206.ckpt",
        "latent",
    )


def ae_lr5e_4_spatial384_stack4_4transformer_latent64(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/temporal/ae_lr5e-4_spatial384_stack4_4transformer_latent64/best_valid_loss-0.419.ckpt",
        "latent",
    )


def ae_lr5e_4_spatial384_stack4_4transformer_latent64_reconstruct(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/temporal/ae_lr5e-4_spatial384_stack4_4transformer_latent64/best_valid_loss-0.419.ckpt",
        "reconstruct",
    )


def ae_lr5e_4_spatial384_stack4_4transformer_latent64_stack2(*args, **kwargs):
    return UpstreamExpert(
        "/home/leo1994122701/cslm/cst/result/temporal/ae_lr5e-4_spatial384_stack4_4transformer_latent64/best_valid_loss-0.419.ckpt",
        "latent",
        stack=2,
    )
