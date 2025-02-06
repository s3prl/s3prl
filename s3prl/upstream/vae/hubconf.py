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
