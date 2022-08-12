import torch
import logging
import argparse
from pathlib import Path

from s3prl import hub
from s3prl.util.pseudo_data import get_pseudo_wavs

SAMPLE_RATE = 16000
logger = logging.getLogger(__name__)


def extract_single_name(
    name: str,
    ckpt: str,
    legacy: bool,
    output_dir: str,
    device: str,
    refresh: bool = False,
):
    output_dir: Path = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = str((output_dir / f"{name}.pt").resolve())

    if Path(output_path).is_file() and not refresh:
        return

    model = getattr(hub, name)(ckpt=ckpt, legacy=legacy).to(device)
    model.eval()

    with torch.no_grad():
        hidden_states = model(get_pseudo_wavs(device=device))["hidden_states"]
        hs = [h.detach().cpu() for h in hidden_states]

    torch.save(hs, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--ckpt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.all:
        options = [
            name
            for name in hub.options(only_registered_ckpt=True)
            if (not name == "customized_upstream")
            and (
                not "mos" in name
            )  # mos models do not have hidden_states key. They only return a single mos score
            and (
                not "stft_mag" in name
            )  # stft_mag upstream must past the config file currently and is not so important. So, skip the test now
            and (
                not "pase" in name
            )  # pase_plus needs lots of dependencies and is difficult to be tested and is not very worthy today
            and (
                not name == "xls_r_1b"
            )  # skip due to too large model, too long download time
            and (
                not name == "xls_r_2b"
            )  # skip due to too large model, too long download time
        ]

        logger.info(f"Extract for: {options}")
        for option in options:
            extract_single_name(
                option,
                args.ckpt,
                args.legacy,
                args.output_dir,
                args.device,
                args.refresh,
            )

    else:
        assert args.name is not None
        extract_single_name(
            args.name,
            args.ckpt,
            args.legacy,
            args.output_dir,
            args.device,
            args.refresh,
        )
