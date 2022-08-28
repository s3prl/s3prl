import torch
import pandas as pd
import torch.nn as nn
from omegaconf import MISSING
from typing import Tuple, List

from s3prl.nn.interface import AbsFrameModel, AbsUpstream
from s3prl.problem.asr.superb_asr import SuperbASR


class CustomizeASR(SuperbASR):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config["prepare_data"] = dict(
            train_utt=MISSING,
            valid_utt=MISSING,
            test_utt=MISSING,
            audio_root=MISSING,
            text=MISSING,
        )
        config["build_downstream"] = dict()
        config["build_tokenizer"] = dict(
            tokenizer_name=None,
            vocab_type="subword",
            vocab_args=dict(
                vocab_size=1000,
            ),
        )
        config["build_upstream"] = dict()
        return config

    @classmethod
    def prepare_data(
        cls,
        _target_dir,
        _cache_dir,
        train_utt: str,
        valid_utt: str,
        test_utt: str,
        audio_root: str,
        text: str,
        _get_path_only=False,
    ):
        train_csv = (
            _target_dir
            / f"{Path(train_utt).resolve().as_posix().replace('/', '-')}.csv"
        )
        valid_csv = (
            _target_dir
            / f"{Path(valid_utt).resolve().as_posix().replace('/', '-')}.csv"
        )
        test_csv = (
            _target_dir / f"{Path(test_utt).resolve().as_posix().replace('/', '-')}.csv"
        )

        if _get_path_only:
            return train_csv, valid_csv, [test_csv]

        def build_df(utt_path):
            utt2text = {}
            with open(text, "r") as f:
                lines = [line.strip() for line in f.readlines()]

            for line in lines:
                utt, trans = line.split(" ", maxsplit=1)
                utt2text[utt] = trans

            with open(utt_path, "r") as f:
                utts = [line.strip() for line in f.readlines()]

            wav_paths = []
            trans = []
            for utt in utts:
                wav_paths.append(str(Path(audio_root) / f"{utt}.wav"))
                trans.append(utt2text[utt])

            df = pd.DataFrame(
                {"wav_path": wav_paths, "transcription": trans, "id": utts}
            )
            return df

        build_df(train_utt).to_csv(train_csv, index=False)
        build_df(valid_utt).to_csv(valid_csv, index=False)
        build_df(test_utt).to_csv(test_csv, index=False)
        return train_csv, valid_csv, [test_csv]

    @classmethod
    def build_upstream(cls) -> AbsUpstream:
        class CustomUpstream(AbsUpstream):
            def __init__(self) -> None:
                super().__init__()

            @property
            def num_layers(self) -> int:
                return 1

            @property
            def downsample_rates(self) -> List[int]:
                return [1]

            @property
            def hidden_sizes(self) -> List[int]:
                return [1]

            def forward(self, x, x_len):
                x = x[:, ::320, :]
                return [x], [(x_len.float() / 320).round().long()]

        return CustomUpstream()

    @classmethod
    def build_downstream(
        cls,
        _downstream_input_size: int,
        _downstream_output_size: int,
        _downstream_downsample_rate: int,
    ) -> AbsFrameModel:
        class Linear(AbsFrameModel):
            def __init__(self, input_size, output_size) -> None:
                super().__init__()
                self.linear = nn.Linear(input_size, output_size)

            @property
            def input_size(self):
                return self.linear.in_features

            @property
            def output_size(self) -> int:
                return self.linear.out_features

            def forward(
                self, x: torch.FloatTensor, x_len: torch.LongTensor
            ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
                return self.linear(x), x_len

        model = Linear(_downstream_input_size, _downstream_output_size)
        return model


if __name__ == "__main__":
    from pathlib import Path

    config = CustomizeASR.main()

    model, model_extra_conf = CustomizeASR.load_model(
        Path(config["target_dir"]) / "train" / "valid_best" / "model.pt"
    )
    print(model)

    model, model_extra_conf, task, task_extra_conf = CustomizeASR.load_model_and_task(
        Path(config["target_dir"]) / "train" / "valid_best"
    )
    print(task)
