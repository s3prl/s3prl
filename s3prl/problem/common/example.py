from pathlib import Path

import pandas as pd
import torch
import torchaudio

from .superb_sid import SuperbSID

torchaudio.set_audio_backend("sox_io")


class CommonExample(SuperbSID):
    def default_config(self) -> dict:
        config = super().default_config()
        config["prepare_data"] = {}
        config["train"] = dict(
            total_steps=10,
            log_step=1,
            eval_step=5,
            save_step=5,
            gradient_clipping=1.0,
            gradient_accumulate=1,
            valid_metric="accuracy",
            valid_higher_better=True,
            auto_resume=True,
        )
        return config

    def prepare_data(
        self,
        prepare_data: dict,
        target_dir: str,
        cache_dir: str,
        get_path_only: bool = False,
    ):
        target_dir: Path = Path(target_dir)

        wavs = [torch.randn(1, 16000 * 2) for i in range(5)]
        wav_paths = []
        for idx, wav in enumerate(wavs):
            wav_path = str(Path(target_dir) / f"{idx}.wav")
            torchaudio.save(wav_path, wav, sample_rate=16000)
            wav_paths.append(wav_path)

        ids = [Path(path).stem for path in wav_paths]
        labels = ["a", "a", "b", "c", "d"]

        df = pd.DataFrame({"id": ids, "wav_path": wav_paths, "label": labels})
        train_csv, valid_csv, test_csv = (
            target_dir / "train.csv",
            target_dir / "valid.csv",
            target_dir / "test.csv",
        )
        df.iloc[:3].to_csv(train_csv)
        df.iloc[3:4].to_csv(valid_csv)
        df.iloc[4:].to_csv(test_csv)

        return train_csv, valid_csv, [test_csv]
