import tempfile
import pandas as pd
from pathlib import Path
from s3prl.problem import SuperbASR
from s3prl.util.pseudo_data import pseudo_audio


def test_superb_asr():

    with tempfile.TemporaryDirectory() as tempdir:

        with pseudo_audio([10, 2, 1, 8, 5]) as (wav_paths, num_samples):

            class TestASR(SuperbASR):
                def default_config(self) -> dict:
                    config = super().default_config()
                    config["prepare_data"] = {}
                    return config

                def prepare_data(
                    self,
                    prepare_data: dict,
                    target_dir: str,
                    cache_dir: str,
                    get_path_only: bool = False,
                ):
                    all_wav_paths = wav_paths
                    all_text = [
                        "hello how are you today",
                        "fine",
                        "oh",
                        "I think is good",
                        "maybe okay",
                    ]

                    ids = list(range(len(all_wav_paths)))
                    df = pd.DataFrame(
                        data={
                            "id": ids,
                            "wav_path": all_wav_paths,
                            "transcription": all_text,
                        }
                    )
                    train_path = Path(target_dir) / "train.csv"
                    valid_path = Path(target_dir) / "valid.csv"
                    test_path = Path(target_dir) / "test.csv"

                    df.iloc[:3].to_csv(train_path, index=False)
                    df.iloc[3:4].to_csv(valid_path, index=False)
                    df.iloc[4:].to_csv(test_path, index=False)

                    return train_path, valid_path, [test_path]

            asr = TestASR()
            config = asr.default_config()
            config["target_dir"] = tempdir
            config["device"] = "cpu"
            config["train"]["total_steps"] = 4
            config["train"]["log_step"] = 1
            config["train"]["eval_step"] = 2
            config["train"]["save_step"] = 2
            config["eval_batch"] = 2
            asr.run(**config)
