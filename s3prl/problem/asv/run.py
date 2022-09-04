import os
import yaml
import logging
import pickle
import shutil
import pandas as pd
from typing import List
from pathlib import Path

from s3prl.problem.base import Problem
from s3prl.task.speaker_verification_task import SpeakerVerification


logger = logging.getLogger(__name__)


class ASV(Problem):
    @classmethod
    def run(
        cls,
        target_dir: str,
        cache_dir: str,
        remove_all_cache: bool = False,
        start: int = 0,
        stop: int = None,
        start_stage_id: int = 0,
        num_workers: int = 6,
        eval_batch: int = -1,
        device: str = "cuda",
        world_size: int = 1,
        rank: int = 0,
        test_ckpt_dir: str = None,
        test_ckpt_steps: List[int] = None,
        prepare_data: dict = {},
        build_encoder: dict = {},
        build_dataset: dict = {},
        build_batch_sampler: dict = {},
        build_upstream: dict = {},
        build_featurizer: dict = {},
        build_downstream: dict = {},
        build_model: dict = {},
        build_task: dict = {},
        build_optimizer: dict = {},
        build_scheduler: dict = {},
        save_model: dict = {},
        save_task: dict = {},
        train: dict = {},
    ):
        cls._save_yaml(
            cls._get_current_arguments(),
            Path(target_dir) / "configs" / f"{cls._get_time_tag()}.yaml",
        )

        target_dir: Path = Path(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)
        if remove_all_cache:
            shutil.rmtree(cache_dir)

        stage_id = start_stage_id
        if start <= stage_id:
            logger.info(f"Stage {stage_id}: prepare data")
            train_csv, test_csvs = cls.prepare_data(
                target_dir, cache_dir, _get_path_only=False, **prepare_data
            )

        train_csv, test_csvs = cls.prepare_data(
            target_dir, cache_dir, _get_path_only=True, **prepare_data
        )

        def check_fn():
            assert Path(train_csv).is_file()
            for test_csv in test_csvs:
                assert Path(test_csv).is_file()

        cls._stage_check(stage_id, stop, check_fn)

        stage_id += 1
        if start <= stage_id:
            logger.info(f"Stage {stage_id}: build encoder")
            encoder_path = cls.build_encoder(
                target_dir,
                cache_dir,
                train_csv,
                test_csvs,
                _get_path_only=False,
                **build_encoder,
            )

        encoder_path = cls.build_encoder(
            target_dir,
            cache_dir,
            train_csv,
            test_csvs,
            _get_path_only=True,
            **build_encoder,
        )

        def check_fn():
            assert Path(encoder_path).is_file()

        cls._stage_check(stage_id, stop, check_fn)

        stage_id += 1
        train_dir = target_dir / "train"
        if start <= stage_id:
            logger.info(f"Stage {stage_id}: Train Model")
            train_ds, train_bs = cls._build_dataset_and_sampler(
                target_dir,
                cache_dir,
                "train",
                train_csv,
                encoder_path,
                build_dataset,
                build_batch_sampler,
            )

            with Path(encoder_path).open("rb") as f:
                encoder = pickle.load(f)

            init_model = dict(
                _model_output_size=len(encoder),
                _build_upstream=build_upstream,
                _build_featurizer=build_featurizer,
                _build_downstream=build_downstream,
                **build_model,
            )
            init_task = dict(
                _encoder=encoder,
                **build_task,
            )

            cls.train(
                train_dir,
                init_model,
                init_task,
                save_model,
                save_task,
                build_optimizer,
                build_scheduler,
                train_ds,
                train_bs,
                None,
                None,
                _device=device,
                _eval_batch=eval_batch,
                _num_workers=num_workers,
                _world_size=world_size,
                _rank=rank,
                **train,
            )

        test_ckpt_dirs = []
        if test_ckpt_dir is not None:
            test_ckpt_dirs.append(test_ckpt_dir)
        for step in test_ckpt_steps:
            test_ckpt_dirs.append(Path(train_dir) / f"step_{step}")

        def check_fn():
            for ckpt_dir in test_ckpt_dirs:
                assert Path(ckpt_dir).is_dir(), ckpt_dir

        cls._stage_check(stage_id, stop, check_fn)

        stage_id += 1
        if start <= stage_id:
            for test_idx, test_csv in enumerate(test_csvs):
                for ckpt_idx, ckpt_dir in enumerate(test_ckpt_dirs):
                    test_name = Path(test_csv).stem
                    test_dir: Path = (
                        target_dir
                        / "evaluate"
                        / test_name
                        / ckpt_dir.relative_to(train_dir).as_posix().replace("/", "-")
                    )
                    test_dir.mkdir(exist_ok=True, parents=True)

                    logger.info(
                        f"Stage {stage_id}.{test_idx}.{ckpt_idx}: Test on {test_csv} with model {ckpt_dir}"
                    )
                    test_ds, test_bs = cls._build_dataset_and_sampler(
                        target_dir,
                        cache_dir,
                        "test",
                        test_csv,
                        encoder_path,
                        build_dataset,
                        build_batch_sampler,
                    )

                    csv = pd.read_csv(test_csv)
                    test_trials = []
                    for rowid, row in csv.iterrows():
                        test_trials.append(
                            (int(row["label"]), str(row["id1"]), str(row["id2"]))
                        )

                    overrides = dict(test_trials=test_trials)
                    _, _, task, _ = cls.load_model_and_task(ckpt_dir, overrides)
                    logs = cls._evaluate(
                        "test",
                        task,
                        test_ds,
                        test_bs,
                        eval_batch,
                        test_dir,
                        device,
                        num_workers,
                    )
                    test_metrics = {name: float(value) for name, value in logs.items()}
                    assert "EER" in test_metrics
                    cls._save_yaml(test_metrics, test_dir / f"result.yaml")

        cls._stage_check(stage_id, stop, lambda: True)

        stage_id += 1
        if start <= stage_id:
            for test_idx, test_csv in enumerate(test_csvs):
                test_name = Path(test_csv).stem
                logger.info(f"Report results on {test_name}")

                eer_ckpts = []
                for ckpt_dir in os.listdir(target_dir / "evaluate" / test_name):
                    result_yaml: Path = (
                        target_dir / "evaluate" / test_name / ckpt_dir / "result.yaml"
                    )
                    if result_yaml.is_file():
                        with open(result_yaml) as f:
                            eer_ckpts.append(
                                (
                                    float(yaml.load(f, Loader=yaml.FullLoader)["EER"]),
                                    str(result_yaml.parent),
                                )
                            )

                logger.info(f"All EERs on {test_name}:")
                for eer, ckpt in eer_ckpts:
                    logger.info(f"ckpt_dir: {ckpt}, eer: {eer}")

                eer_ckpts.sort(key=lambda x: x[0])
                best_eer, best_ckpt_dir = eer_ckpts[0]

                logger.info(
                    f"Best EER on {test_name} is from {best_ckpt_dir}: {best_eer}"
                )

                cls._save_yaml(
                    dict(
                        EER=best_eer,
                    ),
                    target_dir / "evaluate" / test_name / "best_result.yaml",
                )

    @classmethod
    def _build_dataset_and_sampler(
        cls,
        _target_dir: str,
        _cache_dir: str,
        _mode: str,
        _data_csv: str,
        _encoder_path: str,
        _build_dataset: dict,
        _build_batch_sampler: dict,
    ):
        logger.info(f"Build {_mode} dataset")
        dataset = cls.build_dataset(
            _target_dir,
            _cache_dir,
            _mode,
            _data_csv,
            _encoder_path,
            **_build_dataset,
        )
        logger.info(f"Build {_mode} batch sampler")
        batch_sampler = cls.build_batch_sampler(
            _target_dir,
            _cache_dir,
            _mode,
            _data_csv,
            dataset,
            **_build_batch_sampler,
        )
        return dataset, batch_sampler

    @classmethod
    def build_task(cls, _model, _encoder, **build_task):
        task = SpeakerVerification(_model, _encoder, **build_task)
        return task
