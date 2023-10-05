"""
The backbone run procedure for Speaker Diarization

Authors:
  * Leo 2022
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

from s3prl.problem.base import Problem
from s3prl.task.diarization import DiarizationPIT

from .util import csv_to_kaldi_dir, kaldi_dir_to_rttm, make_rttm_and_score

logger = logging.getLogger(__name__)

__all__ = ["Diarization"]


class Diarization(Problem):
    def run(
        self,
        target_dir: str,
        cache_dir: str,
        remove_all_cache: bool = False,
        start: int = 0,
        stop: int = None,
        num_workers: int = 6,
        eval_batch: int = -1,
        device: str = "cuda",
        world_size: int = 1,
        rank: int = 0,
        test_ckpt_dir: str = None,
        num_speaker: int = 2,
        prepare_data: dict = None,
        build_dataset: dict = None,
        build_batch_sampler: dict = None,
        build_collate_fn: dict = None,
        build_upstream: dict = None,
        build_featurizer: dict = None,
        build_downstream: dict = None,
        build_model: dict = None,
        build_task: dict = None,
        build_optimizer: dict = None,
        build_scheduler: dict = None,
        save_model: dict = None,
        save_task: dict = None,
        train: dict = None,
        evaluate: dict = None,
        scoring: dict = None,
    ):
        """
        ========  ====================
        stage     description
        ========  ====================
        0         Parse the corpus and save the Kaldi-style data directory for speaker diarization
        1         Train the model
        2         Inference the prediction
        3         Score the prediction
        ========  ====================

        Args:
            target_dir (str):
                The directory that stores the script result.
            cache_dir (str):
                The directory that caches the processed data.
                Default: /home/user/.cache/s3prl/data
            remove_all_cache (bool):
                Whether to remove all the cache stored under `cache_dir`.
                Default: False
            start (int):
                The starting stage of the problem script.
                Default: 0
            stop (int):
                The stoping stage of the problem script, set `None` to reach the final stage.
                Default: None
            num_workers (int): num_workers for all the torch DataLoder
            eval_batch (int):
                During evaluation (valid or test), limit the number of batch.
                This is helpful for the fast development to check everything won't crash.
                If is -1, disable this feature and evaluate the entire epoch.
                Default: -1
            device (str):
                The device type for all torch-related operation: "cpu" or "cuda"
                Default: "cuda"
            world_size (int):
                How many processes are running this script simultaneously (in parallel).
                Usually this is just 1, however if you are runnig distributed training,
                this should be > 1.
                Default: 1
            rank (int):
                When distributed training, world_size > 1. Take :code:`world_size == 8` for
                example, this means 8 processes (8 GPUs) are runing in parallel. The script
                needs to know which process among 8 processes it is. In this case, :code:`rank`
                can range from 0~7. All the 8 processes have the same :code:`world_size` but
                different :code:`rank` (process id).
            test_ckpt_dir (str):
                Specify the checkpoint path for testing. If not, use checkpoints specified by
                :code:`test_ckpts_steps`.
            num_speaker (int):
                How many speakers per utterance
            **others:
                The other arguments like :code:`prepare_data` and :code:`build_model` are
                method specific-arguments for methods like :obj:`prepare_data` and
                :obj:`build_model`, and will not be used in the core :obj:`run` logic.
                See the specific method documentation for their supported arguments and
                meaning
        """

        yaml_path = Path(target_dir) / "configs" / f"{self._get_time_tag()}.yaml"
        yaml_path.parent.mkdir(exist_ok=True, parents=True)
        with yaml_path.open("w") as f:
            yaml.safe_dump(self._get_current_arguments(), f)

        cache_dir: str = cache_dir or Path.home() / ".cache" / "s3prl" / "data"
        prepare_data: dict = prepare_data or {}
        build_dataset: dict = build_dataset or {}
        build_batch_sampler: dict = build_batch_sampler or {}
        build_collate_fn: dict = build_collate_fn or {}
        build_upstream: dict = build_upstream or {}
        build_featurizer: dict = build_featurizer or {}
        build_downstream: dict = build_downstream or {}
        build_model: dict = build_model or {}
        build_task: dict = build_task or {}
        build_optimizer: dict = build_optimizer or {}
        build_scheduler: dict = build_scheduler or {}
        save_model: dict = save_model or {}
        save_task: dict = save_task or {}
        train: dict = train or {}
        evaluate: dict = evaluate or {}
        scoring: dict = scoring or {}

        target_dir: Path = Path(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)
        if remove_all_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)

        stage_id = 0
        if start <= stage_id:
            logger.info(f"Stage {stage_id}: prepare data")
            train_csv, valid_csv, test_csvs = self.prepare_data(
                prepare_data, target_dir, cache_dir, get_path_only=False
            )

        train_csv, valid_csv, test_csvs = self.prepare_data(
            prepare_data, target_dir, cache_dir, get_path_only=True
        )

        def check_fn():
            assert Path(train_csv).is_file() and Path(valid_csv).is_file()
            for test_csv in test_csvs:
                assert Path(test_csv).is_file()

        self._stage_check(stage_id, stop, check_fn)

        for csv in [train_csv, valid_csv, *test_csvs]:
            data_dir = target_dir / "kaldi_data" / Path(csv).stem
            csv_to_kaldi_dir(csv, data_dir)

        train_data = target_dir / "kaldi_data" / Path(train_csv).stem
        valid_data = target_dir / "kaldi_data" / Path(valid_csv).stem
        test_datas = [target_dir / "kaldi_data" / Path(csv).stem for csv in test_csvs]

        test_rttms = []
        for test_data in test_datas:
            logger.info(f"Prepare RTTM for {test_data}")
            test_rttm = target_dir / f"{Path(test_data).stem}.rttm"
            kaldi_dir_to_rttm(test_data, test_rttm)
            test_rttms.append(test_rttm)

        model_output_size = num_speaker
        model = self.build_model(
            build_model,
            model_output_size,
            build_upstream,
            build_featurizer,
            build_downstream,
        )
        frame_shift = model.downsample_rate

        stage_id = 1
        train_dir = target_dir / "train"
        if start <= stage_id:
            logger.info(f"Stage {stage_id}: Train Model")
            train_ds, train_bs = self._build_dataset_and_sampler(
                target_dir,
                cache_dir,
                "train",
                train_csv,
                train_data,
                num_speaker,
                frame_shift,
                build_dataset,
                build_batch_sampler,
            )
            valid_ds, valid_bs = self._build_dataset_and_sampler(
                target_dir,
                cache_dir,
                "valid",
                valid_csv,
                valid_data,
                num_speaker,
                frame_shift,
                build_dataset,
                build_batch_sampler,
            )

            build_model_all_args = dict(
                build_model=build_model,
                model_output_size=model_output_size,
                build_upstream=build_upstream,
                build_featurizer=build_featurizer,
                build_downstream=build_downstream,
            )
            build_task_all_args_except_model = dict(
                build_task=build_task,
            )

            self.train(
                train,
                train_dir,
                build_model_all_args,
                build_task_all_args_except_model,
                save_model,
                save_task,
                build_optimizer,
                build_scheduler,
                evaluate,
                train_ds,
                train_bs,
                self.build_collate_fn(build_collate_fn, "train"),
                valid_ds,
                valid_bs,
                self.build_collate_fn(build_collate_fn, "valid"),
                device=device,
                eval_batch=eval_batch,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
            )

        def check_fn():
            assert (train_dir / "valid_best").is_dir()

        self._stage_check(stage_id, stop, check_fn)

        stage_id = 2
        test_ckpt_dir: Path = Path(test_ckpt_dir or target_dir / "train" / "valid_best")
        test_dirs = []
        for test_idx, test_data in enumerate(test_datas):
            test_name = Path(test_data).stem
            test_dir: Path = (
                target_dir
                / "evaluate"
                / test_ckpt_dir.relative_to(train_dir).as_posix().replace("/", "-")
                / test_name
            )
            test_dirs.append(test_dir)

        if start <= stage_id:
            logger.info(f"Stage {stage_id}: Test model: {test_ckpt_dir}")
            for test_idx, test_data in enumerate(test_datas):
                test_csv = test_csvs[test_idx]
                test_dir = test_dirs[test_idx]
                test_dir.mkdir(exist_ok=True, parents=True)

                logger.info(
                    f"Stage {stage_id}.{test_idx}: Test model on {test_dir} and dump prediction"
                )
                test_ds, test_bs = self._build_dataset_and_sampler(
                    target_dir,
                    cache_dir,
                    "test",
                    test_csv,
                    test_data,
                    num_speaker,
                    frame_shift,
                    build_dataset,
                    build_batch_sampler,
                )

                _, valid_best_task = self.load_model_and_task(test_ckpt_dir)
                logs: dict = self.evaluate(
                    evaluate,
                    "test",
                    valid_best_task,
                    test_ds,
                    test_bs,
                    self.build_collate_fn(build_collate_fn, "test"),
                    eval_batch,
                    test_dir,
                    device,
                    num_workers,
                )
                test_metrics = {name: float(value) for name, value in logs.items()}
                with (test_dir / f"result.yaml").open("w") as f:
                    yaml.safe_dump(test_metrics, f)

        def check_fn():
            for test_dir in test_dirs:
                assert (test_dir / "prediction").is_dir()

        self._stage_check(stage_id, stop, check_fn)

        stage_id = 3
        if start <= stage_id:
            logger.info(f"Stage {stage_id}: Score model: {test_ckpt_dir}")
            self.scoring(scoring, stage_id, test_dirs, test_rttms, frame_shift)

        return stage_id

    def scoring(
        self,
        scoring: dict,
        stage_id: int,
        test_dirs: List[str],
        test_rttms: List[str],
        frame_shift: int,
    ):
        """
        Score the prediction

        Args:
            scoring (dict):

                ====================  ====================
                key                   description
                ====================  ====================
                thresholds            (List[int]) - Given the 0~1 (float) soft prediction, the threshold decides \
                                        how to get the 0/1 hard prediction. This list are all the thresholds to try.
                median_filters        (List[int]) - After getting hard prediction, use median filter to smooth out the \
                                        prediction. This list are all the median filter sizes to try.
                ====================  ====================

            *others:
                This method is not designed to be overridden
        """

        @dataclass
        class ScoreConfig:
            thresholds: List[int]
            median_filters: List[int]

        conf = ScoreConfig(**scoring)

        for test_idx, test_dir in enumerate(test_dirs):
            logger.info(
                f"Stage {stage_id}.{test_idx}: Make RTTM and Score from prediction"
            )
            best_der, (best_th, best_med) = make_rttm_and_score(
                test_dir / "prediction",
                test_dir / "score",
                test_rttms[test_idx],
                frame_shift,
                conf.thresholds,
                conf.median_filters,
            )

            logger.info(f"Best dscore DER: {best_der}")
            with (test_dir / "dscore.yaml").open("w") as f:
                yaml.safe_dump(
                    dict(
                        der=best_der,
                        threshold=best_th,
                        median_filter=best_med,
                    ),
                    f,
                )

    def _build_dataset_and_sampler(
        self,
        target_dir: str,
        cache_dir: str,
        mode: str,
        data_csv: str,
        data_dir: str,
        num_speakers: int,
        frame_shift: int,
        build_dataset: dict,
        build_batch_sampler: dict,
    ):
        logger.info(f"Build {mode} dataset")
        dataset = self.build_dataset(
            build_dataset,
            target_dir,
            cache_dir,
            mode,
            data_csv,
            data_dir,
            num_speakers,
            frame_shift,
        )
        logger.info(f"Build {mode} batch sampler")
        batch_sampler = self.build_batch_sampler(
            build_batch_sampler,
            target_dir,
            cache_dir,
            mode,
            data_csv,
            data_dir,
            dataset,
        )
        return dataset, batch_sampler

    def build_task(self, build_task: dict, model):
        """
        Build the task, which defines the logics for every train/valid/test forward step for the :code:`model`,
        and the logics for how to reduce all the batch results from multiple train/valid/test steps into metrics

        By default build :obj:`DiarizationPIT`

        Args:
            build_task (dict): same in :obj:`default_config`, no argument supported for now
            model (torch.nn.Module): the model built by :obj:`build_model`

        Returns:
            Task
        """
        task = DiarizationPIT(model)
        return task
