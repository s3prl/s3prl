import inspect
import logging
import pickle
import shutil
from pathlib import Path

from torch.utils.data import DataLoader

from s3prl.problem.base import Problem

logger = logging.getLogger(__name__)


class ASR(Problem):
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
        prepare_data: dict = {},
        prepare_tokenizer_data: dict = {},
        build_tokenizer: dict = {},
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
        evaluate: dict = {},
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
            train_csv, valid_csv, test_csvs = cls.prepare_data(
                target_dir, cache_dir, _get_path_only=False, **prepare_data
            )

        train_csv, valid_csv, test_csvs = cls.prepare_data(
            target_dir, cache_dir, _get_path_only=True, **prepare_data
        )

        def check_fn():
            assert Path(train_csv).is_file() and Path(valid_csv).is_file()
            for test_csv in test_csvs:
                assert Path(test_csv).is_file()

        cls._stage_check(stage_id, stop, check_fn)

        stage_id += 1
        if start <= stage_id:
            logger.info(f"Stage {stage_id}: prepare tokenizer data")
            tokenizer_data_path = cls.prepare_tokenizer_data(
                target_dir,
                cache_dir,
                train_csv,
                _get_path_only=False,
                **prepare_tokenizer_data,
            )

        tokenizer_data_path = cls.prepare_tokenizer_data(
            target_dir,
            cache_dir,
            train_csv,
            _get_path_only=True,
            **prepare_tokenizer_data,
        )

        def check_fn():
            assert Path(tokenizer_data_path).is_file()

        cls._stage_check(stage_id, stop, check_fn)

        stage_id += 1
        if start <= stage_id:
            logger.info(f"Stage {stage_id}: build tokenizer")
            tokenizer_path = cls.build_tokenizer(
                target_dir,
                cache_dir,
                tokenizer_data_path,
                _get_path_only=False,
                **build_tokenizer,
            )

        tokenizer_path = cls.build_tokenizer(
            target_dir,
            cache_dir,
            tokenizer_data_path,
            _get_path_only=True,
            **build_tokenizer,
        )

        def check_fn():
            assert Path(tokenizer_path).is_file()

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
                tokenizer_path,
                build_dataset,
                build_batch_sampler,
            )
            valid_ds, valid_bs = cls._build_dataset_and_sampler(
                target_dir,
                cache_dir,
                "valid",
                valid_csv,
                tokenizer_path,
                build_dataset,
                build_batch_sampler,
            )

            with Path(tokenizer_path).open("rb") as f:
                tokenizer = pickle.load(f)

            init_model = dict(
                _model_output_size=len(tokenizer),
                _build_upstream=build_upstream,
                _build_featurizer=build_featurizer,
                _build_downstream=build_downstream,
                **build_model,
            )
            init_task = dict(
                _tokenizer=tokenizer,
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
                valid_ds,
                valid_bs,
                _device=device,
                _eval_batch=eval_batch,
                _num_workers=num_workers,
                _world_size=world_size,
                _rank=rank,
                **train,
            )

        def check_fn():
            assert (train_dir / "valid_best").is_dir()

        cls._stage_check(stage_id, stop, check_fn)

        stage_id += 1
        if start <= stage_id:
            test_ckpt_dir: Path = Path(
                test_ckpt_dir or target_dir / "train" / "valid_best"
            )
            logger.info(f"Stage {stage_id}: Test model: {test_ckpt_dir}")
            for test_idx, test_csv in enumerate(test_csvs):
                test_name = Path(test_csv).stem
                test_dir: Path = (
                    target_dir
                    / "evaluate"
                    / test_ckpt_dir.relative_to(train_dir).as_posix().replace("/", "-")
                    / test_name
                )
                test_dir.mkdir(exist_ok=True, parents=True)

                logger.info(f"Stage {stage_id}.{test_idx}: Test model on {test_csv}")
                test_ds, test_bs = cls._build_dataset_and_sampler(
                    target_dir,
                    cache_dir,
                    "test",
                    test_csv,
                    tokenizer_path,
                    build_dataset,
                    build_batch_sampler,
                )
                test_dl = DataLoader(
                    test_ds,
                    batch_sampler=test_bs,
                    num_workers=num_workers,
                    collate_fn=cls.build_collate_fn("test"),
                )

                _, _, valid_best_task, _ = cls.load_model_and_task(test_ckpt_dir)
                logs: dict = cls._evaluate(
                    "test",
                    valid_best_task,
                    test_dl,
                    eval_batch,
                    test_dir,
                    device,
                    **evaluate,
                )
                test_metrics = {name: float(value) for name, value in logs.items()}
                cls._save_yaml(test_metrics, test_dir / f"result.yaml")

        return stage_id

    @classmethod
    def _build_dataset_and_sampler(
        cls,
        _target_dir: str,
        _cache_dir: str,
        _mode: str,
        _data_csv: str,
        _tokenizer_path: str,
        _build_dataset: dict,
        _build_batch_sampler: dict,
    ):
        logger.info(f"Build {_mode} dataset")
        dataset = cls.build_dataset(
            _target_dir,
            _cache_dir,
            _mode,
            _data_csv,
            _tokenizer_path,
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
    def build_task(cls, _model, _tokenizer: str, log_metrics: list):
        from s3prl.task.speech2text_ctc_task import Speech2TextCTCTask

        task = Speech2TextCTCTask(_model, _tokenizer, log_metrics=log_metrics)
        return task
