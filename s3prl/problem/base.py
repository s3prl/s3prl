"""
The general usage of the :py:obj:`~s3prl.problem` package
"""


import logging
from subprocess import call

from s3prl import field
from s3prl.base.container import Container
from s3prl.util.configuration import _add_doc, default_cfg
from s3prl.util.workspace import Workspace

logger = logging.getLogger(__name__)


class Problem:
    """
    A **Problem** is a sequence of Stage to be executed to solve a specific corpus-task pair.
    By running the `run_stages` classmethod, you can execute a complete recipe from scratch.
    Besides `run_stages`, there are other classmethods, and each of them is a **Stage** utility.

    .. note::

        A **Stage** utility a pythonic function which is a binary execution block. That is,
        it reads some files from a directory, do something, and writes some files to the
        directory. Hence, the excecution across two **Stages** can be in different python session.

        This is important, as ususally a machine learning recipe is very resource hungry, while
        you might require different resources at different **Stage**. Like, the data preprocessing
        **Stage** might be very time-consuming and requires lots of CPU but no GPU; the training
        **Stage** might need lots of GPU and high disk I/O speed but less CPU, and the inference **Stage**
        might want some machine specifically desinged to be fast on inference.

    .. code-block:: python

        from s3prl import Workspace
        from s3prl.problem.superb.ic import SuperbIC

        workspace = Workspace("result/tmp")
        SuperbIC.run_stages(workspace=workspace, resume=True)

    The Problem's `run_stages` function also support to run from a specific stage to another stage,
    in case you want to resume the recipe execution from the middle.

    .. code-block:: python

        SuperbIC.run_stages(workspace=workspace, start_stage=1, final_stage=-1)

    You can also execute each Stage one-by-one, you can also easily access the output of each Stage
    to see the results. Given that `SuperbIC.train` will provide a `valid_best` key to the workspace,
    you can access that checkpoint by `workspace["valid_best"]`

    .. code-block:: python

        # in one python session
        SuperbIC.train(
            workspace=workspace,
            optimizer=dict(
                lr=0.0001
            ),
            trainer=dict(
                total_steps=100
            ),
        )

        # in another python session
        ckpt = workspace["valid_best"]

    Basically, workspace help abstract out all the file handling details and act like a dictionary,
    so that you only need to put and get stuffs via their `key`.

    For the :py:obj:`~s3prl.problem.superb.base.SuperbProblem`, there is a Stage before `train`
    called :py:obj:`~s3prl.problem.superb.base.SuperbProblem.setup_problem`. This Stage handles
    the data preprocessing and model building, so that you can simply load them to `train` at the
    next Stage. Also, you can use the workspace to retrive these preprocessed objects in another python
    session and train them with other toolkits, as they are just regular Dataset, Sampler, and
    a Task defining what to do at each train/valid/test step.

    .. code-block:: python

        # in one python session
        SuperbIC.setup_problem(
            workspace=workspace,
            corpus=dict(
                dataset_root="your dataset path"
            )
        )

        # in another python session
        # just a conceptual usage
        import pytorch_lightning as pl

        train_dataset = workspace["train_dataset"]
        valid_dataset = workspace["valid_dataset"]
        task = workspace["task"]

        trainer = pl.Trainer(n_gpus=1)
        trainer.fit(task, train_dataset, valid_dataset)

    """

    def __init_subclass__(cls) -> None:
        _add_doc(
            cls,
            f".. hint::\n\n    **To run all stages at once:** :py:obj:`~{cls.run_stages.__module__}.{cls.run_stages.__qualname__}`\n",
            last=False,
        )
        _add_doc(
            cls,
            f".. hint::\n\n    Please refer to :py:obj:`~s3prl.problem.base.Problem` for the general usage of this :py:obj:`~s3prl.problem` package\n",
            last=False,
        )

    @default_cfg(
        workspace=field(
            "???",
            "The default workspace for all the stages\nEach Stage will use this if a stage-specific workspace is not given",
            "str or Path or Workspace",
        ),
        resume=field(True, "The default resume option for all the stages", bool),
        start_stage=field(0, "Start from this stage", int),
        final_stage=field(
            0,
            "The final stage. End at this stage (inclusive). That is, when start_stage=0, final_stage=1, there will be two Stages executed",
            int,
        ),
        stage_0=dict(
            _method=field(
                "???",
                "The __name__ of the classmethod to be ran for this stage.\n"
                "The other keys below are used as the **kwargs (**cfg) into this classmethod",
            ),
        ),
    )
    @classmethod
    def run_stages(cls, **cfg):
        cfg = Container(cfg)
        stages_state_dir = Workspace(cfg.workspace) / "_stages"
        sorted_stage_cfgs = cls.get_execution_order(cfg)
        for stage_index, stage_cfg in sorted_stage_cfgs[
            cfg.start_stage : cfg.final_stage + 1
        ]:
            done_mark = f"{stage_index}.done"
            if done_mark in stages_state_dir:
                if cfg.resume:
                    logger.info(
                        f"Skip stage {stage_index} since it was already done and 'resume' is True"
                    )
                    continue
                else:
                    logger.info(
                        f"Delete stage {stage_index}.done mark since 'resume' is False and this stage should be re-executed"
                    )
                    stages_state_dir.remove(done_mark)

            func = getattr(cls, stage_cfg._method)
            assert callable(
                func
            ), f"{func} is not callable and cannot be used as a stage"
            logger.info(
                f"Run stage {stage_index}: {func.__qualname__} with config:\n{stage_cfg}"
            )
            func(**stage_cfg)
            stages_state_dir.put("", done_mark, "txt")

    @staticmethod
    def get_execution_order(cfg):
        cfg = Container(cfg)
        stage_func_cfgs = []
        for key in list(cfg.keys()):
            if key.startswith("stage_"):
                stage_func_cfgs.append((int(key.split("stage_")[-1]), cfg[key]))
        stage_func_cfgs.sort(key=lambda x: x[0])
        return stage_func_cfgs
