"""
The general usage of the :py:obj:`~s3prl.problem` package
"""


import logging

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
    to see the results. Given that `SuperbIC.train` will provide a `valid_best_task` key to the workspace,
    you can access that checkpoint by `workspace["valid_best_task"]`

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
        ckpt = workspace["valid_best_task"]

    Basically, workspace help abstract out all the file handling details and act like a dictionary,
    so that you only need to put and get stuffs via their `key`.

    For the :py:obj:`~s3prl.problem.superb.base.SuperbProblem`, there is a Stage before `train`
    called :py:obj:`~s3prl.problem.superb.base.SuperbProblem.setup`. This Stage handles
    the data preprocessing and model building, so that you can simply load them to `train` at the
    next Stage. Also, you can use the workspace to retrive these preprocessed objects in another python
    session and train them with other toolkits, as they are just regular Dataset, Sampler, and
    a Task defining what to do at each train/valid/test step.

    .. code-block:: python

        # in one python session
        SuperbIC.setup(
            workspace=workspace,
            corpus=dict(
                dataset_root="your dataset path",
            )
            upstream=dict(
                name="fbank",
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

    Besides importing these functions and use in your python script, you can also directly call them
    in the command line. We provide an all-in-one CLI tool for S3PRL.

    .. code-block:: shell

        s3prl-cli [module path] [stage function qualname] OVERRIDE1 OVERRIDE2 ...

    .. note:

        OVERRIDE is in the format of "dictionary_key_path=value". That is, if you want to override the `inner` field in
        dict(outer=dict(inner=3)), you can use `outer.inner=5`. The config will becomes dict(outer=dict(inner=5))

    You can see the `--usage` page of the SuperbIC problem to see what are the necessary options to provide

    .. code-block:: shell

        s3prl-cli s3prl.problem.superb.ic SuperbIC.setup --usage

    You can first run a single **Stage** `setup` to create the necessary components of the SuperbIC problem,
    and then run the second **Stage** `train` and so on...

    .. code-block:: shell

        s3prl-cli s3prl.problem.superb.ic SuperbIC.setup workspace=result/tmp upstream.name=fbank corpus.dataset_root='fluent_speech_command_path'

    Or, you can run all the **Stages** at once with `run_stages`.

    .. code-block:: shell

        s3prl-cli s3prl.problem.superb.ic SuperbIC.run_stages workspace=result/tmp upstream.name=fbank corpus.dataset_root='fluent_speech_command_path'

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
        workspace=field("???", "The workspace shared across stages", str),
        resume=field(False, "The resume flag shared across stages", bool),
        stages=[field("???", "The stage methods to run through", str)],
        start_stage=field("???", "Start from this stage", "int or str"),
        final_stage=field(
            "???",
            "The final stage. End at this stage (inclusive). That is, when start_stage=0, final_stage=1, there will be two Stages executed",
            "int or str",
        ),
    )
    @classmethod
    def run_stages(cls, **cfg):
        if isinstance(cfg["start_stage"], str):
            cfg["start_stage"] = cfg["stages"].index(cfg["start_stage"])

        if isinstance(cfg["final_stage"], str):
            cfg["final_stage"] = cfg["stages"].index(cfg["final_stage"])

        for stage_name in cfg["stages"][cfg["start_stage"] : cfg["final_stage"] + 1]:
            stage_func = getattr(cls, stage_name)
            stage_cfg = cfg[stage_name]
            stage_cfg.override(
                dict(
                    workspace=cfg["workspace"],
                    resume=cfg["resume"],
                )
            )
            stage_func(**stage_cfg)
