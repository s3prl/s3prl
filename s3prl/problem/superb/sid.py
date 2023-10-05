from s3prl import Container
from s3prl.corpus.voxceleb1sid import voxceleb1_for_utt_classification
from s3prl.dataset.base import DataPipe, SequentialDataPipe
from s3prl.dataset.common_pipes import RandomCrop, SetOutputKeys
from s3prl.dataset.utterance_classification_pipe import UtteranceClassificationPipe
from s3prl.nn import MeanPoolingLinear
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task.utterance_classification_task import UtteranceClassificationTask
from s3prl.util.configuration import default_cfg


class SuperbSIDTrainPipe(DataPipe):
    def __init__(
        self,
        train_category_encoder: bool = False,
        max_secs: float = None,
    ) -> None:
        self.pipes = SequentialDataPipe(
            UtteranceClassificationPipe(
                train_category_encoder=train_category_encoder,
            ),
            RandomCrop(max_secs=max_secs),
            SetOutputKeys(
                dict(
                    x="wav_crop",
                    x_len="wav_crop_len",
                )
            ),
        )

    def forward(self, dataset):
        dataset = self.pipes(dataset)
        return dataset


from .base import SuperbProblem


class SuperbSID(SuperbProblem):
    """
    Superb SID
    """

    @default_cfg(
        **SuperbProblem.setup.default_except(
            corpus=dict(
                CLS=voxceleb1_for_utt_classification,
                dataset_root="???",
            ),
            train_datapipe=dict(
                CLS=SuperbSIDTrainPipe,
                train_category_encoder=True,
                max_secs=8.0,
            ),
            train_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=8,
                shuffle=True,
            ),
            valid_datapipe=dict(
                CLS=UtteranceClassificationPipe,
            ),
            valid_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=1,
            ),
            test_datapipe=dict(
                CLS=UtteranceClassificationPipe,
            ),
            test_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=1,
            ),
            downstream=dict(
                CLS=MeanPoolingLinear,
                hidden_size=256,
            ),
            task=dict(
                CLS=UtteranceClassificationTask,
            ),
        )
    )
    @classmethod
    def setup(cls, **cfg):
        """
        This setups the IC problem, containing train/valid/test datasets & samplers and a task object
        """
        super().setup(**cfg)

    @default_cfg(
        **SuperbProblem.train.default_except(
            optimizer=dict(
                CLS="torch.optim.Adam",
                lr=1.0e-4,
            ),
            trainer=dict(
                total_steps=200000,
                log_step=500,
                eval_step=5000,
                save_step=1000,
                gradient_clipping=1.0,
                gradient_accumulate_steps=4,
                valid_metric="accuracy",
                valid_higher_better=True,
            ),
        )
    )
    @classmethod
    def train(cls, **cfg):
        """
        Train the setup problem with the train/valid datasets & samplers and the task object
        """
        super().train(**cfg)

    @default_cfg(**SuperbProblem.inference.default_cfg)
    @classmethod
    def inference(cls, **cfg):
        super().inference(**cfg)

    @default_cfg(
        **SuperbProblem.run.default_except(
            stages=["setup", "train", "inference"],
            start_stage="setup",
            final_stage="inference",
            setup=setup.default_cfg.deselect("workspace", "resume"),
            train=train.default_cfg.deselect("workspace", "resume"),
            inference=inference.default_cfg.deselect("workspace", "resume"),
        )
    )
    @classmethod
    def run(cls, **cfg):
        super().run(**cfg)
