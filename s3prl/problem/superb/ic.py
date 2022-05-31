from s3prl import Container
from s3prl.corpus.fluent_speech_commands import (
    FluentSpeechCommandsForUtteranceMultiClassClassificataion,
)
from s3prl.dataset.utterance_classification_pipe import (
    UtteranceMultipleCategoryClassificationPipe,
)
from s3prl.nn import MeanPoolingLinear
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task.utterance_classification_task import (
    UtteranceMultiClassClassificationTask,
)


class SuperbIC:
    Corpus = FluentSpeechCommandsForUtteranceMultiClassClassificataion
    TrainData = UtteranceMultipleCategoryClassificationPipe
    TrainSampler = MaxTimestampBatchSampler
    ValidData = UtteranceMultipleCategoryClassificationPipe
    ValidSampler = FixedBatchSizeBatchSampler
    TestData = UtteranceMultipleCategoryClassificationPipe
    TestSampler = FixedBatchSizeBatchSampler
    Downstream = MeanPoolingLinear
    Task = UtteranceMultiClassClassificationTask

    default_config = Container(
        Corpus=dict(),
        TrainData=dict(
            train_category_encoder=True,
        ),
        TrainSampler=dict(
            max_timestamp=16000 * 200,
            shuffle=True,
        ),
        ValidData=dict(),
        ValidSampler=dict(
            batch_size=2,
        ),
        TestData=dict(),
        TestSampler=dict(
            batch_size=2,
        ),
        Downstream=dict(
            hidden_size=256,
        ),
        Task=dict(),
        Optimizer=dict(
            cls="torch.optim.Adam",
            lr=1.0e-4,
        ),
        Trainer=dict(
            total_steps=1000,
            log_step=100,
            valid_step=500,
            save_step=100,
            gradient_clipping=1,
            gradient_accumulate_steps=4,
            use_valid=True,
            valid_metric="accuracy",
            valid_higher_better=True,
        ),
    )
