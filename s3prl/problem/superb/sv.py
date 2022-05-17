from s3prl.corpus.voxceleb1sv import VoxCeleb1SV
from s3prl.dataset.speaker_verification_pipe import (
    SpeakerClassificationPipe,
)
from s3prl.sampler import (
    MaxTimestampBatchSampler,
    FixedBatchSizeBatchSampler,
)
from s3prl.task.speaker_verification_task import SpeakerVerification
from s3prl.nn import speaker_embedding_extractor
from s3prl import Container


class SuperbSV:
    Corpus = VoxCeleb1SV
    TrainData = SpeakerClassificationPipe
    TrainSampler = MaxTimestampBatchSampler
    ValidData = SpeakerClassificationPipe
    ValidSampler = FixedBatchSizeBatchSampler
    TestData = SpeakerClassificationPipe
    TestSampler = FixedBatchSizeBatchSampler
    Downstream = speaker_embedding_extractor
    Task = SpeakerVerification

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
            output_size=1500,
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
