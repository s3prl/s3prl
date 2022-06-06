from torch.nn import L1Loss

from s3prl import Container
from s3prl.corpus.librispeech_for_pretrain import LibriSpeechForPretrain
from s3prl.dataset.pretrain_apc_pipe import PretrainTaskPipe
from s3prl.nn.identity import Identity
from s3prl.nn.rnn_apc import ApcModel
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task.autoregressive_reconstruction_task import (
    AutoregressiveReconstructionTask,
)


class Apc:
    Corpus = LibriSpeechForPretrain
    TrainData = PretrainTaskPipe
    TrainSampler = MaxTimestampBatchSampler
    ValidData = PretrainTaskPipe
    ValidSampler = FixedBatchSizeBatchSampler
    TestData = PretrainTaskPipe
    TestSampler = FixedBatchSizeBatchSampler
    Body = ApcModel
    Head = Identity
    Task = AutoregressiveReconstructionTask
    Loss = L1Loss

    input_size = 80

    default_config = Container(
        Corpus=dict(
            train_split=["train-clean-100", "train-clean-360", "train-other-500"]
        ),
        TrainData=dict(
            n_future=5,
            audio_config=dict(
                feat_type="fbank",  # Feature type
                feat_dim=input_size,  # Feature dimension
                frame_length=25,  # Window size in ms
                frame_shift=10,  # Hop size in ms
                decode_wav=False,
                cmvn=True,  # Apply uttr.-wised CMVN on Mel spectrogram
            ),
            n_jobs=8,
        ),
        TrainSampler=dict(
            max_timestamp=16000 * 20,
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
        Body=dict(
            input_size=input_size,
            num_layers=3,
            hidden_size=512,
            dropout=0.1,
            residual=True,
        ),
        Head=dict(),
        Task=dict(),
        Loss=dict(),
        Optimizer=dict(
            cls="torch.optim.AdamW",
            lr=0.0001,  # set to 0.00001 for some datasets if you encounter NaN during training
        ),
        Trainer=dict(
            total_steps=1000000,
            log_step=50000,
            valid_step=50000,
            save_step=50000,
            gradient_clipping=5.0,
            gradient_accumulate_steps=4,
            use_valid=True,
            valid_metric="loss",
            valid_higher_better=False,
        ),
    )
