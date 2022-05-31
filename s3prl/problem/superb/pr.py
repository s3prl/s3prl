from s3prl import Container
from s3prl.corpus.librispeech import LibriSpeechForSUPERB
from s3prl.dataset.speech2phoneme_pipe import Speech2PhonemePipe
from s3prl.encoder.tokenizer import CharacterTokenizer
from s3prl.nn import RNNEncoder
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task.speech2text_ctc_task import Speech2TextCTCTask


class SuperbPR:
    Corpus = LibriSpeechForSUPERB
    TrainData = Speech2PhonemePipe
    TrainSampler = MaxTimestampBatchSampler
    ValidData = Speech2PhonemePipe
    ValidSampler = FixedBatchSizeBatchSampler
    TestData = Speech2PhonemePipe
    TestSampler = FixedBatchSizeBatchSampler
    Downstream = RNNEncoder
    Task = Speech2TextCTCTask

    default_config = Container(
        Corpus=dict(),
        TrainData=dict(),
        TrainSampler=dict(
            max_timestamp=16000 * 100,
            shuffle=True,
        ),
        ValidData=dict(),
        ValidSampler=dict(
            batch_size=16,
        ),
        TestData=dict(),
        TestSampler=dict(
            batch_size=1,
        ),
        Downstream=dict(
            module="LSTM",
            hidden_size=[1024, 1024, 1024],
            dropout=[0.2, 0.2, 0.2],
            layer_norm=[False, False, False],
            proj=[False, False, False],
            sample_rate=[1, 1, 1],
            sample_style="concat",
            bidirectional=True,
        ),
        Task=dict(),
        Optimizer=dict(
            cls="torch.optim.Adam",
            lr=1.0e-3,
        ),
        Trainer=dict(
            total_steps=100000,
            log_step=100,
            valid_step=1000,
            save_step=100,
            gradient_clipping=1,
            gradient_accumulate_steps=1,
            use_valid=True,
            valid_metric="wer",
            valid_higher_better=False,
        ),
    )
