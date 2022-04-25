from s3prl.corpus.librispeech import LibriSpeechForSUPERB
from s3prl.dataset.speech2text_pipe import Speech2TextPipe
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task.speech2text_ctc_task import Speech2TextCTCTask
from s3prl.nn import RNNEncoder
from s3prl import Container
from s3prl.encoder.tokenizer import CharacterTokenizer


class SuperbASR:
    Corpus = LibriSpeechForSUPERB
    TrainData = Speech2TextPipe
    TrainSampler = MaxTimestampBatchSampler
    ValidData = Speech2TextPipe
    ValidSampler = FixedBatchSizeBatchSampler
    TestData = Speech2TextPipe
    TestSampler = FixedBatchSizeBatchSampler
    Downstream = RNNEncoder
    Task = Speech2TextCTCTask
    tokenizer = CharacterTokenizer()

    default_config = Container(
        Corpus=dict(),
        TrainData=dict(tokenizer=tokenizer),
        TrainSampler=dict(
            max_timestamp=16000 * 1000,
            shuffle=True,
        ),
        ValidData=dict(tokenizer=tokenizer),
        ValidSampler=dict(
            batch_size=32,
        ),
        TestData=dict(tokenizer=tokenizer),
        TestSampler=dict(
            batch_size=1,
        ),
        Downstream=dict(
            module="LSTM",
            hidden_size=[1024, 1024],
            dropout=[0.2, 0.2],
            layer_norm=[False, False],
            proj=[False, False],
            sample_rate=[1, 1],
            sample_style="concat",
            bidirectional=True,
        ),
        Task=dict(tokenizer=tokenizer),
        Optimizer=dict(
            cls="torch.optim.Adam",
            lr=1.0e-4,
        ),
        Trainer=dict(
            total_steps=200000,
            log_step=100,
            valid_step=2000,
            save_step=500,
            gradient_clipping=1,
            gradient_accumulate_steps=1,
            use_valid=True,
            valid_metric="wer",
            valid_higher_better=False,
        ),
    )
