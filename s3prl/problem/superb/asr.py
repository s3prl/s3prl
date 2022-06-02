from pydoc import classname

from s3prl import Container
from s3prl.corpus.librispeech import LibriSpeechForSUPERB
from s3prl.dataset.speech2text_pipe import Speech2TextPipe
from s3prl.encoder.tokenizer import CharacterTokenizer
from s3prl.nn import RNNEncoder
from s3prl.sampler import FixedBatchSizeBatchSampler, MaxTimestampBatchSampler
from s3prl.task.speech2text_ctc_task import Speech2TextCTCTask
from s3prl.util.configuration import default_cfg, override_parent_cfg

from .base import SuperbProblem


class SuperbASR(SuperbProblem):
    pass


#     @override_parent_cfg(
#         corpus=dict(
#             cls=LibriSpeechForSUPERB,
#             args=dict(
#                 dataset_root="???",
#             ),
#         ),
#         train_datapipe=dict(
#             cls=Speech2TextPipe,
#             args=dict(),
#         ),
#         train_sampler=dict(
#             cls=MaxTimestampBatchSampler,
#             args=dict(
#                 max_timestamp=16000 * 100,
#                 shuffle=True,
#             ),
#         ),
#         valid_datapipe=dict(
#             cls=Speech2TextPipe,
#             args=dict(),
#         ),
#         valid_sampler=dict(
#             cls=FixedBatchSizeBatchSampler,
#             args=dict(
#                 batch_size=32,
#             ),
#         ),
#         test_datapipe=dict(
#             cls=Speech2TextPipe,
#             args=dict(),
#         ),
#         test_sampler=dict(
#             cls=FixedBatchSizeBatchSampler,
#             args=dict(
#                 batch_size=1,
#             ),
#         ),
#         downstream=dict(
#             cls=RNNEncoder,
#             args=dict(
#                 module="LSTM",
#                 hidden_size=[1024, 1024],
#                 dropout=[0.2, 0.2],
#                 layer_norm=[False, False],
#                 proj=[False, False],
#                 sample_rate=[1, 1],
#                 sample_style="concat",
#                 bidirectional=True,
#             ),
#         ),
#         task=dict(
#             cls=Speech2TextCTCTask,
#             args=dict(),
#         ),
#     )
#     @classmethod
#     def setup_problem(
#         cls,
#         cfg: Container = None,
#     ):
#         super().setup_problem(cfg=cfg)

#     @override_parent_cfg(
#         optimizer=dict(
#             cls="torch.optim.Adam",
#             args=dict(
#                 lr=1.0e-4,
#             ),
#         ),
#         trainer=dict(
#             total_steps=200000,
#             log_step=100,
#             eval_step=2000,
#             save_step=500,
#             gradient_clipping=1.0,
#             gradient_accumulate_steps=1,
#             valid_metric="wer",
#             valid_higher_better=False,
#         ),
#     )
#     @classmethod
#     def train(cls, cfg: Container = None):
#         super().train(cfg=cfg)
