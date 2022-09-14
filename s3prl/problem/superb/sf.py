from s3prl.corpus.snips import snips_for_speech2text
from s3prl.dataset.speech2text_pipe import Speech2TextPipe
from s3prl.nn import RNNEncoder
from s3prl.nn.specaug import ModelWithSpecaug
from s3prl.sampler import FixedBatchSizeBatchSampler, SortedSliceSampler
from s3prl.task.speech2text_ctc_task import Speech2TextCTCTask
from s3prl.util.configuration import default_cfg
from s3prl.utility.download import _urls_to_filepaths

from .base import SuperbProblem

VOCAB_URL = "https://huggingface.co/datasets/s3prl/SNIPS/raw/main/character.txt"
SLOTS_URL = "https://huggingface.co/datasets/s3prl/SNIPS/raw/main/slots.txt"


class SuperbSF(SuperbProblem):
    @default_cfg(
        **SuperbProblem.setup.default_except(
            corpus=dict(
                CLS=snips_for_speech2text,
                dataset_root="???",
            ),
            train_datapipe=dict(
                CLS=Speech2TextPipe,
                generate_tokenizer=True,
                vocab_type="character-slot",
                vocab_file=_urls_to_filepaths(VOCAB_URL),
                slots_file=_urls_to_filepaths(SLOTS_URL),
            ),
            train_sampler=dict(
                CLS=SortedSliceSampler,
                batch_size=32,
                max_length=300000,
            ),
            valid_datapipe=dict(
                CLS=Speech2TextPipe,
            ),
            valid_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=1,
            ),
            test_datapipe=dict(
                CLS=Speech2TextPipe,
            ),
            test_sampler=dict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=1,
            ),
            downstream=dict(
                CLS=ModelWithSpecaug,
                model_cfg=dict(
                    CLS=RNNEncoder,
                    module="LSTM",
                    proj_size=1024,
                    hidden_size=[1024, 1024],
                    dropout=[0.2, 0.2],
                    layer_norm=[False, False],
                    proj=[False, False],
                    sample_rate=[1, 1],
                    sample_style="concat",
                    bidirectional=True,
                ),
                specaug_cfg=dict(
                    freq_mask_width_range=(0, 50),
                    num_freq_mask=4,
                    time_mask_width_range=(0, 40),
                    num_time_mask=2,
                ),
            ),
            task=dict(
                CLS=Speech2TextCTCTask,
                log_metrics=[
                    "wer",
                    "cer",
                    "slot_type_f1",
                    "slot_value_cer",
                    "slot_value_wer",
                    "slot_edit_f1_full",
                    "slot_edit_f1_part",
                ],
            ),
        )
    )
    @classmethod
    def setup(cls, **cfg):
        super().setup(**cfg)

    @default_cfg(
        **SuperbProblem.train.default_except(
            optimizer=dict(
                CLS="torch.optim.Adam",
                lr=1.0e-4,
            ),
            trainer=dict(
                total_steps=200000,
                log_step=100,
                eval_step=2000,
                save_step=500,
                gradient_clipping=1.0,
                gradient_accumulate_steps=1,
                valid_metric="slot_type_f1",
                valid_higher_better=True,
            ),
        )
    )
    @classmethod
    def train(cls, **cfg):
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
            setup=setup.default_cfg.deselect("workspace", "resume", "dryrun"),
            train=train.default_cfg.deselect("workspace", "resume", "dryrun"),
            inference=inference.default_cfg.deselect("workspace", "resume", "dryrun"),
        )
    )
    @classmethod
    def run(cls, **cfg):
        super().run(**cfg)
