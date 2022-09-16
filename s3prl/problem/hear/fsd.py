import logging

from s3prl import newdict
from s3prl.corpus.hear import hear_scene_trainvaltest
from s3prl.util.configuration import default_cfg, field
from s3prl.sampler import FixedBatchSizeBatchSampler

from .scene import HearScene

logger = logging.getLogger(__name__)


class FSD50k(HearScene):
    @default_cfg(
        **HearScene.setup.default_except(
            corpus=dict(
                CLS=field(
                    hear_scene_trainvaltest,
                    "\nThe corpus class. You can add the **kwargs right below this CLS key",
                    str,
                ),
                dataset_root=field(
                    "???",
                    "The root path of the corpus",
                    str,
                ),
            ),
            train_sampler=newdict(
                CLS=FixedBatchSizeBatchSampler,
                batch_size=10,
                shuffle=True,
            ),
            task=dict(
                prediction_type="multilabel",
                scores=["mAP", "top1_acc", "d_prime", "aucroc"],
            ),
        )
    )
    @classmethod
    def setup(cls, **cfg):
        super().setup(**cfg)

    @default_cfg(
        **HearScene.train.default_except(
            trainer=dict(
                total_steps=40000,
                log_step=100,
                eval_step=1000,
                save_step=100,
                valid_metric="mAP",
                valid_higher_better=True,
            )
        )
    )
    @classmethod
    def train(cls, **cfg):
        super().train(**cfg)

    @default_cfg(**HearScene.inference.default_cfg)
    @classmethod
    def inference(cls, **cfg):
        super().inference(**cfg)

    @default_cfg(
        **HearScene.run.default_except(
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
