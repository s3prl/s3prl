from s3prl.util.configuration import default_cfg, field
from s3prl.corpus.hear import dcase_2016_task2

from .timestamp import HearTimestamp


class Dcase2016Task2(HearTimestamp):
    @default_cfg(
        HearTimestamp.setup.default_except(
            corpus=dict(
                CLS=field(
                    dcase_2016_task2,
                    "\nThe corpus class. You can add the **kwargs right below this CLS key",
                    str,
                ),
                dataset_root=field(
                    "/home/leo/d/datasets/hear-2021.0.6/tasks/dcase2016_task2-hear2021-full/",
                    "The root path of the corpus",
                    str,
                ),
            ),
        )
    )
    @classmethod
    def setup(cls, **cfg):
        super().setup(**cfg)
