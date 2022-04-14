import logging

import torch
from s3prl.dataset.base import AugmentedDynamicItemDataset
from s3prl.dataset.utterance_classification_dataset import UtteranceClassificationDatasetBuilder
from s3prl.sampler.max_timestamp_batch_sampler import MaxTimestampBatchSampler

logger = logging.getLogger(__name__)


class TestClass(UtteranceClassificationDatasetBuilder):
    def load_audio(self, wav_path: int, metadata: bool = False):
        if metadata:
            return dict(
                num_frames=16000 * 12,
            )
        else:
            return torch.randn(16000)


def test_utterance_classification_dataset_builder():
    data = {
        "1": {
            "wav_path": 3,
            "label": 4,
        },
        "2": {
            "wav_path": 10,
            "label": 5,
        },
        "3": {
            "wav_path": 11,
            "label": 6,
        },
    }
    train_dataset, stats = TestClass().build_train_data(data).split(1)
    test_dataset: AugmentedDynamicItemDataset = TestClass().build_data(
        data, **stats
    ).slice(1)
    for item in test_dataset:
        logger.warning(item)
