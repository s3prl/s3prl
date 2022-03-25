from s3prl.task import UtteranceClassification as Task
from s3prl.nn import MeanPoolingLinear as DownstreamModel

from s3prl.dataset import UtteranceClassificationDataset as TrainDataset
from s3prl.sampler import MaxTimestampBatchSampler as TrainSampler

from s3prl.dataset import UtteranceClassificationDataset as ValidDataset
from s3prl.sampler import FixedBatchSizeBatchSampler as ValidSampler

from s3prl.dataset import UtteranceClassificationDataset as TestDataset
from s3prl.sampler import FixedBatchSizeBatchSampler as TestSampler

from s3prl.preprocessor import VoxCeleb1SIDPreprocessor as Preprocessor
