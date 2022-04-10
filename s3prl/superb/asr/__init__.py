from s3prl.task import Speech2TextCTC as Task
from s3prl.nn import RNNEncoder as DownstreamModel

from s3prl.dataset import Speech2TextDataset as TrainDataset
from s3prl.sampler import MaxTimestampBatchSampler as TrainSampler

from s3prl.dataset import Speech2TextDataset as ValidDataset
from s3prl.sampler import FixedBatchSizeBatchSampler as ValidSampler

from s3prl.dataset import Speech2TextDataset as TestDataset
from s3prl.sampler import FixedBatchSizeBatchSampler as TestSampler

from s3prl.preprocessor import LibriSpeechCTCPreprocessor as Preprocessor
