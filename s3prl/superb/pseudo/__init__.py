from s3prl.task import UtteranceClassification as Task
from s3prl.nn import MeanPooling as DownstreamModel

from s3prl.dataset import UtteranceClassificationDataset as TrainDataset
DevDataset = TrainDataset
TestDataset = TrainDataset

from s3prl.preprocessor import PseudoUtteranceClassificationPreprocessor as TrainPreprocessor
DevPreprocessor = TrainPreprocessor
TestPreprocessor = TrainPreprocessor
