from dotenv import dotenv_values
from s3prl import Container
from s3prl.corpus.hear import dcase_2016_task2
from s3prl.dataset.hear_timestamp import HearTimestampDatapipe


def test_dcase_2016_task2():
    DCASE_ROOT = dotenv_values()["DCASE2016_TASK2"]
    train_data, valid_data, test_data = dcase_2016_task2(DCASE_ROOT).slice(3)
    train_dataset = HearTimestampDatapipe(feat_frame_shift=160)(train_data)
    item = train_dataset[0]
    for key in ["x", "x_len", "y", "y_len"]:
        assert key in item
