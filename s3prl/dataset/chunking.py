from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass

from .base import AugmentedDynamicItemDataset, DataPipe


def _count_frames(data_len, size, step):
    return int((data_len - size + step) / step)


def _gen_frame_indices(
    data_length,
    size=2000,
    step=2000,
    use_last_samples=True,
):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step > 0:
            yield (i + 1) * step, data_length


@dataclass
class UnfoldChunkByFrame(DataPipe):
    """
    Given a dataset with items containing `start_sec_name` and `end_sec_name`.
    For each item, produce `((end_sec_name - start_sec_name) * sample_rate / feat_frame_shift) / chunk_frames`
    items with the smaller durations between `min_chunk_frames` and `max_chunk_frames`

    Args:
        sample_rate (int): The sample_rate of the audio.
        feat_frame_shift (int): The target feature's frame_shift
        min_chunk_frames (int): The min frames for a chunk
        max_chunk_frames (int): The max frames for a chunk
        step_frames (int): The step frames for each sliding window
        use_last_samples (bool): whether to drop the last samples of an utterance which cannot form an window
        start_sec_name (str): The key name for the starting second of the audio
        end_sec_name (str): The key name for the ending second of the audio
    """

    sample_rate: int = 16000
    feat_frame_shift: int = 160

    min_chunk_frames: int = 2000
    max_chunk_frames: int = 2000
    step_frames: int = 2000
    use_last_samples: bool = True

    start_sec_name: str = "start_sec"
    end_sec_name: str = "end_sec"

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        unfolded_items = OrderedDict()
        for item in dataset:
            key = item.pop("id")

            # TODO: Check whether int v.s. round affect the final result
            # Original SUPERB uses int, however it makes more sense to use round
            data_len = int(
                (item[self.end_sec_name] - item[self.start_sec_name])
                * self.sample_rate
                / self.feat_frame_shift
            )
            for unfold_index, (start, end) in enumerate(
                _gen_frame_indices(data_len, self.min_chunk_frames, self.step_frames)
            ):
                start_sec = (
                    item[self.start_sec_name]
                    + start * self.feat_frame_shift / self.sample_rate
                )
                end_sec = (
                    item[self.start_sec_name]
                    + end * self.feat_frame_shift / self.sample_rate
                )
                dur_sec = end_sec - start_sec
                utt_id = f"{key}_start-{start_sec}_end-{end_sec}_dur-{dur_sec}"
                subitem = deepcopy(item)
                subitem["unchunked_id"] = key
                subitem["chunk_index"] = unfold_index
                subitem[self.start_sec_name] = start_sec
                subitem[self.end_sec_name] = end_sec
                unfolded_items[utt_id] = subitem
        new_dataset = AugmentedDynamicItemDataset(unfolded_items)
        new_dataset.add_tools(dataset.all_tools(False))
        return new_dataset


@dataclass
class UnfoldChunkBySec(DataPipe):
    use_last_samples: bool = True
    min_chunk_secs: float = 2.5
    max_chunk_secs: float = 2.5
    step_secs: int = 2.5

    start_sec_name: str = "start_sec"
    end_sec_name: str = "end_sec"

    def forward(
        self, dataset: AugmentedDynamicItemDataset
    ) -> AugmentedDynamicItemDataset:
        unfolded_items = OrderedDict()
        for item in dataset:
            key = item.pop("id")
            for unfold_index, (start, end) in enumerate(
                _gen_frame_indices(
                    item[self.end_sec_name] - item[self.start_sec_name],
                    self.min_chunk_secs,
                    self.step_secs,
                )
            ):
                start_sec = item[self.start_sec_name] + start
                end_sec = item[self.start_sec_name] + end
                dur_sec = end_sec - start_sec
                utt_id = f"{key}_start-{start_sec}_end-{end_sec}_dur-{dur_sec}"
                subitem = deepcopy(item)
                subitem["unchunked_id"] = key
                subitem["chunk_index"] = unfold_index
                subitem[self.start_sec_name] = start_sec
                subitem[self.end_sec_name] = end_sec
                unfolded_items[utt_id] = subitem
        new_dataset = AugmentedDynamicItemDataset(unfolded_items)
        new_dataset.add_tools(dataset.all_tools(False))
        return new_dataset
