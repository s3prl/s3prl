import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

__all__ = [
    "default_collate_fn",
]


def default_collate_fn(samples, padding_value: int = 0):
    """
    Each item in **DynamicItemDataset** is a dict
    This function pad (or transform into numpy list) a batch of dict

    Args:
        samples (List[dict]): Suppose each Container is in

            .. code-block:: yaml

                wav: a single waveform
                label: a single string

    Return:
        dict

        .. code-block:: yaml

            wav: padded waveforms
            label: np.array([a list of string labels])
    """
    assert isinstance(samples[0], dict)
    keys = samples[0].keys()
    padded_samples = dict()
    for key in keys:
        values = [sample[key] for sample in samples]
        if isinstance(values[0], int):
            values = torch.LongTensor(values)
        elif isinstance(values[0], float):
            values = torch.FloatTensor(values)
        elif isinstance(values[0], np.ndarray):
            values = [torch.from_numpy(value).float() for value in values]
            values = pad_sequence(values, batch_first=True, padding_value=padding_value)
        elif isinstance(values[0], torch.Tensor):
            values = pad_sequence(values, batch_first=True, padding_value=padding_value)
        else:
            values = np.array(values, dtype="object")
        padded_samples[key] = values
    return padded_samples
