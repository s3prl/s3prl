"""
Authors:
    - Leo (2022)
"""

from typing import Any, List, Tuple

import pandas as pd
import torch

from .base import Dataset
from .load_audio import LoadAudio

__all__ = [
    "chunking",
    "scale_labels_secs",
    "get_chunk_labels",
    "chunk_labels_to_frame_tensor_label",
    "FrameLabelDataset",
]


def chunking(
    start_sec: float,
    end_sec: float,
    chunk_secs: float,
    step_secs: float,
    use_unfull_chunks: bool = True,
) -> List[Tuple[float, float]]:
    """
    Produce chunks (start, end points) from a given start, end seconds

    Args:
        start_sec (float): The start second of the utterance
        end_sec (float): The end second of the utterance
        chunk_secs (float): The length (in seconds) of a chunked chunk
        step_secs (float): The stride seconds between chunks
        use_unfull_chunks (bool): Whether to produce chunks shorter than :code:`chunk_secs`
            at the end of the recording

    Returns:
        List[Tuple[float, float]]: Each tuple describes the starting point (in sec)
            and the ending point (in sec) of each chunk in order
    """

    start, end = start_sec, end_sec
    while end - start > 0:
        if end - start >= chunk_secs:
            yield start, start + chunk_secs

        elif use_unfull_chunks:
            yield start, end

        start = start + step_secs


def scale_labels_secs(labels: List[Tuple[Any, float, float]], ratio: float):
    """
    When the recording length is changed due to like pitch or speed manipulation,
    the start/end timestamp (in seconds) should also be changed

    Args:
        labels (List[Tuple[Any, float, float]]): each chunk label is in (label, start_sec, end_sec)
        ratio (float): the scaling ratio

    Returns:
        List[Tuple[Any, float, float]]: the scaled labels
    """
    assert ratio > 0
    return [(label, start * ratio, end * ratio) for label, start, end in labels]


def get_chunk_labels(
    start_sec: float,
    end_sec: float,
    labels: List[Tuple[Any, float, float]],
):
    """
    Given a pair a start, end points, filter out the relevant labels from the given :code:`labels`
    and refine the start/end points of each label to reside between :code:`start_sec` and :code:`end_sec`

    Args:
        start_sec (float): the starting point
        end_sec (float): the ending point
        labels (List[Tuple[Any, float, float]]): the chunk labels

    Returns:
        List[Tuple[str, float, float]]: filtered labels. Only the labels relevant to the assigned
            start/end point are left
    """

    for label, start, end in labels:
        assert start < end, f"start ({start}) >= end ({end})"
        if start >= end_sec:
            continue
        if end <= start_sec:
            continue
        yield label, max(start_sec, start), min(end_sec, end)


def chunk_labels_to_frame_tensor_label(
    start_sec: float,
    end_sec: float,
    labels: List[Tuple[int, float, float]],
    num_class: int,
    frame_shift: int,
    sample_rate: int = 16000,
):
    """
    Produce frame-level labels for the given chunk labels

    Args:
        start_sec (float): the starting point of the chunk
        end_sec (float): the ending point of the chunk
        labels (List[Tuple[int, float, float]]): the chunk labels, each label is a tuple
            in (class_id, start_sec, end_sec)
        num_class (int): number of classes
        frame_shift (int): produce a frame per :code:`frame_shift` samples
        sample_rate (int): the sample rate of the recording. default: 16000

    Returns:
        torch.FloatTensor: shape (num_frames, num_class).
            the binary frame labels for the given :code:`labels`
    """
    labels = get_chunk_labels(start_sec, end_sec, labels)

    duration = end_sec - start_sec
    num_frames = len(range(0, round(duration * sample_rate), frame_shift))

    frame_labels = torch.zeros(num_frames, num_class)
    for class_id, start, end in labels:
        assert start >= start_sec, f"{start} < {start_sec}"
        assert end >= start_sec, f"{end} < {start_sec}"

        start_frame = round((start - start_sec) * sample_rate) // frame_shift
        end_frame = round((end - start_sec) * sample_rate) // frame_shift
        frame_labels[start_frame : end_frame + 1, class_id] = 1.0

    return frame_labels


class FrameLabelDataset(Dataset):
    """
    Args:
        df (pd.DataFrame): the dataframe should have the following columns
            record_id (str), wav_path (str), duration (float), utt_id (str),
            label (int), start_sec (float), end_sec (float)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        num_class: int,
        frame_shift: int,
        chunk_secs: float,
        step_secs: float,
        use_unfull_chunks: bool = True,
        load_audio_conf: dict = None,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()

        self.num_class = num_class
        self.frame_shift = frame_shift
        self.sample_rate = sample_rate

        recording_df = df[["record_id", "wav_path", "duration"]].drop_duplicates()
        record_ids = recording_df["record_id"].tolist()
        record_to_labels = {}
        for record_id in record_ids:
            subset_df = df[df["record_id"] == record_id]
            labels = list(
                zip(subset_df["label"], subset_df["start_sec"], subset_df["end_sec"])
            )
            record_to_labels[record_id] = labels

        self.chunked_utts = []
        for _, row in recording_df.iterrows():
            chunks = chunking(
                0.0,
                row["duration"],
                chunk_secs,
                step_secs,
                use_unfull_chunks,
            )
            for chunk_id, (start, end) in enumerate(chunks):
                labels = list(
                    get_chunk_labels(start, end, record_to_labels[row["record_id"]])
                )
                self.chunked_utts.append(
                    {
                        "record_id": row["record_id"],
                        "chunk_id": chunk_id,
                        "wav_path": row["wav_path"],
                        "start_sec": start,
                        "end_sec": end,
                        "unique_name": f"{row['record_id']}-{start}-{end}",
                        "labels": labels,
                    }
                )

        def flatten(data: List[dict], key: str):
            return [item[key] for item in data]

        self.audio_loader = LoadAudio(
            flatten(self.chunked_utts, "wav_path"),
            flatten(self.chunked_utts, "start_sec"),
            flatten(self.chunked_utts, "end_sec"),
            **(load_audio_conf or {}),
        )

    def __len__(self) -> int:
        return len(self.chunked_utts)

    def getinfo(self, index: int):
        return self.chunked_utts[index]

    def __getitem__(self, index):
        info = self.getinfo(index)
        audio = self.audio_loader[index]
        label = chunk_labels_to_frame_tensor_label(
            info["start_sec"],
            info["end_sec"],
            info["labels"],
            self.num_class,
            self.frame_shift,
            self.sample_rate,
        )
        return {
            "x": audio["wav"],
            "x_len": audio["wav_len"],
            "y": label,
            "y_len": len(label),
            "unique_name": info["unique_name"],
            "labels": info["labels"],
            "record_id": info["record_id"],
            "chunk_id": info["chunk_id"],
        }
