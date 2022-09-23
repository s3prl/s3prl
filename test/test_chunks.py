from collections import OrderedDict

import torch

from s3prl.dataset.base import AugmentedDynamicItemDataset
from s3prl.dataset.chunking import UnfoldChunkByFrame, UnfoldChunkBySec
from s3prl.dataset.common_pipes import LoadAudio
from s3prl.dataset.multiclass_tagging import BuildMultiClassTagging
from s3prl.util.pseudo_data import pseudo_audio


def test_chunk_pipe():
    with pseudo_audio([8, 20, 15], 16000) as (audio_files, num_samples):
        data = {
            i: dict(wav_path=file, start_sec=0, end_sec=num_sample / 16000)
            for i, (file, num_sample) in enumerate(zip(audio_files, num_samples))
        }
        data = OrderedDict(data)
        dataset = UnfoldChunkByFrame(
            min_chunk_frames=25, step_frames=25, feat_frame_shift=320
        )(data)
        assert dataset[0]["end_sec"] - dataset[0]["start_sec"] == 25 * 320 / 16000
        dataset = LoadAudio()(dataset)
        assert len(dataset[0]["wav"]) == 25 * 320

    with pseudo_audio([8, 20, 15], 16000) as (audio_files, num_samples):
        data = {
            i: dict(wav_path=file, start_sec=0, end_sec=num_sample / 16000)
            for i, (file, num_sample) in enumerate(zip(audio_files, num_samples))
        }
        data = OrderedDict(data)
        dataset = UnfoldChunkBySec(min_chunk_secs=2.5, step_secs=2.5)(data)
        assert (
            dataset.data[list(dataset.data.keys())[0]]["end_sec"]
            - dataset.data[list(dataset.data.keys())[0]]["start_sec"]
            == 2.5
        )
        dataset = LoadAudio()(dataset)
        assert len(dataset[0]["wav"]) == 2.5 * 16000


def test_multiclass_pipe():
    with pseudo_audio([8, 20, 15], 16000) as (audio_files, num_samples):
        data = {
            "another": dict(
                wav_path=audio_files[0],
                start_sec=0,
                end_sec=5.3,
                segments={
                    "spk1": [(0, 3.2), (3.6, 4.3)],
                    "spk2": [(1.1, 4.1)],
                },
            ),
            "best": dict(
                wav_path=audio_files[1],
                start_sec=0,
                end_sec=20,
                segments={
                    "spk1": [(0, 1), (2, 3)],
                    "spk3": [(0.5, 1.1), (1.5, 2.6), (3.5, 10)],
                },
            ),
        }
        data = OrderedDict(data)
        dataset = BuildMultiClassTagging(feat_frame_shift=160)(data)
        assert len(dataset) == 2
        assert dataset[0]["multiclass_tag"].shape == (round(5.3 * 16000 / 160), 2)

        dataset = BuildMultiClassTagging(feat_frame_shift=320)(data)
        assert len(dataset) == 2
        assert dataset[0]["multiclass_tag"].shape == (round(5.3 * 16000 / 320), 2)

        dataset = UnfoldChunkBySec(min_chunk_secs=4, step_secs=4)(data)
        dataset: AugmentedDynamicItemDataset = BuildMultiClassTagging(
            feat_frame_shift=160
        )(dataset)
        assert dataset[0]["multiclass_tag"].shape == (round(4 * 16000 / 160), 2)
        assert dataset[1]["multiclass_tag"].shape == (round(1.3 * 16000 / 160), 2)
        encoder = dataset.get_tool("all_tag_category")
        first_spk = (
            torch.from_numpy(dataset[0]["multiclass_tag"]).argmax(dim=0)[0].item()
        )
        encoder.decode(first_spk) == "spk1"
