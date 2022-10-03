import torch


def unfold_segments(tensor, tgt_duration, sample_rate=16000):
    seg_lengths = int(tgt_duration * sample_rate)
    src_lengths = len(tensor)
    step = seg_lengths // 2
    tgt_lengths = (
        seg_lengths if src_lengths <= seg_lengths else (src_lengths // step + 1) * step
    )

    pad_lengths = tgt_lengths - src_lengths
    padded_tensor = torch.cat([tensor, torch.zeros(pad_lengths).to(tensor.device)])
    segments = padded_tensor.unfold(0, seg_lengths, step).unbind(0)

    return segments
