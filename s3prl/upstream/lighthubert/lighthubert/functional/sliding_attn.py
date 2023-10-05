# --------------------------------------------------------
# LightHuBERT: Lightweight and Configurable Speech Representation Learning with Once-for-All Hidden-Unit BERT (https://arxiv.org/pdf/2203.15610.pdf)
# Github source: https://github.com/mechanicalsea/lighthubert
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
Sliding window attention.
- `sliding_attn` implements it via `torch.as_strided` with reducing memory and speedup computation.
- `sliding_attn_check_mask` implements it via `attn_mask` as global attention.
- `global_attention_forward` is global attention with full-utterance sliding window size.
"""

import torch
import torch.nn.functional as F


def pad_as_attn_swz(query, key, value, swz, attn_mask=None, key_padding_mask=None):
    """
    Force the query, key, and value to be the length of the multiple of sliding window size.
        We add zero at the beginning of them but does not mask them to avoid grad nan.

    Args:
        query, key, value:
            3-d tensor (batch_size, seq_len, hidden_dim)
        swz:
            int, sliding attention window size
        attn_mask:
            2-d bool tensor (tgt_len, src_len)
            3-d bool tensor (bsz, tgt_len, src_len)
        key_padding_mask:
            2-d bool tensor (batch_size, max_seq_len)

    Returns:
        query, key, value, attn_mask, key_padding_mask, tgt_len, src_len
    """
    if (
        isinstance(swz, int)
        and query.size(1) % swz == 0
        and key.size(1) % swz == 0
        and value.size(1) % swz == 0
    ):
        return (
            query,
            key,
            value,
            attn_mask,
            key_padding_mask,
            query.size(1),
            key.size(1),
        )

    tgt_len, src_len = query.size(1), key.size(1)
    pad_tgt = (tgt_len // swz + 1) * swz - tgt_len if tgt_len % swz > 0 else 0
    pad_src = (src_len // swz + 1) * swz - src_len if src_len % swz > 0 else 0
    if attn_mask is not None:
        # assert attn_mask.dtype == torch.bool
        attn_mask = F.pad(attn_mask, (pad_src, 0, pad_tgt, 0), value=False)
    if key_padding_mask is not None:
        assert key_padding_mask.ndim == 2
        assert key_padding_mask.dtype == torch.bool
        key_padding_mask = F.pad(key_padding_mask, (pad_src, 0), value=False)
    query = F.pad(query, (0, 0, pad_tgt, 0))
    key = F.pad(key, (0, 0, pad_src, 0))
    value = F.pad(value, (0, 0, pad_src, 0))
    return (query, key, value, attn_mask, key_padding_mask, tgt_len, src_len)


@torch.no_grad()
def merge_padding_attm_mask(attn_mask, key_padding_mask, num_heads, tgt_len=None):
    r"""
    Merge `key_padding_mask` into `attn_mask`.

    Args:
        attn_mask:
            2-d bool tensor (tgt_len, src_len) or
            3-d bool tensor (batch_size * num_heads, tgt_len, src_len)
            attn_mask[batch_i, t_j, t_k] = True or False
        key_padding_mask:
            1-d tensor long (batch_size) or 2-d tensor bool (batch_size, src_len)
        num_heads:
            int, the number of heads

    Return:
        attn_mask:
            2-d bool tensor (tgt_len, src_len) or
            3-d bool tensor (batch_size * num_heads, 1, src_len) or
            3-d bool tensor (batch_size * num_heads, tgt_len, src_len)
    """
    if key_padding_mask is None:
        return attn_mask
    else:
        assert num_heads is not None
    if key_padding_mask.ndim == 1:
        key_padding_mask = mask_padding(key_padding_mask)
    bsz, src_len = key_padding_mask.size()
    key_padding_mask = (
        key_padding_mask.view(bsz, 1, 1, src_len)
        .expand(bsz, num_heads, 1, src_len)
        .reshape(bsz * num_heads, 1, src_len)
    )
    if attn_mask is None:
        if key_padding_mask.size(1) == 1:
            assert tgt_len is not None
            attn_mask = key_padding_mask.expand(bsz * num_heads, tgt_len, src_len)
        return attn_mask
    assert attn_mask.dtype == torch.bool, f"{attn_mask.dtype}"
    tgt_len = attn_mask.size(-2) if tgt_len is None else tgt_len
    if attn_mask.ndim == 2:
        attn_mask = attn_mask.unsqueeze(0).expand(bsz * num_heads, tgt_len, src_len)
    assert src_len == attn_mask.size(-1)
    key_padding_mask = key_padding_mask.expand(-1, tgt_len, src_len)
    attn_mask = attn_mask.logical_or(key_padding_mask)
    return attn_mask


def mask_padding(seq_len: torch.LongTensor, max_len=None):
    r"""
    Args:
        seq_len: 1-d long tensor (batch_size)

    Returns:
        key_padding_mask: 2-d bool tensor
    """
    max_len = max(seq_len) if max_len is None else max_len
    key_padding_mask = torch.arange(max_len, device=seq_len.device).unsqueeze(0)
    key_padding_mask = key_padding_mask.expand(seq_len.size(0), -1)
    return key_padding_mask >= seq_len.unsqueeze(1)


def size_check(q, k, v):
    bsz, tgt_len, src_len, head_dim = q.size(0), q.size(1), k.size(1), q.size(2)
    assert q.ndim == k.ndim == v.ndim == 3
    assert (
        q.size(0) == k.size(0) == v.size(0)
    ), f"{q.size(0)} ?= {k.size(0)} ?= {v.size(0)}"
    assert tgt_len <= src_len, f"tgt_len {tgt_len} > src_len {src_len}"
    assert k.size(1) == v.size(1), f"seq_len: {k.size(1)} != {v.size(1)}"
    assert (
        q.size(2) == k.size(2) == v.size(2)
    ), f"{q.size(1)} != {k.size(1)} != {v.size(1)}"
    return bsz, tgt_len, src_len, head_dim


def sliding_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    swz: int,
    attn_mask=None,
    dropout_p=0,
    training=False,
):
    r"""
    Sliding window attention.

    Args:
        q, k, v:
            3-dim tensor, e.g., (batch_size * head_num, seq_len, head_dim)
            q is a scaled q by sqrt(head_dim).
        swz:
            int, swz % 2 == 0, sliding window size
        attn_mask:
            2-d bool tensor (tgt_len, src_len) or
            3-d bool tensor (bsz, tgt_len, src_len)

    Notes:
        `torch.as_strided` does not copy data, like `torch.Tensor.view`.
        Pad K matrix with value of -inf, but pad V matrix with value of 0.
    """
    bsz, tgt_len, src_len, head_dim = size_check(q, k, v)
    assert tgt_len == src_len and tgt_len % swz == 0, f"{tgt_len}, {src_len}, {swz}"

    # q * k
    csz = tgt_len // (swz // 2) - 1
    q_chunk = q.view(bsz, tgt_len // swz, swz, head_dim).as_strided(
        size=(bsz, csz, swz, head_dim),
        stride=(tgt_len * head_dim, swz // 2 * head_dim, head_dim, 1),
    )
    k_chunk = k.view(bsz, src_len // swz, swz, head_dim).as_strided(
        size=(bsz, csz, swz, head_dim),
        stride=(src_len * head_dim, swz // 2 * head_dim, head_dim, 1),
    )
    weights_chunk = torch.einsum(
        "bcxd,bcyd->bcxy", q_chunk, k_chunk
    )  # (bsz, csz, swz, swz)
    chunk1 = weights_chunk.as_strided(
        size=(bsz, csz, swz // 2, swz // 2 + 1),
        stride=(csz * swz * swz, swz * swz, swz + 1, 1),
    )
    chunk2 = F.pad(
        weights_chunk[:, -1, swz // 2 :, swz // 2 :],
        pad=(0, swz // 2),
        value=float("-inf"),
    ).as_strided(
        size=(bsz, swz // 2, swz // 2 + 1),
        stride=((swz // 2) * swz, swz + 1, 1),
    )
    chunk3 = F.pad(
        weights_chunk[:, 0, : swz // 2, : swz // 2],
        pad=(swz // 2, 0),
        value=float("-inf"),
    ).as_strided(
        size=(bsz, swz // 2, swz // 2),
        stride=((swz // 2) * swz, swz + 1, 1),
    )
    chunk4 = (
        weights_chunk[:, :, swz // 2 :, :]
        .contiguous()
        .as_strided(
            size=(bsz, csz, swz // 2, swz // 2),
            stride=(csz * (swz // 2) * swz, (swz // 2) * swz, swz + 1, 1),
        )
    )
    attn_weights = torch.cat(
        [
            torch.cat([chunk3.unsqueeze(1), chunk4], dim=1),
            torch.cat([chunk1, chunk2.unsqueeze(1)], dim=1),
        ],
        dim=3,
    )
    attn_weights = attn_weights.view(bsz, tgt_len, swz + 1)

    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0)
        attn_mask = F.pad(attn_mask, (swz // 2, swz // 2), value=True)
        attn_mask = attn_mask.as_strided(
            size=(1, tgt_len, swz + 1),
            stride=(tgt_len * (src_len + swz), src_len + swz + 1, 1),
        )
        attn_mask = attn_mask.to(attn_weights.dtype).masked_fill(
            attn_mask, float("-inf")
        )
        attn_weights += attn_mask

    attn_probs = F.softmax(attn_weights.to(torch.float32), dim=-1)
    attn_probs = attn_probs.to(attn_weights.dtype)
    attn_probs = F.dropout(attn_probs, p=dropout_p, training=training)

    # probs * v
    kpad_len1 = swz // 2
    kpad_len2 = max(swz // 2 - (src_len - tgt_len), 0)
    v_pad = F.pad(v, (0, 0, kpad_len1, kpad_len2), value=0)  # (bsz, tgt_len, )
    csz = tgt_len // (swz // 2) - 1
    assert tgt_len % (swz // 2) == 0, f"{tgt_len} % {swz // 2} != 0"
    attn_probs = attn_probs.view(bsz, csz + 1, swz // 2, swz + 1)
    attn_probs_pad = F.pad(
        attn_probs, (0, swz // 2 - 1)
    )  # (bsz, tgt_len // (swz // 2), swz // 2, swz + swz // 2)
    attn_probs_pad = attn_probs_pad.as_strided(
        size=(bsz, csz + 1, swz // 2, swz + swz // 2),
        stride=attn_probs_pad.stride()[:2] + (swz + swz // 2 - 1, 1),
    )
    v_size = (bsz, csz + 1, swz + swz // 2, head_dim)
    v_stride = (v_pad.stride(0), swz // 2 * v_pad.stride(1), head_dim, 1)
    v_chunk = v_pad.as_strided(v_size, v_stride)
    attn_chunk = torch.einsum("bcdw,bcwh->bcdh", attn_probs_pad, v_chunk)
    attn = attn_chunk.view(bsz, tgt_len, head_dim)

    return attn, attn_weights


def sliding_attn_check_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    swz: int,
    attn_mask=None,
    dropout_p=0,
    training=False,
    ret_slide=True,
):
    r"""
    Check sliding attention dot product via sliding `attn_mask`.

    Args:
        q, k, v: tensor, (batch_size * head_num, seq_len, head_dim)
        swz: int, swz % 2 == 0, sliding window size
        attn_mask:
            2-d bool tensor (tgt_len, src_len) or
            3-d bool tensor (bsz, tgt_len, src_len)
        ret_slide:
            bool, if true, return sliding `attn_weights`

    Notes:
        The implementation keep compuation as full attention.
    """
    bsz, tgt_len, src_len, head_dim = size_check(q, k, v)
    with torch.no_grad():  # sliding attn mask
        swa_mask = torch.triu(q.new_ones(tgt_len, src_len), swz // 2 + 1) + torch.tril(
            q.new_ones(tgt_len, src_len), -(swz // 2 + 1)
        )
        swa_mask = swa_mask.unsqueeze(0).masked_fill(
            swa_mask.unsqueeze(0).to(torch.bool), float("-inf")
        )

    attn_weights = torch.bmm(q, k.transpose(1, 2))
    attn_weights += swa_mask

    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0)
        attn_mask = attn_mask.to(attn_weights.dtype).masked_fill(
            attn_mask, float("-inf")
        )
        attn_weights += attn_mask

    attn_probs = F.softmax(attn_weights.to(torch.float32), dim=-1)
    attn_probs = attn_probs.to(attn_weights.dtype)
    attn_probs = F.dropout(attn_probs, p=0, training=False)
    attn = torch.bmm(attn_probs, v)

    if not ret_slide:
        return attn, attn_weights

    kpad_len1 = swz // 2
    kpad_len2 = max(swz // 2 - (src_len - tgt_len), 0)
    attn_weights_pad = F.pad(attn_weights, (kpad_len1, kpad_len2), value=float("-inf"))
    attn_weights_slide = torch.as_strided(
        attn_weights_pad,
        size=(bsz, tgt_len, swz + 1),
        stride=(tgt_len * attn_weights_pad.size(2), attn_weights_pad.size(2) + 1, 1),
    ).contiguous()
    attn_probs_slide = F.softmax(attn_weights_slide, dim=-1)

    return attn, attn_weights_slide, attn_weights


def slide_window_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    swz: int,
    attn_mask=None,
    key_padding_mask=None,
    num_heads=None,
    dropout_p=0,
    training=False,
    mode="stride",
):
    r"""
    Sliding window attention.

    Args:
        q, k, v:
            3-dim tensor, e.g., (batch_size * head_num, seq_len, head_dim)
            q is a scaled q by sqrt(head_dim).
        swz:
            int, swz % 2 == 0, sliding window size
        attn_mask:
            2-d bool tensor (tgt_len, src_len) or
            3-d bool tensor (bsz, tgt_len, src_len)
        key_padding_mask:
            1-d long tensor (bsz) or
            2-d bool tensor (bsz, src_len)
        num_heads:
            int, required to merge `attn_mask` and `key_padding_mask`
        dropout_p:
            float, [0, 1], drop out `attn_probs` randomly.
        training:
            bool, it serves in `dropout`
        mode:
            'mask': via `attn_mask`
            'stride': via `torch.as_strided`
    Notes:
        1. `torch.as_strided` does not copy data, like `torch.Tensor.view`, and
        requires padding utterances to be the length of the multiple of `swz`.
        2. Pad K matrix with value of -inf, but pad V matrix with value of 0.
        3. Numerical error occurs while small `swz` is applied with `float16`,
        e.g., swz = 16, seq_len = 512, q.dtype = torch.float16
    """
    assert mode in ["mask", "stride"]
    assert swz > 0 and swz % 2 == 0, f"{swz} ? > 0 and % 2 == 0"
    q, k, v, attn_mask, key_padding_mask, ori_tgt_len, ori_src_len = pad_as_attn_swz(
        q, k, v, swz, attn_mask=attn_mask, key_padding_mask=key_padding_mask
    )
    attn_mask = merge_padding_attm_mask(
        attn_mask, key_padding_mask, num_heads, q.size(1)
    )
    _, tgt_len, src_len, _ = size_check(q, k, v)
    if mode == "mask":
        attn, attn_weights = sliding_attn_check_mask(
            q,
            k,
            v,
            swz,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            training=training,
            ret_slide=False,
        )
        attn = attn[:, tgt_len - ori_tgt_len :].contiguous()
        attn_weights = attn_weights[
            :, tgt_len - ori_tgt_len :, src_len - ori_src_len :
        ].contiguous()
    else:  # mode == "stride"
        assert (
            tgt_len == src_len and src_len % swz == 0
        ), f"{tgt_len} == {src_len} and {src_len} % {swz} == 0"
        attn, attn_weights = sliding_attn(
            q, k, v, swz, attn_mask=attn_mask, dropout_p=dropout_p, training=training
        )
        attn = attn[:, tgt_len - ori_tgt_len :].contiguous()
        attn_weights = attn_weights[:, tgt_len - ori_tgt_len :].contiguous()
    return attn, attn_weights


def global_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask=None,
    key_padding_mask=None,
    num_heads=None,
    dropout_p=0,
    training=False,
):
    r"""
    Full/Global attention dot product as sliding attention with
        full-utterance window size.

    Args:
        q, k, v: tensor, (batch_size * head_num, seq_len, head_dim)
        attn_mask:
            2-d bool tensor (tgt_len, src_len) or
            3-d bool tensor (bsz, tgt_len, src_len)
        key_padding_mask:
            2-d bool tensor (bsz, src_len)

    Returns:
        attn, attn_weights: attention and unnormalized attention weights.
    """
    attn_weights = torch.bmm(q, k.transpose(1, 2))
    attn_mask = merge_padding_attm_mask(
        attn_mask, key_padding_mask, num_heads, q.size(1)
    )
    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0)
        attn_mask = attn_mask.to(attn_weights.dtype).masked_fill(
            attn_mask, float("-inf")
        )
        attn_weights += attn_mask
    attn_probs = F.softmax(attn_weights.to(torch.float32), dim=-1)
    attn_probs = attn_probs.to(attn_weights.dtype)
    attn_probs = F.dropout(attn_probs, p=0, training=False)
    attn = torch.bmm(attn_probs, v)

    return attn, attn_weights


def latency(
    func, inputs: dict, device="cpu", warmup_steps=50, measure_steps=100, backward=False
):
    import time

    if not backward:
        with torch.no_grad():
            if device != "cpu":
                torch.cuda.synchronize(device=device)
            for i in range(warmup_steps):
                func(**inputs)
            if device != "cpu":
                torch.cuda.synchronize(device=device)

            if device != "cpu":
                torch.cuda.synchronize(device=device)
            st = time.time()
            for i in range(measure_steps):
                func(**inputs)
            if device != "cpu":
                torch.cuda.synchronize(device=device)
            ed = time.time()
            total_time = ed - st
    else:
        original_grad = list(inputs.values())[0].requires_grad
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.dtype != torch.bool:
                value.requires_grad = True
        if device != "cpu":
            torch.cuda.synchronize(device=device)
        for i in range(warmup_steps):
            ans = func(**inputs)[0].mean()
            ans.backward()
        if device != "cpu":
            torch.cuda.synchronize(device=device)

        if device != "cpu":
            torch.cuda.synchronize(device=device)
        st = time.time()
        for i in range(measure_steps):
            ans = func(**inputs)[0].mean()
            ans.backward()
        if device != "cpu":
            torch.cuda.synchronize(device=device)
        ed = time.time()
        total_time = ed - st
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.dtype != torch.bool:
                value.requires_grad = original_grad

    avg_time = total_time / measure_steps * 1000  # ms

    return avg_time


if __name__ == "__main__":
    """
    python fairseq/models/compress_hubert/functional/sliding_attn.py --device cuda --swz 64 --len 510 --bsz 48 --half
    """
    print(__doc__)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--cross", action="store_true", help="cross attention uses different q, k, v"
    )
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--swz", default=64, type=int)
    parser.add_argument("--len", default=510, type=int)
    parser.add_argument(
        "--ken", default=None, type=int, help="applied while cross attention"
    )
    parser.add_argument("--bsz", default=48, type=int)
    args = parser.parse_args()

    size = (args.bsz, args.len, 64)
    dsize = (args.bsz, args.len if args.ken is None else args.ken, 64)
    if args.cross:
        q, k, v = torch.rand(size), torch.rand(dsize), torch.rand(dsize)
    else:
        q = k = v = torch.rand(size)
    device = args.device
    q, k, v = q.to(device), k.to(device), v.to(device)
    if args.double:
        q, k, v = q.double(), k.double(), v.double()
    if args.half:
        q, k, v = q.half(), k.half(), v.half()
    swz = args.swz

    attn_mask = torch.rand((q.size(1), k.size(1))).to(device=device).to(q.dtype) > 0.5

    avg_time = latency(
        global_attention_forward,
        {"q": q, "k": k, "v": v, "attn_mask": attn_mask},
        device=device,
    )
    print(f"global_attention_forward: {avg_time} ms")
    avg_time = latency(
        slide_window_attention_forward,
        {"q": q, "k": k, "v": v, "swz": swz, "attn_mask": attn_mask, "mode": "stride"},
        device=device,
    )
    print(f"sliding_attn [slide]: {avg_time} ms")
    avg_time = latency(
        slide_window_attention_forward,
        {"q": q, "k": k, "v": v, "swz": swz, "attn_mask": attn_mask, "mode": "mask"},
        device=device,
    )
    print(f"sliding_attn [mask]: {avg_time} ms")
    avg_time = latency(
        sliding_attn_check_mask,
        {"q": q, "k": k, "v": v, "swz": swz, "attn_mask": attn_mask, "ret_slide": True},
        device=device,
    )
    print(f"sliding_attn_check_mask: {avg_time} ms")

    a0, aw0 = slide_window_attention_forward(
        q, k, v, swz, attn_mask=attn_mask, mode="stride"
    )
    a1, aw1 = slide_window_attention_forward(
        q, k, v, swz, attn_mask=attn_mask, mode="mask"
    )
    a2, aw2, aw_full = sliding_attn_check_mask(q, k, v, swz, attn_mask=attn_mask)

    print("Sliding vs. Masked:", torch.allclose(a0, a1))
    diff_a = "%.3f - %.3f = %.6e" % (
        a0.view(-1)[(a0 - a2).abs().argmax()].item(),
        a2.view(-1)[(a0 - a2).abs().argmax()].item(),
        (a0 - a2).abs().max().item(),
    )
    aw0s = aw0.softmax(-1)
    aw2s = aw2.softmax(-1)
    diff_aw = "%.3f - %.3f = %.6e" % (
        aw0s.view(-1)[(aw0s - aw2s).abs().argmax()].item(),
        aw2s.view(-1)[(aw0s - aw2s).abs().argmax()].item(),
        (aw0s - aw2s).abs().max().item(),
    )
    print(
        f"Sliding vs. Masked: {torch.allclose(a0, a2)} ({diff_a}) {torch.allclose(aw0s, aw2s)} ({diff_aw})"
    )

    if args.profile:
        with torch.autograd.profiler.profile(
            use_cuda=True, with_stack=False, profile_memory=True
        ) as prof:
            a0, aw0 = slide_window_attention_forward(
                q, k, v, swz, attn_mask=attn_mask, mode="stride"
            )
        print("sliding_attn", prof.table())

        with torch.autograd.profiler.profile(
            use_cuda=True, with_stack=False, profile_memory=True
        ) as prof:
            a2, aw2 = slide_window_attention_forward(
                q, k, v, swz, attn_mask=attn_mask, mode="mask"
            )
        print("sliding_attn_check_mask", prof.table())
