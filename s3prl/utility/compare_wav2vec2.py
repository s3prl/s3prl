import os
import torch
import random
import argparse

SAMPLE_RATE = 16000
BATCH_SIZE = 8


s3prl_path = os.path.dirname(os.path.abspath(__file__)) + "/../"

parser = argparse.ArgumentParser()
parser.add_argument("--base", action="store_true")
parser.add_argument("--large", action="store_true")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

assert args.base or args.large
s3prl_str = "wav2vec2_base_960" if args.base else "wav2vec2_large_ll60k"
huggingface_str = "wav2vec2_hug_base_960" if args.base else "wav2vec2_hug_large_ll60k"

s3prl = torch.hub.load(s3prl_path, s3prl_str, source="local").to(args.device)
huggingface = torch.hub.load(s3prl_path, huggingface_str, source="local").to(
    args.device
)

if args.base:
    s3prl.wav_normalize = True
    s3prl.apply_padding_mask = False
s3prl.numpy_wav_normalize = True

s3prl.eval()
huggingface.eval()

wavs = [
    torch.randn(random.randint(SAMPLE_RATE * 1, SAMPLE_RATE * 15)).to(args.device)
    for _ in range(BATCH_SIZE)
]

with torch.no_grad():
    hiddens1 = s3prl(wavs)["hidden_states"]
    hiddens2 = huggingface(wavs)["hidden_states"]
    assert len(hiddens1) == len(hiddens2)

    diffs = []
    for idx, (hidden1, hidden2) in enumerate(zip(hiddens1, hiddens2)):
        diff = (hidden1 - hidden2).abs().max().item()
        print(f"hidden {idx} difference: {diff}")
        diffs.append(diff)

    print(f"Max difference: {torch.FloatTensor(diffs).max().item()}")
