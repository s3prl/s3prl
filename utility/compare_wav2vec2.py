import os
import torch
import argparse

s3prl_path = os.path.dirname(os.path.abspath(__file__)) + "/../"

parser = argparse.ArgumentParser()
parser.add_argument("--base", action="store_true")
parser.add_argument("--large", action="store_true")
parser.add_argument("--wav_normalize", action="store_true")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

assert args.base or args.large
fairseq_str = "wav2vec2_base_960" if args.base else "wav2vec2_large_ll60k"
huggingface_str = "wav2vec2_hug_base_960" if args.base else "wav2vec2_hug_large_ll60k"

fairseq = torch.hub.load(s3prl_path, fairseq_str, source="local").to(args.device)
huggingface = torch.hub.load(s3prl_path, huggingface_str, source="local").to(
    args.device
)

if args.wav_normalize:
    fairseq.wav_normalize = True

fairseq.eval()
huggingface.eval()

wavs = [torch.randn(16020).to(args.device), torch.randn(16040).to(args.device)]

with torch.no_grad():
    hiddens1 = fairseq(wavs)["hidden_states"]
    hiddens2 = huggingface(wavs)["hidden_states"]
    assert len(hiddens1) == len(hiddens2)

    diffs = []
    for idx, (hidden1, hidden2) in enumerate(zip(hiddens1, hiddens2)):
        diff = (hidden1 - hidden2).abs().max().item()
        print(f"hidden {idx} difference: {diff}")
        diffs.append(diff)

    print(f"Max difference: {torch.FloatTensor(diffs).max().item()}")
