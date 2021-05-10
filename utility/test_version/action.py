import torch
import argparse
import torchaudio

parser = argparse.ArgumentParser()
parser.add_argument('--upstream', '-u', required=True)
parser.add_argument('--filepath', '-f', required=True)
parser.add_argument('--outpath', '-o', required=True)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

wav, sr = torchaudio.load(args.filepath)
wav = wav.view(-1).to(args.device)

model = torch.hub.load('s3prl/s3prl:develop', args.upstream, force_reload=True).to(args.device)
with torch.no_grad():
    model.eval()
    feature = model([wav])[0]

torch.save(feature.cpu(), args.outpath)

