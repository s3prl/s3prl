import os
import argparse

import torch
import torchaudio
torchaudio.set_audio_backend('sox_io')

import hubconf
from utility.download import _urls_to_filepaths


def test(files, args):
    pths = [f'http://140.112.21.12:9000/extracted/{args.upstream}/' + file.split('/')[-1] + '.pth' for file in files]

    model = getattr(hubconf, args.upstream)().to(device=args.device)
    model.eval()

    with torch.no_grad():
        for file, pth in zip(files, pths):
            file_path, pth_path = _urls_to_filepaths(file, pth, refresh=True)
            wav, sr = torchaudio.load(file_path)
            wav = wav.view(-1).to(device=args.device)
            repre = model([wav])[0].detach().cpu()

            if not torch.allclose(repre, torch.load(pth_path), atol=args.atol):
                print(f'[torch.allclose] - Failed with {args.upstream} and {file.split("/")[-1]}')
            else:
                print('[torch.allclose] - Success.')


def extract(files, args):
    expdir = os.path.join(args.extract_dir, args.upstream)
    os.makedirs(expdir, exist_ok=True)

    model = getattr(hubconf, args.upstream)().to(device=args.device)
    model.eval()

    with torch.no_grad():
        for file in files:
            file_path = _urls_to_filepaths(file, refresh=True)
            wav, sr = torchaudio.load(file_path)
            wav = wav.view(-1).to(device=args.device)
            repre = model([wav])[0].detach().cpu()

            outpath = os.path.join(expdir, file.split('/')[-1] + '.pth')
            torch.save(repre, outpath)


def main():
    parser = argparse.ArgumentParser()

    # upstream settings
    upstreams = [attr for attr in dir(hubconf) if callable(getattr(hubconf, attr)) and attr[0] != '_']
    parser.add_argument('--mode', '-m', choices=['extract', 'test'], required=True)
    parser.add_argument('--upstream', '-u', choices=upstreams, required=True)
    parser.add_argument('--extract_dir', '-d', default='./extracted')
    parser.add_argument('--atol', '-a', type=float, default=0.02)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    files = [
        'http://140.112.21.12:9000/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac',
        'http://140.112.21.12:9000/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac',
    ]
    eval(f'{args.mode}')(files, args)


if __name__ == '__main__':
    main()
