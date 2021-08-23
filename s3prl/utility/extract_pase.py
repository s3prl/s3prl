from pase.models.frontend import wf_builder
pase = wf_builder('cfg/frontend/PASE+.cfg').eval()
pase.load_pretrained('FE_e199.ckpt', load_last=True, verbose=True)
pase.cuda()

import sys
wav_path = sys.argv[1]
out_path = sys.argv[2]

# Now we can forward waveforms as Torch tensors
import torch
import torchaudio
torchaudio.set_audio_backend('sox')

x, sr = torchaudio.load(wav_path)
x = x.view(-1).cuda()

x = x.view(1, 1, -1)

with torch.no_grad():
    pase.eval()

    # y size will be (1, 256, 625), which are 625 frames of 256 dims each
    y = pase(x)[0].transpose(0, 1)

torch.save(y.detach().cpu(), out_path)
