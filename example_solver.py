import torch
from runner_mockingjay import get_mockingjay_model

example_path = 'result/result_mockingjay/mockingjay_libri_sd1337_LinearLarge/mockingjay-500000.ckpt'
mockingjay = get_mockingjay_model(from_path=example_path)

# A batch of spectrograms: (batch_size, seq_len, hidden_size)
spec = torch.zeros(3, 800, 160)

# reps.shape: (batch_size, num_hiddem_layers, seq_len, hidden_size)
reps = mockingjay.forward(spec=spec, all_layers=True, tile=True)

# reps.shape: (batch_size, num_hiddem_layers, seq_len // downsample_rate, hidden_size)
reps = mockingjay.forward(spec=spec, all_layers=True, tile=False)

# reps.shape: (batch_size, seq_len, hidden_size)
reps = mockingjay.forward(spec=spec, all_layers=False, tile=True)

# reps.shape: (batch_size, seq_len // downsample_rate, hidden_size)
reps = mockingjay.forward(spec=spec, all_layers=False, tile=False)
