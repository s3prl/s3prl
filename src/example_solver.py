import torch
from utility.helper import get_transformer_tester

example_path = './result/result_transformer/tera/fmllrBase960-F-N-K-libri/states-1000000.ckpt'
tester= get_transformer_tester(from_path=example_path)

# A batch of spectrograms: (batch_size, seq_len, feature_size)
spec = torch.zeros(3, 800, 40)

# reps.shape: (batch_size, num_hiddem_layers, seq_len, hidden_size)
reps = tester.forward(spec=spec, all_layers=True, tile=True)

# reps.shape: (batch_size, num_hiddem_layers, seq_len // downsample_rate, hidden_size)
reps = tester.forward(spec=spec, all_layers=True, tile=False)

# reps.shape: (batch_size, seq_len, hidden_size)
reps = tester.forward(spec=spec, all_layers=False, tile=True)

# reps.shape: (batch_size, seq_len // downsample_rate, hidden_size)
reps = tester.forward(spec=spec, all_layers=False, tile=False)
