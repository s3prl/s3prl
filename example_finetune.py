import torch
from runner_mockingjay import get_mockingjay_model
from downstream.model import example_classifier
from downstream.solver import get_mockingjay_optimizer

# setup the mockingjay model
example_path = 'result/result_mockingjay/mockingjay_libri_sd1337_MelBase/mockingjay-500000.ckpt'
solver = get_mockingjay_model(from_path=example_path)

# setup your downstream class model
classifier = example_classifier(input_dim=768, hidden_dim=128, class_num=2).cuda()

# construct the Mockingjay optimizer
params = list(solver.mockingjay.named_parameters()) + list(classifier.named_parameters())
optimizer = get_mockingjay_optimizer(params=params, lr=4e-3, warmup_proportion=0.7, training_steps=50000)

# forward
example_inputs = torch.zeros(3, 800, 160) # A batch of spectrograms: (batch_size, seq_len, hidden_size)
reps = solver.forward_fine_tune(spec=example_inputs) # returns: (batch_size, seq_len, hidden_size)
loss = classifier(reps, torch.LongTensor([0, 1, 0]).cuda())

# update
loss.backward()
optimizer.step()

# save
PATH_TO_SAVE_YOUR_MODEL = 'example.ckpt'
states = {'Classifier': classifier.state_dict(), 'Mockingjay': solver.mockingjay.state_dict()}
torch.save(states, PATH_TO_SAVE_YOUR_MODEL)
