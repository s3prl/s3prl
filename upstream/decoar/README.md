Decoar 1.0 uses **MXNet**, we wish to convert the network into **PyTorch** to enable finetuning the pretrained model in a few months, while we first stick to the original implementation to ensure the fidelity on representation quality.

1. Modify the **MXNet** version use wish to use, here I use `mxnet-cu102mkl` for GPU with CUDA 10.2

2. `pip install -r requirements.txt`

**Note.** Decoar 2.0 is not yet released by *https://github.com/awslabs/speech-representations*
