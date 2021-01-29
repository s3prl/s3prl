## ASR with s3prl upstream

Since ASR is a complicated system including LM training and beam search decoding during testing. Instead of reinventing the wheel on this repo, we decided to incorporate s3prl pretrained models into the framework of an existing well-organized ASR repo [Alexander-H-Liu/End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch), which is created by a senior member from our lab. We created a fork from that repo: [s3prl/End-to-end-ASR-Pytorch](https://github.com/s3prl/End-to-end-ASR-Pytorch) and make it support finetuning pretrained models registered in s3prl by the following procedure:

1. The dataloader prepares batches of waveforms.
2. The pretrained upstream model extracts features from the input waveforms.
3. The downstream ASR model uses pretrained features as input to train CTC or LAS loss (or hybrid).

You can choose to treat pretrained models as representation extrator and fixed the pretrained models by not setting `--upstream_trainable`. In this case, only the downstream ASR model is trained using the pretrained representation. Or, you can choose to finetune the whole pretrained models and downstream ASR model by setting `--upstream_trainable`. In this case typically you only wish to use a very small downstream ASR model like 1 linear layer, and this can be achieved by properly setting the config file.

We further add the support for **distributed data parallel** to enable large batch size training, since some pretrained models can be huge like wav2vec2. One can refer to the README in [s3prl/End-to-end-ASR-Pytorch](https://github.com/s3prl/End-to-end-ASR-Pytorch) for more details.
