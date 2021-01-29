## ASR with s3prl upstream

Since ASR is a complicated system including LM training and beam search decoding during testing. Instead of reinventing the wheel on this repo, we decided to incorporate s3prl pretrained models into the framework of an existing well-organized ASR repo [Alexander-H-Liu/End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch), which is created by a senior member from our lab. We created a fork from that repo and make it support finetuning pretrained models registered in s3prl: [s3prl/End-to-end-ASR-Pytorch](https://github.com/s3prl/End-to-end-ASR-Pytorch) by:

- Delay the feature extraction part from dataloader to training loop, and accelerate it by GPU either for fbank extraction or pretrained feature extraction.
- Support distributed data parallel to enable large batch size finetuning, since some pretrained models can be huge like wav2vec2.

One can refer to the README in [s3prl/End-to-end-ASR-Pytorch](https://github.com/s3prl/End-to-end-ASR-Pytorch) for more details.
