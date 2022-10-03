This is the second version of the speech enhancement benchmark. It uses the same Voicebank-DEMAND dataset as enhancement_stft. We made the following changes to improve the training speed and system performance.

1. Reduce total training steps from 150000 to 100000.
2. Adjust the learning rate from 1e-4 to 1e-3.
3. Adjust the win_length of STFT from 512 to 400. This is a small change that doesn't change the result. 400 sample size (25ms) is more commonly used for ASR.
4. Reduce the model size from 896 to 256. We found that the influence of model size on PESQ is very small.
5. Reduce the dropout rate from 0.5 to 0.1.
6. Change the loss type from L2 to L1.
7. Change the mask activation from relu to sigmoid and the mask type from PSM to AM. The combination of the mask type and activation function is the main reason for the improvement. It seems that the phase sensitive mask (PSM) is not suitable for the enhancement task.
8. For upstream with stride size 320 (e.g. wav2vec2, HuBERT), we are repeating the features so that it has a stride size of 160. Now the enhancement downstream only supports stride sizes of 160 and 320.

![enh](https://user-images.githubusercontent.com/35029997/187686370-97a8cfd5-21db-4033-bc67-d2f888e90bbf.png)

The above figure shows the PESQ comparison between the original (enhancement_stft, represented with the red line) and new version (enhancement_stft2, represented with the blue line). The large performance gain mainly comes from the mask type and activation function.
