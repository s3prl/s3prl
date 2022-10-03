This is the second version of the source separation benchmark. It uses the same Libri2Mix dataset as separation_stft. We made the following changes to improve the training speed and system performance.

1. Adjust the learning rate from 1e-4 to 1e-3.
2. Adjust n_fft and window_size of STFT from 512 to 1024. For separation task, it seems a larger window size gives better performance.
3. Reduce the model size from 896 to 256. Following the principle of SUPERB, we should use a smaller model size. The training speed is greatly improved due to this change.
4. Reduce the dropout rate from 0.5 to 0.1.
5. Change the mask activation from relu to sigmoid.

Among the modifications, the 2nd one leads to the most improvement. The 3rd one results in much smaller model size and improved training speed.

![sep](https://user-images.githubusercontent.com/35029997/187686425-7e5e3a01-51d7-430d-95ff-e779fc2b8aaf.png)

The above figure shows the SI-SDRi comparison between the original (separation_stft, represented with the red line) and new version (separation_stft2, represented with the blue line).
