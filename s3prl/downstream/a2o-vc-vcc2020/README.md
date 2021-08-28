# S3PRL-VC: open-source voice conversion framework with self-supervised speech representations

Development: [Wen-Chin Huang](https://github.com/unilight) @ Nagoya University (2021).  
If you have any questions, please open an issue, or contact through email: wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp

## Task

In this downstream, we focus on training any-to-one (A2O) voice conversion (VC) models on the two tasks in [voice conversion challenge 2020 (VCC2020)](https://github.com/nii-yamagishilab/VCC2020-database).
The first task is intra-lingual VC, and the second task is cross-lingual VC.
For more details, please refer to the original paper:
[Yi, Z., Huang, W., Tian, X., Yamagishi, J., Das, R.K., Kinnunen, T., Ling, Z., Toda, T. (2020) Voice Conversion Challenge 2020 –- Intra-lingual semi-parallel and cross-lingual voice conversion –-. Proc. Joint Workshop for the Blizzard Challenge and Voice Conversion Challenge 2020, 80-98, DOI: 10.21437/VCC_BC.2020-14.](https://www.isca-speech.org/archive_v0/VCC_BC_2020/pdfs/VCC2020_paper_13.pdf)

## Implementation

We implement three models. Two of them resemble the the top systems in VCC2018 and VCC2020, as described in the following papers:
[Liu, L., Ling, Z., Jiang, Y., Zhou, M., Dai, L. (2018) WaveNet Vocoder with Limited Training Data for Voice Conversion. Proc. Interspeech 2018, 1983-1987, DOI: 10.21437/Interspeech.2018-1190.](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1190.pdf)  
[Liu, L., Chen, Y., Zhang, J., Jiang, Y., Hu, Y., Ling, Z., Dai, L. (2020) Non-Parallel Voice Conversion with Autoregressive Conversion Model and Duration Adjustment. Proc. Joint Workshop for the Blizzard Challenge and Voice Conversion Challenge 2020, 126-130, DOI: 10.21437/VCC_BC.2020-17.](https://www.isca-speech.org/archive_v0/VCC_BC_2020/pdfs/VCC2020_paper_17.pdf)

We made several modifications.
1. **Input**: instead of using the bottleneck features (BNFs) of a pretrained ASR model, we use the various upstreams provided in S3PRL.
2. **Output**: instead of using acoustic features extracted using a high-quality vocoder, STRAIGHT, we use the log-melspectrograms.
3. **Data**: we benchmark on the [VCC2020](https://github.com/nii-yamagishilab/VCC2020-database) dataset. 
4. **Training strategy**: instead of pretraining on a multispeaker dataset first, we directly trained on the target speaker training set.
5. **Vocoder**: instead of using the WaveNet vocoder, we used the [Parallel WaveGAN](https://arxiv.org/abs/1910.11480) (PWG) based on the [open source project](https://github.com/kan-bayashi/ParallelWaveGAN) developed by [kan-bayashi](https://github.com/kan-bayashi).

## Dependencies:

- `parallel-wavegan`
- `fastdtw`
- `pyworld`
- `pysptk`
- `jiwer`
- `resemblyzer`

## Usage

### Preparation
```
# Download the VCC2020 dataset.
cd <root-to-s3prl>/s3prl/downstream/a2o-vc-vcc2020
cd data
./data_download.sh vcc2020/
cd ../

# Download the pretrained PWGs.
./vocoder_download.sh ./
```

### Training
Here we train an A2O VC model using the `config_ar_taco2.yaml` configuration. Please replace the `<trgspk>`, `<upstream>` in the following command to your desired settings.
```
cd <root-to-s3prl>/s3prl
python run_downstream.py -m train --config downstream/a2o-vc-vcc2020/config_ar_taco2.yaml -n a2o_vc_vcc2020_<trgspk>_<upstream> -u <upstream> -d a2o-vc-vcc2020 -o "config.downstream_expert.trgspk='<trgspk>'" 
```
For example:
```
python run_downstream.py -m train --config downstream/a2o-vc-vcc2020/config_ar_taco2.yaml -n a2o_vc_vcc2020_ar_taco2_TEF1_wav2vec -u wav2vec -d a2o-vc-vcc2020 -o "config.downstream_expert.trgspk='TEF1'"
```
Note that `-n a2o_vc_vcc2020_ar_taco2_TEF1_wav2vec` can be changed freely.
Along training, the converted samples generated using the Griffin-Lim algorithm will be saved in `<root-to-s3prl>/s3prl/result/downstream/a2o_vc_vcc2020_ar_taco2_<trgspk>_<upstream>/<step>/test/wav/`.

### Decoding using a neural vocoder & objective evaluation
Decoding using not Griffin-Lim but a neural vocoder, as well as the objective evaluation of a model trained with a specific number of steps can be performed by the following command:
```
cd <root-to-s3prl>/s3prl/downstream/a2o-vc-vcc2020
./decode.sh <vocoder_dir> <root-to-s3prl>/s3prl/result/downstream/a2o_vc_vcc2020_ar_taco2_<trgspk>_<upstream>/<step> <trgspk>
```
For example:
```
./decode.sh pwg_task1 ../../result/downstream/a2o_vc_vcc2020_ar_taco2_TEF1_wav2vec/10000 TEF1
```
The generated samples using PWG will be saved in `<root-to-s3prl>/s3prl/result/downstream/a2o_vc_vcc2020_ar_taco2_<trgspk>_<upstream>/<step>/test/<vocoder_name>_wav/`.  
The output of the evaluation should look like:
```
Mean MCD, f0RMSE, f0CORR, DDUR, CER: 7.79 39.02 0.422 0.356 7.0 15.4
```
Utterance-wise evaluation results can be found in `<root-to-s3prl>/s3prl/result/downstream/a2o_vc_vcc2020_ar_taco2_<trgspk>_<upstream>/<step>/test/<vocoder_name>_wav/obj.log`.

### Batch training
If your GPU memory is sufficient, we can train multiple models in one GPU to avoid executing repeated commands.
Here we train 4 models w.r.t. the 4 target speakers in VCC2020 task 1, again using the `config_ar_taco2.yaml` configuration.
```
cd <root-to-s3prl>/s3prl/downstream/a2o-vc-vcc2020
./batch_vc_train.sh <upstream> config_ar_taco2.yaml ar_taco2 task1_all
```
Note that in this batch training mode, the training log are not output to stdout, but redirected to `<root-to-s3prl>/s3prl/result/downstream/a2o_vc_vcc2020_ar_taco2_<trgspk>_<upstream>/train.log`.
If the GPU memory is insufficient, you can also specify different parts. Please refer to `batch_vc_train.sh` for different specifications.

### Batch decoding & objective evaluation
After you train models for all target speakers for each task (which can be done by either batch training or one-by-one), we can use batch decoding to evaluate all models at once.
```
./batch_vc_decode.sh <upstream> <task> ar_taco2
```
The best result will be automatically shown.
