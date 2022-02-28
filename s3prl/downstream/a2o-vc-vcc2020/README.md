# S3PRL-VC: open-source voice conversion framework with self-supervised speech representations

Development: [Wen-Chin Huang](https://github.com/unilight) @ Nagoya University (2021).  
If you have any questions, please open an issue, or contact through email: wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp

**Note**: This is the **any-to-one** recipe. For the **any-to-any** recipe, please go to the [a2a-vc-vctk](../a2a-vc-vctk/) recipe.

We have a [preprint paper](https://arxiv.org/abs/2110.06280) describing this toolkit. If you find this recipe useful, please consider citing:
```
@inproceedings{huang2021s3prl,
  title={S3PRL-VC: Open-source Voice Conversion Framework with Self-supervised Speech Representations},
  author={Huang, Wen-Chin and Yang, Shu-Wen and Hayashi, Tomoki and Lee, Hung-Yi and Watanabe, Shinji and Toda, Tomoki},
  booktitle={Proc. ICASSP},
  year={2022}
}
```

## Table of contents
- [Task](#task)
- [Implementation](#implementation)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Preparation](#preparation)
  - [Dry run / benchmarking an upstream](#dryrun)
  - [Advanced usage](#advanced)
  - (**NEW!!**) [Custom decoding](#custom)


## <a name="task"></a> Task

In this downstream, we focus on training any-to-one (A2O) voice conversion (VC) models on the two tasks in **voice conversion challenge 2020 (VCC2020)**
The first task is _intra-lingual VC_, and the second task is _cross-lingual VC_.
For more details about the two tasks and the VCC2020 dataset, please refer to the original paper:

- Yi, Z., Huang, W., Tian, X., Yamagishi, J., Das, R.K., Kinnunen, T., Ling, Z., Toda, T. (2020) Voice Conversion Challenge 2020 –- Intra-lingual semi-parallel and cross-lingual voice conversion –-. Proc. Joint Workshop for the Blizzard Challenge and Voice Conversion Challenge 2020, 80-98, DOI: 10.21437/VCC_BC.2020-14. [[paper](https://www.isca-speech.org/archive_v0/VCC_BC_2020/pdfs/VCC2020_paper_13.pdf)] [[database](https://github.com/nii-yamagishilab/VCC2020-database)]


## <a name="implementation"></a> Implementation

We implement three models: the **simple** model, **simple-AR** model and **Taco2-AR** model. The simple model and the Taco2-AR model resemble the top systems in VCC2018 and VCC2020, respectively. They are described in the following papers:
- Liu, L., Ling, Z., Jiang, Y., Zhou, M., Dai, L. (2018) WaveNet Vocoder with Limited Training Data for Voice Conversion. Proc. Interspeech 2018, 1983-1987, DOI: 10.21437/Interspeech.2018-1190. [[paper](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1190.pdf)]
- Liu, L., Chen, Y., Zhang, J., Jiang, Y., Hu, Y., Ling, Z., Dai, L. (2020) Non-Parallel Voice Conversion with Autoregressive Conversion Model and Duration Adjustment. Proc. Joint Workshop for the Blizzard Challenge and Voice Conversion Challenge 2020, 126-130, DOI: 10.21437/VCC_BC.2020-17. [[paper](https://www.isca-speech.org/archive_v0/VCC_BC_2020/pdfs/VCC2020_paper_17.pdf)]

We made several modifications.
1. **Input**: instead of using the phonetic posteriorgrams (PPGs) / bottleneck features (BNFs) of a pretrained ASR model, we use the various upstreams provided in S3PRL.
2. **Output**: instead of using acoustic features extracted using a high-quality vocoder, STRAIGHT, we use the log-melspectrograms.
3. **Data**: we benchmark on the [VCC2020](https://github.com/nii-yamagishilab/VCC2020-database) dataset. 
4. **Training strategy**: instead of pretraining on a multispeaker dataset first, we directly trained on the target speaker training set.
5. **Vocoder**: instead of using the WaveNet vocoder, we offer non-AR neural vocoders including [Parallel WaveGAN](https://arxiv.org/abs/1910.11480) (PWG) and [Hifi-GAN](https://arxiv.org/abs/2010.05646), implemented in the [open source project](https://github.com/kan-bayashi/ParallelWaveGAN) developed by [kan-bayashi](https://github.com/kan-bayashi).

## <a name="dependencies"></a> Dependencies:

- `parallel-wavegan`
- `fastdtw`
- `pyworld`
- `pysptk`
- `jiwer`
- `resemblyzer`

You can install them via the `requirements.txt` file.

## <a name="usage"></a> Usage

### <a name="preparation"></a> Preparation
```
# Download the VCC2020 dataset.
cd <root-to-s3prl>/s3prl/downstream/a2o-vc-vcc2020
cd data
./data_download.sh vcc2020/
cd ../

# Download the pretrained vocoders.
./vocoder_download.sh ./
```

### <a name="dryrun"></a> Dry run / benchmarking an upstream
#### Training 
The following command starts a dry run (testing run) given any `<upstream>`.
```
cd <root-to-s3prl>/s3prl
python run_downstream.py -m train -n test -u <upstream> -d a2o-vc-vcc2020
```
By default, the `config.yaml` is used, which has the exact same configuration as `config_taco2_ar.yaml`. Also, the default is to convert to target speaker `TEF1`. We can change the target speaker, the upstream, as well as the exp name to our desired setting. For example, we can train a model that converts to `TEF2` using the `wav2vec` upstream using the following command:
```
python run_downstream.py -m train -n a2o_vc_vcc2020_taco2_ar_TEF2_wav2vec -u wav2vec -d a2o-vc-vcc2020 -o "config.downstream_expert.trgspk='TEF2'"
```
Along the training process, you may find converted speech samples generated using the Griffin-Lim algorithm automatically saved in `<root-to-s3prl>/s3prl/result/downstream/a2o_vc_vcc2020_taco2_ar_TEF2_wav2vec/<step>/test/wav/`.

#### Waveform synthesis (decoding) using a neural vocoder & objective evaluation
We provide a shell script to conveniently perform the followings: (1) waveform synthesis (or _decoding_) using not Griffin-Lim but a neural vocoder, and (2) objective evaluation of a model trained with a specific number of steps. **Note that decoding is done in the `s3prl` directory!**
```
cd <root-to-s3prl>/s3prl/
.downstream/a2o-vc-vcc2020/decode.sh <vocoder_dir> result/downstream/<expname>/<step> <trgspk>
```
For example:
```
./downstream/a2o-vc-vcc2020/decode.sh downstream/a2o-vc-vcc2020/hifigan_vctk result/downstream/a2o_vc_vcc2020_taco2_ar_TEF1_wav2vec/10000 TEF1
```
The generated speech samples will be saved in `<root-to-s3prl>/s3prl/result/downstream/a2o_vc_vcc2020_taco2_ar_<trgspk>_<upstream>/<step>/test/<vocoder_name>_wav/`. 
Also, the output of the evaluation will be shown directly:
```
Mean MCD, f0RMSE, f0CORR, DDUR, CER: 7.79 39.02 0.422 0.356 7.0 15.4
```
And detailed utterance-wise evaluation results can be found in `<root-to-s3prl>/s3prl/result/downstream/a2o_vc_vcc2020_taco2_ar_<trgspk>_<upstream>/<step>/test/<vocoder_name>_wav/obj.log`.

### <a name="advanced"></a> Advanced usage
This section describes advanced usage, targeted at potential VC researchers that wants to evaluate the VC performance using different models in a more efficient way.

#### Batch training
If your GPU memory is sufficient, we can train multiple models in one GPU to avoid executing repeated commands. 
We can also specify a different config file.
In the following command, we train multiple models. **Note that this is done in the `s3prl` directory!**
```
cd <root-to-s3prl>/s3prl
./downstream/a2o-vc-vcc2020/batch_vc_train.sh <upstream> <config_file> <tag> <part>
```
For example, if we want to use the `hubert` upstream with the `config_simple.yaml` configuration to train 4 models w.r.t. the 4 target speakers in VCC2020 task 1:
```
./downstream/a2o-vc-vcc2020/batch_vc_train.sh hubert downstream/a2o-vc-vcc2020/config_simple.yaml simple task1_all
```
Notes:
- In batch training mode, the training log are not output to stdout, but redirected to `<root-to-s3prl>/s3prl/result/downstream/a2o_vc_vcc2020_simple_<trgspk>_hubert/train.log`.
- All exp names will have the format: `a2o_vc_vcc2020_<tag>_<trgspk>_<upstream>`. This can be useful to distinguish different exps if you change the configs.
- We can change `<part>` to specify which target speakers to train. For example, passing `fin` to the script starts two training processes for the two Finnish target speakers. If the GPU memory is insufficient, we can also specify different parts. Please refer to `batch_vc_train.sh` for different specifications.

#### Batch decoding & objective evaluation
After you train models for all target speakers for each task (which can be done by batch training), we can use batch decoding to evaluate all models at once.
```
./downstream/a2o-vc-vcc2020/batch_vc_decode.sh <upstream> <task> <tag> <vocoder_dir>
```
Using the example above, we can run:
```
./downstream/a2o-vc-vcc2020/batch_vc_decode.sh hubert task1 simple downstream/a2o-vc-vcc2020/hifigan_vctk
```
The best result will then be automatically shown.

### <a name="custom"></a> Custom decoding

Since we are training in the A2O setting, the model accepts source speech from arbitrary speakers.

#### Preparation

Prepare a text file, which each line corresponding to an **absolute** path to a source speech file. Here's an example:

```
/mrnas02/internal/wenchin-h/Experiments/s3prl-merge/s3prl/downstream/a2a-vc-vctk/data/wenchin_recording/wenchin_001.wav
/mrnas02/internal/wenchin-h/Experiments/s3prl-merge/s3prl/downstream/a2a-vc-vctk/data/wenchin_recording/wenchin_002.wav
/mrnas02/internal/wenchin-h/Experiments/s3prl-merge/s3prl/downstream/a2a-vc-vctk/data/wenchin_recording/wenchin_003.wav
/mrnas02/internal/wenchin-h/Experiments/s3prl-merge/s3prl/downstream/a2a-vc-vctk/data/wenchin_recording/wenchin_004.wav
/mrnas02/internal/wenchin-h/Experiments/s3prl-merge/s3prl/downstream/a2a-vc-vctk/data/wenchin_recording/wenchin_005.wav
```

#### Decoding

After model training finishes (by following either the [Dry run]($dryrun) or the [Advanced usage](#advanced) sections), use the following script to perform custom decoding:

```
cd <root-to-s3prl>/s3prl
./downstream/a2o-vc-vcc2020/custom_decode.sh <upstream> <trgspk> <tag> <ep> <vocoder_dir> <list_path>
```

For example:

```
./downstream/a2o-vc-vcc2020/custom_decode.sh vq_wav2vec TEF1 ar_taco2 10000 downstream/a2o-vc-vcc2020/hifigan_vctk downstream/a2o-vc-vcc2020/data/lists/custom_eval.yaml
```

After the decoding process ends, you should be able to find the generated files in `result/downstream/a2o_vc_vcc2020_<tag>_<trgspk>_<upstream>/custom_test/`.