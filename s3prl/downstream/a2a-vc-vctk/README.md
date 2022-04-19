# S3PRL-VC: open-source voice conversion framework with self-supervised speech representations

Development: [Wen-Chin Huang](https://github.com/unilight) @ Nagoya University (2021).  
If you have any questions, please open an issue, or contact through email: wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp

**Note**: This is the **any-to-any** recipe. For the **any-to-one** recipe, please go to the [a2o-vc-vcc2020](../a2o-vc-vcc2020/) recipe.

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
  - [Training](#training)
  - [Waveform synthesis (decoding) using a neural vocoder & objective evaluation](#decoding)
  - (**NEW!!**) [Custom decoding](#custom)


## <a name="task"></a> Task

In this downstream, we focus on training any-to-any (A2A) voice conversion (VC) models.
We perform model training on the **VCTK** corpus, and evaluate on the _intra-lingual VC_ task in **voice conversion challenge 2020 (VCC2020)**.
For more details about the intra-lingual VC task and the VCC2020 dataset, please refer to the original paper:

- Yi, Z., Huang, W., Tian, X., Yamagishi, J., Das, R.K., Kinnunen, T., Ling, Z., Toda, T. (2020) Voice Conversion Challenge 2020 –- Intra-lingual semi-parallel and cross-lingual voice conversion –-. Proc. Joint Workshop for the Blizzard Challenge and Voice Conversion Challenge 2020, 80-98, DOI: 10.21437/VCC_BC.2020-14. [[paper](https://www.isca-speech.org/archive_v0/VCC_BC_2020/pdfs/VCC2020_paper_13.pdf)] [[database](https://github.com/nii-yamagishilab/VCC2020-database)]


## <a name="implementation"></a> Implementation

We only provide the config for the **Taco2-AR** model. It is essentialy a modified, attention-free Tacotron2 model. For the speaker embedding, we used a [d-vector](https://static.googleusercontent.com/media/research.google.com/zh-TW//pubs/archive/41939.pdf) implementation by the [Resemblyzer](https://github.com/resemble-ai/Resemblyzer). For the vocoder, we only offer the [Hifi-GAN](https://arxiv.org/abs/2010.05646), implemented in the [open source project](https://github.com/kan-bayashi/ParallelWaveGAN) developed by [kan-bayashi](https://github.com/kan-bayashi).

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
# Download the VCTK and the VCC2020 datasets.
cd <root-to-s3prl>/s3prl/downstream/a2a-vc-vctk
cd data
./vcc2020_download.sh vcc2020/
./vctk_download.sh ./
cd ../

# Download the pretrained vocoders.
./vocoder_download.sh ./
```

### <a name="training"></a> Training 
The following command starts a training run given any `<upstream>`.
```
cd <root-to-s3prl>/s3prl
./downstream/a2a-vc-vctk/vc_train.sh <upstream> downstream/a2a-vc-vctk/config_ar_taco2.yaml <tag>
```
Along the training process, you may find converted speech samples generated using the Griffin-Lim algorithm automatically saved in `<root-to-s3prl>/s3prl/result/downstream/a2a_vc_vctk_<tag>_<upstream>/<step>/test/wav/`.
**NOTE**: to avoid extracting d-vectors on-the-fly (which is very slow), all d-vectors are extracted beforehand and saved in `data/spk_embs`. Since there are 44 hours of data in VCTK, the whole extraction can take a long time. On a NVIDIA GeForce RTX 3090, it takes 5-6 hours.
**NOTE 2**: By default, during testing, the d-vector of the target speaker is the average of random samples from the training set, of number `num_ref_samples`. You can change this number in the config file. The list of samples is generated automatically and saved in `data/eval_<num>sample_list.txt`.

### <a name="decoding"></a> Waveform synthesis (decoding) using a neural vocoder & objective evaluation

#### Single model checkpoint decoding & evaluation
```
cd <root-to-s3prl>/s3prl
./downstream/a2a-vc-vctk/decode.sh <vocoder> <result_dir>/<step>
```
For example,
```
./downstream/a2a-vc-vctk/decode.sh ./downstream/a2a-vc-vctk/hifigan_vctk result/downstream/a2a_vc_vctk_taco2_ar_decoar2/50000
```

#### Upstream-wise decoding & evaluation
The following command performs objective evaluation of a model trained with a specific number of steps.
```
cd <root-to-s3prl>/s3prl
./downstream/a2a-vc-vctk/batch_vc_decode.sh <upstream> taco2_ar downstream/a2a-vc-vctk/hifigan_vctk
```
If the command fails, please make sure there are trained results in `result/downstream/a2a_vc_vctk_<tag>_<upstream>/`. The generated speech samples will be saved in `<root-to-s3prl>/s3prl/result/downstream/a2a_vc_vctk_taco2_ar_<upstream>/<step>/hifigan_wav/`. 

Also, the output of the evaluation will be shown directly:
```
decoar2 10 samples epoch 48000 best: 9.28 41.80 0.197 1.3 4.0 27.00
```
And detailed utterance-wise evaluation results can be found in `<root-to-s3prl>/s3prl/result/downstream/a2a_vc_vctk_taco2_ar_<upstream>/<step>/hifigan_wav/obj_10samples.log`.

### <a name="custom"></a> Custom decoding

Since we are training in the A2O setting, the model accepts source speech from arbitrary speakers.

#### Preparation

Prepare a text file with the form of yaml. We follow the format in [S2VC](https://github.com/howard1337/S2VC#voice-coanversion-with-pretrained-models). Please see the following example:

```
TEF1_wenchin_001: # this will be the name of the generated file.
    src_spk_name: "wenchin" # This field is needed to match the interface. It is not really used.
    ref_spk_name: "TEF1" # This field is needed to match the interface. It is not really used.
    src: /mrnas02/internal/wenchin-h/Experiments/s3prl-merge/s3prl/downstream/a2a-vc-vctk/data/wenchin_recording/wenchin_001.wav # Note that absolute path is needed.
    ref: # An average of d-vectors extracted with these speech files will be used. Note that absolute path is needed.
        - /mrnas02/internal/wenchin-h/Experiments/s3prl-merge/s3prl/downstream/a2o-vc-vcc2020/data/vcc2020/TEF1/E20008.wav
        - /mrnas02/internal/wenchin-h/Experiments/s3prl-merge/s3prl/downstream/a2o-vc-vcc2020/data/vcc2020/TEF1/E20024.wav
        - /mrnas02/internal/wenchin-h/Experiments/s3prl-merge/s3prl/downstream/a2o-vc-vcc2020/data/vcc2020/TEF1/E20011.wav
```

#### Decoding

After model training finishes (by followingthe [Training](#training) section), use the following script to perform custom decoding:

```
cd <root-to-s3prl>/s3prl
./downstream/a2a-vc-vctk/custom_decode.sh <tag> <upstream> <ep> <vocoder_dir> <list_path>
```

For example:

```
./downstream/a2a-vc-vctk/custom_decode.sh ar_taco2 vq_wav2vec 50000 downstream/a2a-vc-vctk/hifigan_vctk downstream/a2a-vc-vctk/data/lists/custom_eval_wenchin.yaml
```

After the decoding process ends, you should be able to find the generated files in `result/downstream/a2a_vc_vctk_<tag>_<upstream>/custom_test/`.