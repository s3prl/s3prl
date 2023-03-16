<p align="center">
    <img src="https://raw.githubusercontent.com/s3prl/s3prl/main/file/S3PRL-logo.png" width="900"/>
    <br>
    <br>
    <a href="./LICENSE.txt"><img alt="Apache License 2.0" src="https://raw.githubusercontent.com/s3prl/s3prl/main/file/license.svg" /></a>
    <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img alt="CC_BY_NC License" src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg" /></a>
    <a href="https://github.com/s3prl/s3prl/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/s3prl/s3prl/actions/workflows/ci.yml/badge.svg?branch=main&event=push"></a>
    <a href="#development-pattern-for-contributors"><img alt="Codecov" src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg"></a>
    <a href="https://github.com/s3prl/s3prl/issues"><img alt="Bitbucket open issues" src="https://img.shields.io/github/issues/s3prl/s3prl"></a>
</p>

## Notice for pull requests

Please first discuss with us on the issue page about your feature request before implementing the actual pull request, so we can discuss about how to achieve the functionality. If we did not discuss about the detail, it is highly possible that we are not accepting the pull request due to the difficulty of maintenance.

## Environment compatibilities [![CI](https://github.com/s3prl/s3prl/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/s3prl/s3prl/actions/workflows/ci.yml)

We support the following environments. The test cases are ran with **[tox](./tox.ini)** locally and on **[github action](.github/workflows/ci.yml)**:

| Env | versions |
| --- | --- |
| os  | `ubuntu-18.04`, `ubuntu-20.04` |
| python | `3.7`, `3.8`, `3.9`, `3.10` |
| pytorch | `1.8.1`, `1.9.1`, `1.10.2`, `1.11.0`, `1.12.1` |

## Change Log

**Please note that**

As the core developers and repository owners, [Andy T. Liu](https://github.com/andi611) and [Shu-wen (Leo) Yang](https://github.com/leo19941227) **never** forced others to contribute to the S3PRL project. Instead, most of the times **people want and ask** to contribute to this proejct. Feel free to ask old contributors to check. We built this project from scratch and developed most of the functions ourselves **since 2019** and received huge amount of stars. There were around 400 stars before we started to develop SUPERB Benchmark. (Not to mention that we also recruited most of the first generation SUPERB team members at National Taiwan University ourselves and led the development of SUPERB and rebuilt the entire downstream evaluation codebase. **No one ever paid us** for building this project which we started ourselves in the hope to help others and we did. There are several papers using our upstream collection function citing the S3PRL github link before SUPERB's existence. For example, the S2VC paper in the `Used by` section below.) Hence people were interested to contribute to our project and develop new functions together. From our perspective, **we were kind enough** to help reviewing PRs and help others build their resume. (We could just refuse them in the beginning.) But later, someone started to question about our ownership on this repository and manipulate the information, just because we allowed others to contribute. This made us **really disappointed**. We hence thought about never allowing new contributions in the future. If any old contributors felt unhappy about the current situation, it is okay to dicuss about removing your previous contributions, and we are happy to help. Our principle is always to give the biggest credit to those who already contributed (especially who fought with us when no one knew S3PRL,) instead of someone wishing to take over the repository by undiscussed and overwhelming PRs after the repository is successful. We still welcome new contributions as long as the development is fully discussed and the new contributors accept our principle above. We are grateful and care about every precious old contributions and refuse to dilute old contributors' credit by accepting lots of new PRs. Please find the names below to know the people who put huge effort on making this project successful.

As a worth-mentioning point, we are not accepting new SUPERB development in the future. There are two major reasons:

- The SUPERB team members changed a lot and rapidly. Andy and Leo are not available to involve every new effort, hence we are afraid to accept undiscussed and unreviewed while huge amount of new implementations. We won't get benefit from helping reviewing these efforts, while people will further question about our ownership and try to take over it, and the new members might not always want to contribute to our project. Hence, there is no benefit to both sides.
- The current S3PRL is getting too large and becoming hard to maintain and keep the dependency small.

Hence, we will do our best in maintaining SUPERB and SUPERB-SG, and we believe it is better to let the new SUPERB members to maintain their own implementation in their own repositories, just like how we started from nothing when developing the first version of SUPERB.
So both sides get proper credit to their own development.

* *Mar 2022*: Introduce [**SUPERB-SG**](https://arxiv.org/abs/2203.06849), see [Speech Translation](./s3prl/downstream/speech_translation) by [Hsiang-Sheng Tsai](https://github.com/bearhsiang), [Out-of-domain ASR](./s3prl/downstream/ctc/) by [Heng-Jui Chang](https://github.com/vectominist), [Voice Conversion](./s3prl/downstream/a2o-vc-vcc2020/) by [Wen-Chin Huang](https://github.com/unilight), [Speech Separation](./s3prl/downstream/separation_stft/) and [Speech Enhancement](./s3prl/downstream/enhancement_stft/) by [Zili Huang](https://github.com/HuangZiliAndy) for more info.
* *Nov 2021*: Introduce [**S3PRL-VC**](https://arxiv.org/abs/2110.06280) by [Wen-Chin Huang](https://github.com/unilight), see [Any-to-one](https://github.com/s3prl/s3prl/tree/master/s3prl/downstream/a2o-vc-vcc2020) for more info. We highly recommend to consider the [newly released official repo of S3PRL-VC](https://github.com/unilight/s3prl-vc) which is developed and actively maintained by [Wen-Chin Huang](https://github.com/unilight). The standalone repo contains much more recepies for the VC experiments. In S3PRL we only include the Any-to-one recipe for reproducing the SUPERB results.
* *Oct 2021*: Support [**DistilHuBERT**](https://arxiv.org/abs/2110.01900) by [Heng-Jui Chang](https://github.com/vectominist), see [docs](./s3prl/upstream/distiller/README.md) for more info
* *Sep 2021:* We host a *challenge* in [*AAAI workshop: The 2nd Self-supervised Learning for Audio and Speech Processing*](https://aaai-sas-2022.github.io/)! See [**SUPERB official site**](https://superbbenchmark.org/) for the challenge details and the [**SUPERB documentation**](./s3prl/downstream/docs/superb.md) in this toolkit!
* *Aug 2021:* We now have a tutorial that introduces our toolkit, you can **[watch it on Youtube](https://youtu.be/PkMFnS6cjAc)**!
* *July 2021:* We are now working on packaging s3prl and reorganizing the file structure in **v0.3**. Please consider using the stable **v0.2.0** for now. We will test and release **v0.3** before August.
* *June 2021:* Support [**SUPERB:** **S**peech processing **U**niversal **PER**formance **B**enchmark](https://arxiv.org/abs/2105.01051), submitted to Interspeech 2021. Use the tag **superb-interspeech2021** or **v0.2.0**. The individual task contributors include: [Shu-wen (Leo) Yang](https://github.com/leo19941227), [Pohan Chi](https://github.com/pohanchi), [Yist Lin](https://github.com/yistLin), [Yung-Sung Chuang](https://github.com/voidism), [Jiatong Shi](https://github.com/ftshijt), [Xuankai](https://github.com/simpleoier), [Wei-Cheng Tseng](https://github.com/RayTzeng) and Tzu-Hsien Huang.
* *June 2021:* Support extracting multiple hidden states from the SSL pretrained models
* *Jan 2021:* Readme updated with detailed instructions on how to use our latest version!
* *Dec 2020:* We are migrating to a newer version for a more general, flexible, and scalable code. See the introduction below for more information! The legacy version can be accessed the tag **v0.1.0**.
* *Oct 2020:* We added lots of classic upstream models, including PASE+, APC, VQ-APC, NPC, wav2vec, vq-wav2vec ...etc.
* *Oct 2019:* [Andy T. Liu](https://github.com/andi611), [Shu-wen (Leo) Yang](https://github.com/leo19941227) and [Pohan Chi](https://github.com/pohanchi) implemented the Mockingjay pre-training and several simple downstream evaluation tasks. This work was the very start of the S3PRL project which established lots of foundamental modules and coding styles. Feel free to checkout to the oldest commits to explore our legacy codebase!

****

## Introduction and Usages

This is an open source toolkit called **s3prl**, which stands for **S**elf-**S**upervised **S**peech **P**re-training and **R**epresentation **L**earning.
Self-supervised speech pre-trained models are called **upstream** in this toolkit, and are utilized in various **downstream** tasks.

The toolkit has **three major usages**:

### Pretrain

- Pretrain upstream models, including Mockingjay, Audio ALBERT and TERA.
- Document: [**pretrain/README.md**](./s3prl/pretrain/README.md)

### Upstream

- Easily load most of the existing upstream models with pretrained weights in a unified I/O interface.
- Pretrained models are registered through **torch.hub**, which means you can use these models in your own project by one-line plug-and-play without depending on this toolkit's coding style.
- Document: [**upstream/README.md**](./s3prl/upstream/README.md)

### Downstream

- Utilize upstream models in lots of downstream tasks
- Benchmark upstream models with [**SUPERB Benchmark**](./s3prl/downstream/docs/superb.md)
- Document: [**downstream/README.md**](./s3prl/downstream/README.md)

Below is an **intuitive illustration** on how this toolkit may help you:
\
\
<img src="https://raw.githubusercontent.com/s3prl/s3prl/main/file/S3PRL-interface.png" width="900"/>
\
\
Feel free to use or modify our toolkit in your research. Here is a [list of papers using our toolkit](#used-by). Any question, bug report or improvement suggestion is welcome through [opening up a new issue](https://github.com/s3prl/s3prl/issues). 

If you find this toolkit helpful to your research, please do consider citing [our papers](#citation), thanks!

## Installation

1. **Python** >= 3.6
2. Install **sox** on your OS
3. Install s3prl: [Read doc](https://s3prl.github.io/s3prl/tutorial/installation.html#) or `pip install -e ".[all]"`
4. (Optional) Some upstream models require special dependencies. If you encounter error with a specific upstream model, you can look into the `README.md` under each `upstream` folder. E.g., `upstream/pase/README.md`

## Development pattern for contributors

1. [Create a personal fork](https://help.github.com/articles/fork-a-repo/) of the [main S3PRL repository](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning) in GitHub.
2. Make your changes in a named branch different from `master`, e.g. you create a branch `new-awesome-feature`.
3. Contact us if you have any questions during development.
4. [Generate a pull request](https://help.github.com/articles/creating-a-pull-request/) through the Web interface of GitHub.
5. Please verify that your code is free of basic mistakes, we appreciate any contribution!

## Reference Repositories

* [Pytorch](https://github.com/pytorch/pytorch), Pytorch.
* [Audio](https://github.com/pytorch/audio), Pytorch.
* [Kaldi](https://github.com/kaldi-asr/kaldi), Kaldi-ASR.
* [Transformers](https://github.com/huggingface/transformers), Hugging Face.
* [PyTorch-Kaldi](https://github.com/mravanelli/pytorch-kaldi), Mirco Ravanelli.
* [fairseq](https://github.com/pytorch/fairseq), Facebook AI Research.
* [CPC](https://github.com/facebookresearch/CPC_audio), Facebook AI Research.
* [APC](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding), Yu-An Chung.
* [VQ-APC](https://github.com/s3prl/VQ-APC), Yu-An Chung.
* [NPC](https://github.com/Alexander-H-Liu/NPC), Alexander-H-Liu.
* [End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch), Alexander-H-Liu
* [Mockingjay](https://github.com/andi611/Mockingjay-Speech-Representation), Andy T. Liu.
* [ESPnet](https://github.com/espnet/espnet), Shinji Watanabe
* [speech-representations](https://github.com/awslabs/speech-representations), aws lab
* [PASE](https://github.com/santi-pdp/pase), Santiago Pascual and Mirco Ravanelli
* [LibriMix](https://github.com/JorisCos/LibriMix), Joris Cosentino and Manuel Pariente

## License

The majority of S3PRL Toolkit is licensed under the Apache License version 2.0, however all the files authored by Facebook, Inc. (which have explicit copyright statement on the top) are licensed under CC-BY-NC.

## Used by
<details><summary>List of papers that used our toolkit (Feel free to add your own paper by making a pull request)</summary><p>

### Self-Supervised Pretraining

+ [Mockingjay: Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders (Liu et al., 2020)](https://arxiv.org/abs/1910.12638)
  ```
  @article{mockingjay,
     title={Mockingjay: Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders},
     ISBN={9781509066315},
     url={http://dx.doi.org/10.1109/ICASSP40776.2020.9054458},
     DOI={10.1109/icassp40776.2020.9054458},
     journal={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
     publisher={IEEE},
     author={Liu, Andy T. and Yang, Shu-wen and Chi, Po-Han and Hsu, Po-chun and Lee, Hung-yi},
     year={2020},
     month={May}
  }
  ```
+ [TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech (Liu et al., 2020)](https://arxiv.org/abs/2007.06028)
  ```
  @misc{tera,
      title={TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech},
      author={Andy T. Liu and Shang-Wen Li and Hung-yi Lee},
      year={2020},
      eprint={2007.06028},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
  }
  ```
+ [Audio ALBERT: A Lite BERT for Self-supervised Learning of Audio Representation (Chi et al., 2020)](https://arxiv.org/abs/2005.08575)
  ```
  @inproceedings{audio_albert,
      title={Audio ALBERT: A Lite BERT for Self-supervised Learning of Audio Representation},
      author={Po-Han Chi and Pei-Hung Chung and Tsung-Han Wu and Chun-Cheng Hsieh and Shang-Wen Li and Hung-yi Lee},
      year={2020},
      booktitle={SLT 2020},
  }
  ```

### Explanability

+ [Understanding Self-Attention of Self-Supervised Audio Transformers (Yang et al., 2020)](https://arxiv.org/abs/2006.03265)
  ```
  @inproceedings{understanding_sat,
      author={Shu-wen Yang and Andy T. Liu and Hung-yi Lee},
      title={{Understanding Self-Attention of Self-Supervised Audio Transformers}},
      year=2020,
      booktitle={Proc. Interspeech 2020},
      pages={3785--3789},
      doi={10.21437/Interspeech.2020-2231},
      url={http://dx.doi.org/10.21437/Interspeech.2020-2231}
  }
  ```

### Adversarial Attack

+ [Defense for Black-box Attacks on Anti-spoofing Models by Self-Supervised Learning (Wu et al., 2020)](https://arxiv.org/abs/2006.03214), code for computing LNSR: [utility/observe_lnsr.py](https://github.com/s3prl/s3prl/blob/master/utility/observe_lnsr.py)
  ```
  @inproceedings{mockingjay_defense,
      author={Haibin Wu and Andy T. Liu and Hung-yi Lee},
      title={{Defense for Black-Box Attacks on Anti-Spoofing Models by Self-Supervised Learning}},
      year=2020,
      booktitle={Proc. Interspeech 2020},
      pages={3780--3784},
      doi={10.21437/Interspeech.2020-2026},
      url={http://dx.doi.org/10.21437/Interspeech.2020-2026}
  }
  ```

+ [Adversarial Defense for Automatic Speaker Verification by Cascaded Self-Supervised Learning Models (Wu et al., 2021)](https://arxiv.org/abs/2102.07047)
  ```
  @misc{asv_ssl,
      title={Adversarial defense for automatic speaker verification by cascaded self-supervised learning models}, 
      author={Haibin Wu and Xu Li and Andy T. Liu and Zhiyong Wu and Helen Meng and Hung-yi Lee},
      year={2021},
      eprint={2102.07047},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
  ```

### Voice Conversion

+ [S2VC: A Framework for Any-to-Any Voice Conversion with Self-Supervised Pretrained Representations (Lin et al., 2021)](https://arxiv.org/abs/2104.02901)
  ```
  @misc{s2vc,
        title={S2VC: A Framework for Any-to-Any Voice Conversion with Self-Supervised Pretrained Representations}, 
        author={Jheng-hao Lin and Yist Y. Lin and Chung-Ming Chien and Hung-yi Lee},
        year={2021},
        eprint={2104.02901},
        archivePrefix={arXiv},
        primaryClass={eess.AS}
  }
  ```

### Benchmark and Evaluation

+ [SUPERB: Speech processing Universal PERformance Benchmark (Yang et al., 2021)](https://arxiv.org/abs/2105.01051)
  ```
  @misc{superb,
        title={SUPERB: Speech processing Universal PERformance Benchmark}, 
        author={Shu-wen Yang and Po-Han Chi and Yung-Sung Chuang and Cheng-I Jeff Lai and Kushal Lakhotia and Yist Y. Lin and Andy T. Liu and Jiatong Shi and Xuankai Chang and Guan-Ting Lin and Tzu-Hsien Huang and Wei-Cheng Tseng and Ko-tik Lee and Da-Rong Liu and Zili Huang and Shuyan Dong and Shang-Wen Li and Shinji Watanabe and Abdelrahman Mohamed and Hung-yi Lee},
        year={2021},
        eprint={2105.01051},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
  }
  ```

+ [Utilizing Self-supervised Representations for MOS Prediction (Tseng et al., 2021)](https://arxiv.org/abs/2104.03017)
  ```
  @misc{ssr_mos,
      title={Utilizing Self-supervised Representations for MOS Prediction}, 
      author={Wei-Cheng Tseng and Chien-yu Huang and Wei-Tsung Kao and Yist Y. Lin and Hung-yi Lee},
      year={2021},
      eprint={2104.03017},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
  }
  ```
}

</p></details>

## Citation

If you find this toolkit useful, please consider citing following papers.

- If you use our pre-training scripts, or the downstream tasks considered in *TERA* and *Mockingjay*, please consider citing the following:
```
@misc{tera,
  title={TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech},
  author={Andy T. Liu and Shang-Wen Li and Hung-yi Lee},
  year={2020},
  eprint={2007.06028},
  archivePrefix={arXiv},
  primaryClass={eess.AS}
}
```
```
@article{mockingjay,
   title={Mockingjay: Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders},
   ISBN={9781509066315},
   url={http://dx.doi.org/10.1109/ICASSP40776.2020.9054458},
   DOI={10.1109/icassp40776.2020.9054458},
   journal={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
   publisher={IEEE},
   author={Liu, Andy T. and Yang, Shu-wen and Chi, Po-Han and Hsu, Po-chun and Lee, Hung-yi},
   year={2020},
   month={May}
}
```

- If you use our organized upstream interface and features, or the *SUPERB* downstream benchmark, please consider citing the following:
```
@inproceedings{yang21c_interspeech,
  author={Shu-wen Yang and Po-Han Chi and Yung-Sung Chuang and Cheng-I Jeff Lai and Kushal Lakhotia and Yist Y. Lin and Andy T. Liu and Jiatong Shi and Xuankai Chang and Guan-Ting Lin and Tzu-Hsien Huang and Wei-Cheng Tseng and Ko-tik Lee and Da-Rong Liu and Zili Huang and Shuyan Dong and Shang-Wen Li and Shinji Watanabe and Abdelrahman Mohamed and Hung-yi Lee},
  title={{SUPERB: Speech Processing Universal PERformance Benchmark}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={1194--1198},
  doi={10.21437/Interspeech.2021-1775}
}
```
