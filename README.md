<p align="center">
    <img src="https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/blob/master/file/S3PRL-logo.png" width="900"/>
    <br>
    <br>
    <a href="https://github.com/s3prl/s3prl/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
    <a href="https://github.com/s3prl/s3prl/actions"><img alt="Build" src="https://github.com/allenai/allennlp/workflows/Master/badge.svg?event=push&branch=master"></a>
    <a href="#development-pattern-for-contributors"><img alt="Codecov" src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg"></a>
    <a href="https://github.com/s3prl/s3prl/issues"><img alt="Bitbucket open issues" src="https://img.shields.io/github/issues/s3prl/s3prl"></a>
</p>

What's New
------------------------------------
* Jan 2021: Readme updated with detailed instructions on how to use our latest version!
* Dec 2020: We are migrating to a newer version for a more general, flexible, and scalable code. See the introduction below for more information! The legacy verison can be accessed by checking out to the tag `v0.1.0`: `git checkout v0.1.0`.

------------------------------------

Introduction
------------------------------------
* This is an open source toolkit called S3PRL, which stands for **S**elf-**S**upervised **S**peech **P**re-training and **R**epresentation **L**earning.
* In this toolkit, various *upstream* self-supervised speech models are available with easy-to-load setups, and *downstream* evaluation tasks are available with easy-to-use scripts.
* Below is an intuitive illustration on how this toolkit may help you:

<img src="https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning/blob/master/file/S3PRL-interface.png" width="900"/>

- **View the list of *upstreams* we support: [Upstream README](https://github.com/s3prl/s3prl/blob/master/upstream/README.md)**
- **View the list of *downstreams* we support: [Downstream README](https://github.com/s3prl/s3prl/tree/master/downstream/README.md)**

* Feel free to use or modify our toolkit in your research, any bug report or improvement suggestion will be appreciated.
* If you have any questions, please [open up a new issue](https://github.com/s3prl/s3prl/issues).
* If you find this toolkit helpful to your research, please do consider to cite [our papers](#Citation), thanks!

<details><summary>List of papers that used our toolkit (Feel free to add your own paper by making a pull request)</summary><p>

* **Self-Supervised Pretraining**
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
* **Explanability**
  + [Understanding Self-Attention of Self-Supervised Audio Transformers (Yang et al., 2020)](https://arxiv.org/abs/2006.03265)
    ```
    @misc{understandingSAT,
        title={Understanding Self-Attention of Self-Supervised Audio Transformers},
        author={Shu-wen Yang and Andy T. Liu and Hung-yi Lee},
        year={2020},
        eprint={2006.03265},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    ```
* **Adversarial Attack**
  + [Defense for Black-box Attacks on Anti-spoofing Models by Self-Supervised Learning (Wu et al., 2020)](https://arxiv.org/abs/2006.03214), code for computing LNSR: [utility/observe_lnsr.py](https://github.com/s3prl/s3prl/blob/master/utility/observe_lnsr.py)
    ```
    @misc{mockingjay_defense,
        title={Defense for Black-box Attacks on Anti-spoofing Models by Self-Supervised Learning},
        author={Haibin Wu and Andy T. Liu and Hung-yi Lee},
        year={2020},
        eprint={2006.03214},
        archivePrefix={arXiv},
        primaryClass={eess.AS}
    }
    ```
  + [Adversarial Defense for Automatic Speaker Verification by Cascaded Self-Supervised Learning Models (Wu et al., 2021)](https://andi611.github.io/)

</p></details>

------------------------------------

Table of Contents
------------------------------------

<!--ts-->
   * [Table of contents](#table-of-contents)
   * [Installation](#installation)
   * [Using upstreams](https://github.com/s3prl/s3prl/tree/master/upstream/README.md)
   * [Using downstreams](https://github.com/s3prl/s3prl/tree/master/downstream/README.md)
   * [Train upstream models](#train-upstream-models)
   * [Development pattern for contributors](#development-pattern-for-contributors)
   * [Reference](#reference)
   * [Citation](#citation)
<!--te-->

------------------------------------

Installation
------------------------------------

* **Python** >= 3.6
* **PyTorch** version >= 1.7.0
* For pre-training new upstream models, you'll also need high-end GPU(s).
* To develop locally, install s3prl by:
```bash=
git clone https://github.com/s3prl/s3prl.git
cd s3prl
pip install -r requirements.txt
```
* If you encounter error with a specific upstream model, you can look into the `README.md` under each `upsream` folder.
* To use upstream models with the hub interface, cloning this repo is not required, only the `requirements.txt` in root directory and the one located at each `upstream` folder are needed.

[Back to Top](#table-of-contents)

------------------------------------

Using upstreams
------------------------------------
- Instructions are documented here: [Upstream README](https://github.com/s3prl/s3prl/tree/master/upstream/README.md)

[Back to Top](#table-of-contents)

------------------------------------

Using downstreams
------------------------------------
- *Warning: we are still developing and testing some downstream tasks, documentation of a task will be added once it has been fully tested.*
- Instructions are documented here: [Downstream README](https://github.com/s3prl/s3prl/tree/master/downstream/README.md)

[Back to Top](#table-of-contents)

------------------------------------

Train upstream models
------------------------------------
- If you wish to train your own upstream models, 
please follow the instructions here: [Pretrain README](https://github.com/s3prl/s3prl/tree/master/pretrain/README.md)

[Back to Top](#table-of-contents)

------------------------------------

Development pattern for contributors
------------------------------------
1. [Create a personal fork](https://help.github.com/articles/fork-a-repo/) of the [main S3PRL repository](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning) in GitHub.
2. Make your changes in a named branch different from `master`, e.g. you create a branch `new-awesome-feature`.
3. Contact us if you have any questions during development.
4. [Generate a pull request](https://help.github.com/articles/creating-a-pull-request/) through the Web interface of GitHub.
5. Please verify that your code is free of basic mistakes, we appreciate any contribution!

[Back to Top](#table-of-contents)

------------------------------------

Reference Repos
------------------------------------
* [Pytorch](https://github.com/pytorch/pytorch), Pytorch.
* [Audio](https://github.com/pytorch/audio), Pytorch.
* [Kaldi](https://github.com/kaldi-asr/kaldi), Kaldi-ASR.
* [Transformers](https://github.com/huggingface/transformers), Hugging Face.
* [PyTorch-Kaldi](https://github.com/mravanelli/pytorch-kaldi), Mirco Ravanelli.
* [fairseq](https://github.com/pytorch/fairseq), Facebook AI Research.
* [CPC](https://github.com/facebookresearch/CPC_audio), Facebook AI Research.
* [APC](https://github.com/iamyuanchung/Autoregressive-Predictive-Coding), Yu-An Chung.
* [NPC](https://github.com/Alexander-H-Liu/NPC), Alexander-H-Liu.

[Back to Top](#table-of-contents)

Citation
------------------------------------
- The S3PRL Toolkit:
```
@misc{S3PRL,
  author = {Andy T. Liu and Yang Shu-wen},
  title = {S3PRL: The Self-Supervised Speech Pre-training and Representation Learning Toolkit},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/s3prl/s3prl}
}
```
