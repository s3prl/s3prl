:tocdepth: 2

S3PRL Upstream Collection
=======================================

We collect almost all the existing SSL pre-trained models in S3PRL,
so you can import and use them easily in an unified I/O interface.

:obj:`s3prl.nn.upstream.S3PRLUpstream` is an easy interface to retrieve all the self-supervised learning (SSL) pre-trained models
available in S3PRL. the :code:`name` argument for :obj:`s3prl.nn.upstream.S3PRLUpstream` specifies the checkpoint,
and then the pre-trained models in this checkpoint will be automatically constructed and
initialized.

Here is an example on how to get a hubert model and its representation using the :code:`name='hubert'`:

.. code-block:: python

    import torch
    from s3prl.nn import S3PRLUpstream

    model = S3PRLUpstream("hubert")
    model.eval()

    with torch.no_grad():
        wavs = torch.randn(2, 16000 * 2)
        wavs_len = torch.LongTensor([16000 * 1, 16000 * 2])
        all_hs, all_hs_len = model(wavs, wavs_len)

    for hs, hs_len in zip(all_hs, all_hs_len):
        assert isinstance(hs, torch.FloatTensor)
        assert isinstance(hs_len, torch.LongTensor)

        batch_size, max_seq_len, hidden_size = hs.shape
        assert hs_len.dim() == 1

.. tip::

    For each SSL learning method, like wav2vec 2.0, there are several checkpoint variants, trained by
    different amount of unlabeled data, or different model sizes. Hence there are also various
    :code:`name` to retrieve these different models.

    Like, the HuBERT method has "hubert" and "hubert_large_ll60k" different names for different
    checkpoint variants.

The following includes the model and checkpoint information for each :code:`name`, including the releasing date,
paper, citation, model architecture, pre-training data, criterion, and their source code. The format follows:



SSL Method
--------------------------------------------------------
`Paper full title with arxiv link <https://arxiv.org/>`_

.. code-block:: bash

    @article{citation-block,
        title={Paper Title},
        author={Authors},
        year={2020},
        month={May}
    }

The information shared across checkpoint variants.

name1
~~~~~~~~~~~~~~~~~~~

The detailed specific information for this checkpoint variant (:code:`name=name1`)

name2
~~~~~~~~~~~~~~~~~~~

The detailed specific information for this checkpoint variant (:code:`name=name2`)



Mockingjay
--------------------------------------------------------
`Mockingjay: Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders <https://arxiv.org/abs/1910.12638>`_

.. code-block:: bash

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

Mockingjay is a BERT on Spectrogram, with 12-layers of transformer encoders in the paper.


mockingjay
~~~~~~~~~~~~~~~~

This is alias for `mockingjay_origin`_

mockingjay_origin
~~~~~~~~~~~~~~~~~~~~~~~~

This is alias for `mockingjay_logMelLinearLarge_T_AdamW_b32_500k_360hr_drop1`_

mockingjay_100hr
~~~~~~~~~~~~~~~~

This is alias for `mockingjay_logMelBase_T_AdamW_b32_200k_100hr`_

mockingjay_960hr
~~~~~~~~~~~~~~~~

This is alias for `mockingjay_logMelBase_T_AdamW_b32_1m_960hr_drop1`_

mockingjay_logMelBase_T_AdamW_b32_200k_100hr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Feature: 80-dim log Mel
- Alteration: time
- Optimizer: AdamW
- Batch size: 32
- Total steps: 200k
- Unlabled Speech: LibriSpeech 100hr

mockingjay_logMelLinearLarge_T_AdamW_b32_500k_360hr_drop1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Feature: 80-dim log Mel (input) / 201-dim Linear (target)
- Alteration: time
- Optimizer: AdamW
- Batch size: 32
- Total steps: 500k
- Unlabled Speech: LibriSpeech 360hr

mockingjay_logMelBase_T_AdamW_b32_1m_960hr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Feature: 80-dim log Mel
- Alteration: time
- Optimizer: AdamW
- Batch size: 32
- Total steps: 1M
- Unlabled Speech: LibriSpeech 960hr

mockingjay_logMelBase_T_AdamW_b32_1m_960hr_drop1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Feature: 80-dim log Mel
- Alteration: time
- Optimizer: AdamW
- Batch size: 32
- Total steps: 1M
- Unlabled Speech: LibriSpeech 960hr
- Differences: Dropout of 0.1 (instead of 0.3)


mockingjay_logMelBase_T_AdamW_b32_1m_960hr_seq3k
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Feature: 80-dim log Mel
- Alteration: time
- Optimizer: AdamW
- Batch size: 32
- Total steps: 1M
- Unlabled Speech: LibriSpeech 960hr
- Differences: sequence length of 3k (instead of 1.5k)



TERA
--------------------------------------------------------
`TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech <https://arxiv.org/abs/2007.06028>`_

.. code-block:: bash

    @misc{tera,
        title={TERA: Self-Supervised Learning of Transformer Encoder Representation for Speech},
        author={Andy T. Liu and Shang-Wen Li and Hung-yi Lee},
        year={2020},
        eprint={2007.06028},
        archivePrefix={arXiv},
        primaryClass={eess.AS}
    }


tera
~~~~~~~~~~~~~~~~

This is alias for `tera_960hr`_

tera_100hr
~~~~~~~~~~~~~~~~~~

This is alias for `tera_logMelBase_T_F_M_AdamW_b32_200k_100hr`_

tera_960hr
~~~~~~~~~~~~~~~~~~~

This is alias for `tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1`_

tera_logMelBase_T_F_AdamW_b32_200k_100hr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Feature: 80-dim log Mel
- Alteration: time + freq
- Optimizer: AdamW
- Batch size: 32
- Total steps: 200k
- Unlabled Speech: LibriSpeech 100hr

tera_logMelBase_T_F_M_AdamW_b32_200k_100hr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Feature: 80-dim log Mel
- Alteration: time + freq + mag
- Optimizer: AdamW
- Batch size: 32
- Total steps: 200k
- Unlabled Speech: LibriSpeech 100hr

tera_logMelBase_T_F_AdamW_b32_1m_960hr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Feature: 80-dim log Mel
- Alteration: time + freq
- Optimizer: AdamW
- Batch size: 32
- Total steps: 1M
- Unlabled Speech: LibriSpeech 960hr

tera_logMelBase_T_F_AdamW_b32_1m_960hr_drop1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Feature: 80-dim log Mel
- Alteration: time + freq
- Optimizer: AdamW
- Batch size: 32
- Total steps: 1M
- Unlabled Speech: LibriSpeech 960hr
- Differences: Dropout of 0.1 (instead of 0.3)

tera_logMelBase_T_F_AdamW_b32_1m_960hr_seq3k
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Feature: 80-dim log Mel
- Alteration: time + freq
- Optimizer: AdamW
- Batch size: 32
- Total steps: 1M
- Unlabled Speech: LibriSpeech 960hr
- Differences: sequence length of 3k (instead of 1.5k)

tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Feature: 80-dim log Mel
- Alteration: time + freq + mag
- Optimizer: AdamW
- Batch size: 32
- Total steps: 1M
- Unlabled Speech: 960hr
- Differences: Dropout of 0.1 (instead of 0.3)

tera_fbankBase_T_F_AdamW_b32_200k_100hr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Feature: 240-dim fbank
- Alteration: time + freq
- Optimizer: AdamW
- Batch size: 32
- Total steps: 200k
- Unlabled Speech: LibriSpeech 100hr



Audio ALBERT
--------------------------------------------------------
`Audio ALBERT: A Lite BERT for Self-supervised Learning of Audio Representation <https://arxiv.org/abs/2007.06028>`_

.. code-block:: bash

    @inproceedings{chi2021audio,
        title={Audio albert: A lite bert for self-supervised learning of audio representation},
        author={Chi, Po-Han and Chung, Pei-Hung and Wu, Tsung-Han and Hsieh, Chun-Cheng and Chen, Yen-Hao and Li, Shang-Wen and Lee, Hung-yi},
        booktitle={2021 IEEE Spoken Language Technology Workshop (SLT)},
        pages={344--350},
        year={2021},
        organization={IEEE}
    }


audio_albert
~~~~~~~~~~~~~~~~

This is alias of `audio_albert_960hr`_


audio_albert_960hr
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is alias of `audio_albert_logMelBase_T_share_AdamW_b32_1m_960hr_drop1`_


audio_albert_logMelBase_T_share_AdamW_b32_1m_960hr_drop1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Feature: 80-dim log Mel
- Alteration: time
- Optimizer: AdamW
- Batch size: 32
- Total steps: 1M
- Unlabled Speech: LibriSpeech 960hr



APC
--------------------------------------------------------
`An Unsupervised Autoregressive Model for Speech Representation Learning <https://arxiv.org/abs/1904.03240>`_

.. code-block:: bash

    @inproceedings{chung2019unsupervised,
        title = {An unsupervised autoregressive model for speech representation learning},
        author = {Chung, Yu-An and Hsu, Wei-Ning and Tang, Hao and Glass, James},
        booktitle = {Interspeech},
        year = {2019}
    }


apc
~~~~~~~~~~~~~~~~

This is alias of `apc_360hr`_


apc_360hr
~~~~~~~~~~~~~~~~~~

- Unlabled Speech: LibriSpeech 360hr


apc_960hr
~~~~~~~~~~~~~~~~~~

- Unlabled Speech: LibriSpeech 960hr



VQ-APC
--------------------------------------------------------
`Vector-Quantized Autoregressive Predictive Coding <https://arxiv.org/abs/2005.08392>`_

.. code-block:: bash

    @inproceedings{chung2020vqapc,
        title = {Vector-quantized autoregressive predictive coding},
        autohor = {Chung, Yu-An and Tang, Hao and Glass, James},
        booktitle = {Interspeech},
        year = {2020}
    }

vq_apc
~~~~~~~~~~~~~~~~

This is alias of `vq_apc_360hr`_


vq_apc_360hr
~~~~~~~~~~~~~~~~

- Unlabled Speech: LibriSpeech 360hr


vq_apc_960hr
~~~~~~~~~~~~~~~~~

- Unlabled Speech: LibriSpeech 960hr



NPC
--------------------------------------------------------
`Non-Autoregressive Predictive Coding for Learning Speech Representations from Local Dependencies <https://arxiv.org/abs/2011.00406>`_

.. code-block:: bash

    @article{liu2020nonautoregressive,
        title   = {Non-Autoregressive Predictive Coding for Learning Speech Representations from Local Dependencies},
        author  = {Liu, Alexander and Chung, Yu-An and Glass, James},
        journal = {arXiv preprint arXiv:2011.00406},
        year    = {2020}
    }


npc
~~~~~~~~~~~~~~~~

This is alias of `npc_360hr`_


npc_360hr
~~~~~~~~~~~~~~~~~~

- Unlabled Speech: LibriSpeech 360hr


npc_960hr
~~~~~~~~~~~~~~~~~~~~

- Unlabled Speech: LibriSpeech 960hr



PASE+
--------------------------------------------------------
`Multi-task self-supervised learning for Robust Speech Recognition <https://arxiv.org/abs/2001.09239>`_

.. code-block:: bash

    @inproceedings{ravanelli2020multi,
        title={Multi-task self-supervised learning for robust speech recognition},
        author={Ravanelli, Mirco and Zhong, Jianyuan and Pascual, Santiago and Swietojanski, Pawel and Monteiro, Joao and Trmal, Jan and Bengio, Yoshua},
        booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        pages={6989--6993},
        year={2020},
        organization={IEEE}
    }

.. hint::

    To use PASE models, there are many extra dependencies required to install.
    Please follow the below installation instruction:

    .. code-block:: bash

        pip install -r https://raw.githubusercontent.com/s3prl/s3prl/master/s3prl/upstream/pase/requirements.txt


pase_plus
~~~~~~~~~~~~~~~~

- Unlabled Speech: LibriSpeech 50hr



Modified CPC
--------------------------------------------------------
`Unsupervised pretraining transfers well across languages <https://arxiv.org/abs/2002.02848>`_

.. code-block:: bash

    @inproceedings{riviere2020unsupervised,
        title={Unsupervised pretraining transfers well across languages},
        author={Riviere, Morgane and Joulin, Armand and Mazar{\'e}, Pierre-Emmanuel and Dupoux, Emmanuel},
        booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        pages={7414--7418},
        year={2020},
        organization={IEEE}
    }

.. note::

    This is a slightly improved version on the original CPC by DeepMind. To cite the DeepMind version:

    .. code-block:: bash

        @article{oord2018representation,
            title={Representation learning with contrastive predictive coding},
            author={Oord, Aaron van den and Li, Yazhe and Vinyals, Oriol},
            journal={arXiv preprint arXiv:1807.03748},
            year={2018}
        }


modified_cpc
~~~~~~~~~~~~~~~~

- Unlabled Speech: LibriLight 60k hours



DeCoAR
--------------------------------------------------------
`Deep contextualized acoustic representations for semi-supervised speech recognition <https://arxiv.org/abs/1912.01679>`_

.. code-block:: bash

    @inproceedings{ling2020deep,
        title={Deep contextualized acoustic representations for semi-supervised speech recognition},
        author={Ling, Shaoshi and Liu, Yuzong and Salazar, Julian and Kirchhoff, Katrin},
        booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        pages={6429--6433},
        year={2020},
        organization={IEEE}
    }


decoar_layers
~~~~~~~~~~~~~~~~

- Unlabled Speech: LibriSpeech 960hr


DeCoAR 2.0
--------------------------------------------------------
`DeCoAR 2.0: Deep Contextualized Acoustic Representations with Vector Quantization <https://arxiv.org/abs/2012.06659>`_

.. code-block:: bash

    @misc{ling2020decoar,
        title={DeCoAR 2.0: Deep Contextualized Acoustic Representations with Vector Quantization}, 
        author={Shaoshi Ling and Yuzong Liu},
        year={2020},
        eprint={2012.06659},
        archivePrefix={arXiv},
        primaryClass={eess.AS}
    }


decoar2
~~~~~~~~~~~~~~~~~~~~~

- Unlabled Speech: LibriSpeech 960hr



wav2vec
--------------------------------------------------
`wav2vec: Unsupervised Pre-Training for Speech Recognition <https://arxiv.org/abs/1904.05862>`_

.. code-block:: bash

    @article{schneider2019wav2vec,
        title={wav2vec: Unsupervised Pre-Training for Speech Recognition},
        author={Schneider, Steffen and Baevski, Alexei and Collobert, Ronan and Auli, Michael},
        journal={Proc. Interspeech 2019},
        pages={3465--3469},
        year={2019}
    }


wav2vec
~~~~~~~~~~~

This is alias of `wav2vec_large`_


wav2vec_large
~~~~~~~~~~~~~~~

This is the official wav2vec model from fairseq.

- Unlabled Speech: LibriSpeech 960hr


vq-wav2vec
--------------------------------------------------
`vq-wav2vec: Self-supervised learning of discrete speech representations <https://arxiv.org/abs/1910.05453>`_

.. code-block:: bash

    @inproceedings{baevski2019vq,
        title={vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations},
        author={Baevski, Alexei and Schneider, Steffen and Auli, Michael},
        booktitle={International Conference on Learning Representations},
        year={2019}
    }

.. note::

    We only take the Conv encoders' hidden_states for vq-wav2vec in this SSL method category.
    If you wish to consider the BERT model after ths Conv encoders, please refer to `Discrete BERT`_.

vq_wav2vec
~~~~~~~~~~~

This is alias of `vq_wav2vec_gumbel`_


vq_wav2vec_gumbel
~~~~~~~~~~~~~~~~~~~~

This is the official vq-wav2vec model from fairseq.
This model uses gumbel-softmax as the quantization technique

- Unlabled Speech: LibriSpeech 960hr


vq_wav2vec_kmeans
~~~~~~~~~~~~~~~~~~~~~

This is the official vq-wav2vec model from fairseq.
This model uses K-means as the quantization technique


Discrete BERT
--------------------------------------------------
`vq-wav2vec: Self-supervised learning of discrete speech representations <https://arxiv.org/abs/1910.05453>`_

.. code-block:: bash

    @inproceedings{baevski2019vq,
        title={vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations},
        author={Baevski, Alexei and Schneider, Steffen and Auli, Michael},
        booktitle={International Conference on Learning Representations},
        year={2019}
    }

This method takes the Conv feature encoder's output, quantize it into token ids, and feed the
tokens into a NLP BERT (Specifically, RoBERTa). The output hidden_states are all the hidden hidden_states
of the NLP BERT (excluding the hidden_states in `vq-wav2vec`_)


discretebert
~~~~~~~~~~~~~~~~

Alias of `vq_wav2vec_kmeans_roberta`_


vq_wav2vec_kmeans_roberta
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This model uses `vq_wav2vec_kmeans`_ as the frontend waveform tokenizer. After the waveform is tokenized
into a sequence of token ids, tokens are then fed into a RoBERTa model.



wav2vec 2.0
--------------------------------------------------
`wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations <https://arxiv.org/abs/2006.11477>`_

.. code-block:: bash

    @article{baevski2020wav2vec,
        title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
        author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
        journal={Advances in Neural Information Processing Systems},
        volume={33},
        pages={12449--12460},
        year={2020}
    }


wav2vec2
~~~~~~~~~~~~~~~~

This is the alias of `wav2vec2_base_960`_


wav2vec2_base_960
~~~~~~~~~~~~~~~~~~~~~~~~~~
This is the official wav2vec 2.0 model in fairseq

- Architecture: 12-layer Transformer encoders
- Unlabled Speech: LibriSpeech 960hr


wav2vec2_large_960
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Architecture: 24-layer Transformer encoders
- Unlabled Speech: LibriSpeech 960hr


wav2vec2_large_ll60k
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Architecture: 24-layer Transformer encoders
- Unlabled Speech: LibriLight LL60k hours


wav2vec2_large_lv60_cv_swbd_fsh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Large model trained on Libri-Light 60k hours + CommonVoice + Switchboard + Fisher

- Architecture: 24-layer Transformer encoders
- Unlabeled Speech: Libri-Light 60k hours + CommonVoice + Switchboard + Fisher


wav2vec2_conformer_relpos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Architecture: 24-layer Conformer encoders with relative positional encoding
- Unlabeled Speech: LibriLight LL60k hours


wav2vec2_conformer_rope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Architecture: 24-layer Conformer encoders with ROPE positional encoding
- Unlabeled Speech: LibriLight LL60k hours


xlsr_53
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The wav2vec 2.0 model trained on multilingual presented in `Unsupervised Cross-lingual Representation Learning for Speech Recognition <https://arxiv.org/abs/2006.13979>`_

.. code-block:: bash

    @article{conneau2020unsupervised,
        title={Unsupervised cross-lingual representation learning for speech recognition},
        author={Conneau, Alexis and Baevski, Alexei and Collobert, Ronan and Mohamed, Abdelrahman and Auli, Michael},
        journal={arXiv preprint arXiv:2006.13979},
        year={2020}
    }


XLS-R
--------------------------------------------------
`XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale <https://arxiv.org/abs/2111.09296>`_

.. code-block:: bash

    @article{babu2021xls,
    title={XLS-R: Self-supervised cross-lingual speech representation learning at scale},
    author={Babu, Arun and Wang, Changhan and Tjandra, Andros and Lakhotia, Kushal and Xu, Qiantong and Goyal, Naman and Singh, Kritika and von Platen, Patrick and Saraf, Yatharth and Pino, Juan and others},
    journal={arXiv preprint arXiv:2111.09296},
    year={2021}
    }


xls_r_300m
~~~~~~~~~~~~~~~~~~~~~

- Unlabled Speech: 128 languages, 436K hours


xls_r_1b
~~~~~~~~~~~~~~~~~~~~~

- Unlabled Speech: 128 languages, 436K hours


xls_r_2b
~~~~~~~~~~~~~~~~~~~~~

- Unlabled Speech: 128 languages, 436K hours


HuBERT
--------------------------------------------------
`HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units <https://arxiv.org/abs/2106.07447>`_

.. code-block:: bash

    @article{hsu2021hubert,
        title={Hubert: Self-supervised speech representation learning by masked prediction of hidden units},
        author={Hsu, Wei-Ning and Bolte, Benjamin and Tsai, Yao-Hung Hubert and Lakhotia, Kushal and Salakhutdinov, Ruslan and Mohamed, Abdelrahman},
        journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
        volume={29},
        pages={3451--3460},
        year={2021},
        publisher={IEEE}
    }


hubert
~~~~~~~~~~~~~~~~~~~~~

This is alias of `hubert_base`_


hubert_base
~~~~~~~~~~~~~~~~~~~~~

- Unlabled Speech: LibriSpeech 960hr


hubert_large_ll60k
~~~~~~~~~~~~~~~~~~~~~

- Unlabled Speech: LibriLight ll60k hours


DistilHuBERT
----------------------
`DistilHuBERT: Speech Representation Learning by Layer-wise Distillation of Hidden-unit BERT <https://arxiv.org/abs/2110.01900>`_

.. code-block:: bash

    @inproceedings{chang2022distilhubert,
        title={DistilHuBERT: Speech representation learning by layer-wise distillation of hidden-unit BERT},
        author={Chang, Heng-Jui and Yang, Shu-wen and Lee, Hung-yi},
        booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        pages={7087--7091},
        year={2022},
        organization={IEEE}
    }


distilhubert
~~~~~~~~~~~~~~~~~~~~~

Alias of `distilhubert_base`_


distilhubert_base
~~~~~~~~~~~~~~~~~~~~~

- Teacher: `hubert_base`_
- Unlabled Speech: LibriSpeech 960hr


HuBERT-MGR
--------------------------------------------------
`Improving Distortion Robustness of Self-supervised Speech Processing Tasks with Domain Adaptation <https://arxiv.org/abs/2203.16104>`_

.. code-block:: bash

    @article{huang2022improving,
        title={Improving Distortion Robustness of Self-supervised Speech Processing Tasks with Domain Adaptation},
        author={Huang, Kuan Po and Fu, Yu-Kuan and Zhang, Yu and Lee, Hung-yi},
        journal={arXiv preprint arXiv:2203.16104},
        year={2022}
    }


hubert_base_robust_mgr
~~~~~~~~~~~~~~~~~~~~~~~

- Unlabled Speech: LibriSpeech 960hr
- Augmentation: MUSAN, gaussian, reverberation


Unispeech-SAT
--------------------------------------------------
`Unispeech-sat: Universal speech representation learning with speaker aware pre-training <https://arxiv.org/abs/2110.05752>`_

.. code-block:: bash

    @inproceedings{chen2022unispeech,
        title={Unispeech-sat: Universal speech representation learning with speaker aware pre-training},
        author={Chen, Sanyuan and Wu, Yu and Wang, Chengyi and Chen, Zhengyang and Chen, Zhuo and Liu, Shujie and Wu, Jian and Qian, Yao and Wei, Furu and Li, Jinyu and others},
        booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        pages={6152--6156},
        year={2022},
        organization={IEEE}
    }


unispeech_sat
~~~~~~~~~~~~~~~~~~~~~

Alias of `unispeech_sat_base`_


unispeech_sat_base
~~~~~~~~~~~~~~~~~~~~~~

- Model Architecture: 12 layers Transformer blocks
- Unlabled Speech: LibriSpeech 960 hours


unispeech_sat_base_plus
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Model Architecture: 12 layers Transformer blocks
- Unlabled Speech: LibriLight 60k hours + Gigaspeech 10k hours + VoxPopuli 24k hours = 94k hours


unispeech_sat_large
~~~~~~~~~~~~~~~~~~~~~~~~

- Model Architecture: 24 layers Transformer blocks
- Unlabled Speech: LibriLight 60k hours + Gigaspeech 10k hours + VoxPopuli 24k hours = 94k hours



WavLM
--------------------------------------------------
`WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing <https://arxiv.org/abs/2110.13900>`_

.. code-block:: bash

    @article{Chen2021WavLM,
        title   = {WavLM: Large-Scale Self-Supervised  Pre-training   for Full Stack Speech Processing},
        author  = {Sanyuan Chen and Chengyi Wang and Zhengyang Chen and Yu Wu and Shujie Liu and Zhuo Chen and Jinyu Li and Naoyuki Kanda and Takuya Yoshioka and Xiong Xiao and Jian Wu and Long Zhou and Shuo Ren and Yanmin Qian and Yao Qian and Jian Wu and Michael Zeng and Furu Wei},
        eprint={2110.13900},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        year={2021}
    }


wavlm
~~~~~~~~~~~~~~~~~

Alias of `wavlm_base_plus`_


wavlm_base
~~~~~~~~~~~~~~~~

- Model Architecture: 12 layers Transformer blocks
- Unlabled Speech: LibriSpeech 960 hours


wavlm_base_plus
~~~~~~~~~~~~~~~~~~~~~

- Model Architecture: 12 layers Transformer blocks
- Unlabled Speech: LibriLight 60k hours + Gigaspeech 10k hours + VoxPopuli 24k hours = 94k hours


wavlm_large
~~~~~~~~~~~~~~~~~~~~~

- Model Architecture: 24 layers Transformer blocks
- Unlabled Speech: LibriLight 60k hours + Gigaspeech 10k hours + VoxPopuli 24k hours = 94k hours


data2vec
--------------------------------------------------
`data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language <https://arxiv.org/abs/2202.03555>`_

.. code-block:: bash

    @article{baevski2022data2vec,
        title={Data2vec: A general framework for self-supervised learning in speech, vision and language},
        author={Baevski, Alexei and Hsu, Wei-Ning and Xu, Qiantong and Babu, Arun and Gu, Jiatao and Auli, Michael},
        journal={arXiv preprint arXiv:2202.03555},
        year={2022}
    }


data2vec
~~~~~~~~~~~~~~~~~

Alias of `data2vec_base_960`_


data2vec_base_960
~~~~~~~~~~~~~~~~~~

- Model Architecture: 12 layers Transformer blocks
- Unlabled Speech: LibriSpeech 960 hours


data2vec_large_ll60k
~~~~~~~~~~~~~~~~~~~~~

- Model Architecture: 24 layers Transformer blocks
- Unlabled Speech: LibriLight 60k hours


AST
--------------------------------------------------
`AST: Audio Spectrogram Transformer <https://arxiv.org/abs/2104.01778>`_

.. code-block:: bash

    @article{gong2021ast,
        title={Ast: Audio spectrogram transformer},
        author={Gong, Yuan and Chung, Yu-An and Glass, James},
        journal={arXiv preprint arXiv:2104.01778},
        year={2021}
    }


ast
~~~~~~~~~~~~~~~~~~

- Labeled Data: AudioSet


SSAST
--------------------------------------------------
`SSAST: Self-Supervised Audio Spectrogram Transformer <https://arxiv.org/abs/2110.09784>`_

.. code-block:: bash

    @inproceedings{gong2022ssast,
        title={Ssast: Self-supervised audio spectrogram transformer},
        author={Gong, Yuan and Lai, Cheng-I and Chung, Yu-An and Glass, James},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={36},
        number={10},
        pages={10699--10709},
        year={2022}
    }


ssast_frame_base
~~~~~~~~~~~~~~~~~~

- Unlabled Data: LibriSpeech & AudioSet


MAE-AST
--------------------------------------------------
`MAE-AST: Masked Autoencoding Audio Spectrogram Transformer <https://arxiv.org/abs/2203.16691>`_

.. code-block:: bash

    @article{baade2022mae,
        title={MAE-AST: Masked Autoencoding Audio Spectrogram Transformer},
        author={Baade, Alan and Peng, Puyuan and Harwath, David},
        journal={arXiv preprint arXiv:2203.16691},
        year={2022}
    }


mae_ast_frame
~~~~~~~~~~~~~~~~~~

- Unlabled Data: LibriSpeech & AudioSet


mae_ast_patch
~~~~~~~~~~~~~~~~~~

- Unlabled Data: LibriSpeech & AudioSet



Byol-A
--------------------------------------------------
`BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation <https://arxiv.org/abs/2103.06695>`_

.. code-block:: bash

    @inproceedings{niizumi2021byol,
        title={BYOL for audio: Self-supervised learning for general-purpose audio representation},
        author={Niizumi, Daisuke and Takeuchi, Daiki and Ohishi, Yasunori and Harada, Noboru and Kashino, Kunio},
        booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
        pages={1--8},
        year={2021},
        organization={IEEE}
    }


byol_a_2048
~~~~~~~~~~~~~~~~~~

- Unlabled Data: AudioSet


byol_a_1024
~~~~~~~~~~~~~~~~~~

- Unlabled Data: AudioSet


byol_a_512
~~~~~~~~~~~~~~~~~~

- Unlabled Data: AudioSet


Byol-S
--------------------------------------------------
`BYOL-S: Learning Self-supervised Speech Representations by Bootstrapping <https://arxiv.org/abs/2206.12038>`_

.. code-block:: bash

    @article{elbanna2022byol,
        title={Byol-s: Learning self-supervised speech representations by bootstrapping},
        author={Elbanna, Gasser and Scheidwasser-Clow, Neil and Kegler, Mikolaj and Beckmann, Pierre and Hajal, Karl El and Cernak, Milos},
        journal={arXiv preprint arXiv:2206.12038},
        year={2022}
    }


byol_s_default
~~~~~~~~~~~~~~~~~~

- Unlabled Data: AudioSet (Speech subset)


byol_s_cvt
~~~~~~~~~~~~~~~~~~

- Unlabled Data: AudioSet (Speech subset)


byol_s_resnetish34
~~~~~~~~~~~~~~~~~~

- Unlabled Data: AudioSet (Speech subset)


VGGish
--------------------------------------------------
`CNN Architectures for Large-Scale Audio Classification <https://arxiv.org/abs/1609.09430>`_

.. code-block:: bash

    @inproceedings{hershey2017cnn,
        title={CNN architectures for large-scale audio classification},
        author={Hershey, Shawn and Chaudhuri, Sourish and Ellis, Daniel PW and Gemmeke, Jort F and Jansen, Aren and Moore, R Channing and Plakal, Manoj and Platt, Devin and Saurous, Rif A and Seybold, Bryan and others},
        booktitle={2017 ieee international conference on acoustics, speech and signal processing (icassp)},
        pages={131--135},
        year={2017},
        organization={IEEE}
    }


vggish
~~~~~~~~~~~~~~~~~~

- Labaled Data: AudioSet
