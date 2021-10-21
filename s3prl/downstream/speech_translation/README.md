## ST: Speech Translation

specified by the command `-d speech_translation`

#### Prepare data

Following is the example to prepare [COVOST2](https://github.com/facebookresearch/covost) en-de dataset

1. Download [Common Voice audio clips and transcripts (english)](https://commonvoice.mozilla.org/en/datasets) (Common Voice Corpus 4).

2. Change the path in `prepare_data/prepare_covo.sh` (you can also change the `src_lang` and `tgt_lang` to prepare data of other language pairs)

```bash
covo_root="root directory of covost"
src_lang=en
tgt_lang=de
```

3. Run the following script

```bash
cd downstream/speech_translation/prepare_data/
bash prepare_covo.sh
```

4. Check the prepared file structure

* processed data

```bash
s3prl/
└── data/
    └──covost_en_de/
        ├── train.tsv
        ├── dev.tsv
        ├── test.tsv
        ├── spm-[src|tgt]_text.[model|vocab|text]
        ├── config.yaml
        └── prepare_data.log
```

* origin tsv files

```bash
s3prl/
└── downstream/
    └──speech_translation/
        └──prepare_data/
            └──covost_tsv/
                └──covost_v2.<src_lang>_<tgt_lang>.[train|dev|test].tsv
```

5. [Optional] If you use other dataset or change the language/data config in `prepare_data/prepare_covo.sh`, you will also need to change the language/data config in `config.yaml`.

```yaml
downstream_expert:
    src_lang: "other source language"
    tgt_lang: "other target language"
    taskrc:
        data: "other data directory"
```

6. Details about preprocessing

    * For the text data, we do the following preprocessing
        * transcript: lowercasing, removing punctuations execpt apostrophe and hyphen
        * translation: normalizing punctuation
        * the normalization is done with [alvations/sacremoses](https://github.com/alvations/sacremoses)
    
    * We also remove the noise examples by length and ratio of transcript and translation, also all the examples contains "REMOVE".

    * For the tokenization, we create char dictionary with [google/sentencepiece](https://github.com/google/sentencepiece) for transcript and translation seperately.

    * For more details, you can check the files under `prepare_data/`.

#### Training

```bash
python3 run_downstream.py -n ExpName -m train -u fbank -d speech_translation
```

* For downstream model architecture, we delegate the configuration and creation to [pytorch/fairseq](https://github.com/pytorch/fairseq). You could adjust the model archtecture directly at `downstream_expert/modelrc` in `config.yaml`. (For more configurations, please refer to [pytorch/fairseq](https://github.com/pytorch/fairseq/blob/master/fairseq/models/speech_to_text/s2t_transformer.py))

```yaml
downstream_expert:
    modelrc:
        # you could set the model architecture here
        arch: s2t_transformer
        
        # set other model configurations here
        max_source_positions: 6000
        max_target_positions: 1024
        encoder_layers: 3
        decoder_layers: 3
```

* we will truncate the wav to the maximum input size, which is `max_source_positions`*`upstream_rate`.

* We also support multitask learning with ASR. You could set `downstream_expert/taskrc/use_asr=True` in `config.yaml` to enable it. (Make sure you have transcripts in the training tsv file.)

```yaml
downstream_expert:
    taskrc:
        use_asr: True
    asrrc:
        weight: 0.3 # the weight of ASR loss in [0, 1]
        datarc:
            key: src_text # header of transcript in tsv file
```

* You could downsample the upstream feature to certain upstream rate by setting `downstream_expert/upstream_rate` in `config.yaml` with different `downstream_expert/downsample_method`.

```yaml
downstream_expert:
    upstream_rate: -1 # -1 for no downsample, 320 for applying downsampling
    downsample_method: 'drop' # 'drop'/'concat'/'average'
```

* For other training configurations, including batch size, learning rate, etc, you can also change them in `config.yaml`.

#### Testing

```bash
python3 run_downstream.py -m evaluate -t test -e result/downstream/ExpName/dev-best.ckpt
```

You could change the beam size and maximum decoding length in `config.yaml`.

```yaml
downstream_expert:
    generatorrc:
        beam: 20
        max_len_a: 0
        max_len_b: 400
```

We report case-sensitive detokenized BLEU for ST using [mjpost/sacrebleu](https://github.com/mjpost/sacrebleu) and CER/WER for ASR using [roy-ht/editdistance](https://github.com/roy-ht/editdistance) when using multitask learning.
The decoding results will be written into file `<output_prefix>-[st|asr]-[dev|test].tsv`. You could change the prefix of the output files in `config.yaml`.
```yaml
downstream_expert:
    output_prefix: output # set prefix of output files
```