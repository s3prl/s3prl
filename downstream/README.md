# Downstream Tasks

Table of Contents
------------------------------------

<!--ts-->
   * [Table of contents](#table-of-contents)
   * [General rule of thumb](#general-rule-of-thumb)
   * [Recognition](#recognition)
      * [(PC) Phone Classification](#pc-phone-classification)
      * [(PR) Phoneme Recognition](#pr-phoneme-recognition)
      * [(ASR) Automatic Speech Recognition](#asr-automatic-speech-recognition)
   * [Detection](#detection)
      * [(KS) Speech Commands](#ks-speech-commands)
      * [(STD) Spoken Term Detection: SWS2013](#std-spoken-term-detection-sws2013)
      * [(QBE) Spoken Term Detection: QUESST14](#qbe-spoken-term-detection-quesst14)
  * [Semantics](#semantics)
      * [(IC) Intent Classification: Fluent Speech Commands](#ic-intent-classification-fluent-speech-commands)
      * [(IC) Intent Classification: Audio SNIPS](#ic-intent-classification-audio-snips)
      * [(IC) Intent Classification: ATIS](#ic-intent-classification-atis)
      * [(SF) Slot Filling: SNIPS](#sf-slot-filling-snips)
      * [(SSA) Spoken Sentiment Analysis](#ssa-spoken-sentiment-analysis)
  * [Speaker](#speaker)
      * [(SID) Speaker Identification](#sid-speaker-identification)
      * [(ASV) Automatic Speaker Verification](#asv-automatic-speaker-verification)
      * [(SD) Speaker Diarization](#sd-speaker-diarization)
  * [Emotion](#emotion)
      * [(EC) Emotion Classification](#ec-emotion-classification)
  * [Adding new downstream tasks](#adding-new-downstream-tasks)
<!--te-->

------------------------------------

## General rule of thumb
* Downstream tasks are to be run with upstream models, check the list of upstream models available [here](https://github.com/s3prl/s3prl/tree/master/upstream#upstream-models), or through the following:
```python
import torch
print(torch.hub.list('s3prl/s3prl'))
```
* The setup of all tasks are very simple, open up the config files upder each downstream directory (`downstream/*/config.yaml`) and download the required data.
* Run training with this command: `python run_downstream.py -m train -u baseline -d example -n NameOfExp`
* Downstream tasks can be specified with the argument `-d`, for example `-d phone_linear`.
* Upstream models can be specified with the argyment `-u`, for example `-u tera`.
* For ASR, install Fairseq (https://github.com/pytorch/fairseq) and Flashlight Python Bindings (https://github.com/facebookresearch/flashlight/tree/master/bindings/python).

[Back to Top](#table-of-contents)

------------------------------------

## Recognition

### (PC) Phone Classification
- Specified with the command `-d` (with different variants): 
    - `phone_linear`
    - `phone_linear_concat` 
    - `phone_1hidden`
- **Prepare data:**
    1) Download the raw [LibriSpeech](https://www.openslr.org/12) corpus and unzip.
    ```bash=
    cd /path/to/put/data
    wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
    tar zxvf train-clean-100.tar.gz
    ```
    2) After extracting the file, you should have the file structure as following: 
    ```bash
    LibriSpeech
    ├── train-clean-100
    └── README.TXT
    ```
    3) unzip phone labels:
    ```bash=
    cd data/cpc_phone
    unzip converted_aligned_phones.zip
    ```
    4)  *(Optional)* Allow bucketing to increase training efficientcy & speed, this will generate a directory called `data/len_for_bucket`:
    ```bash=
    python preprocess/generate_len_for_bucket.py --data_root "your_libri_root" --output_path ./data/
    ```
    5) Change the following paths under `phone_*/config.yaml` to your own:
    ```yaml=
    libri_root: '/media/andi611/1TBSSD/LibriSpeech/'
    bucket_file: 'data/len_for_bucket'
    ```
- **Training:**
    ```bash=
    python run_downstream.py -m train -u baseline -d phone_linear -n MyExpName
    python run_downstream.py -m train -u baseline -d phone_linear_concat -n MyExpName
    python run_downstream.py -m train -u baseline -d phone_1hidden -n MyExpName
    ```
- **Testing:**
Testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt:
    ```bash=
    python utility/get_best_dev.py result/downstream/MyExpName/log.log
    ```
    
[Back to Top](#table-of-contents)

------------------------------------

### (PR) Phoneme Recognition
- Specified with the command: `-d ctc`
- **Prepare data:**
    1) Download the raw [LibriSpeech](https://www.openslr.org/12) corpus and unzip.
    ```bash=
    cd /path/to/put/data
    wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
    tar zxvf train-clean-100.tar.gz
    ```
    2) After extracting the file, you should have the file structure as following: 
    ```bash
    LibriSpeech
    ├── train-clean-100
    └── README.TXT
    ```
    3) Modify the LibriSpeech `path` in: `downstream/ctc/libriphone.yaml`
- **Training:**
    ```bash=
    python run_downstream.py -m train -u wav2vec2 -d ctc --config downstream/ctc/libriphone.yaml -n MyExpName -o "config.optimizer.lr=1.0e-2"
    ```
- **Testing:**
    ```bash=
    expdir=result/downstream/MyExpName;
    python run_downstream.py -m evaluate -e $expdir/dev-best.ckpt > $expdir/test.result
    ```

[Back to Top](#table-of-contents)

------------------------------------

### (ASR) Automatic Speech Recognition
- Specified with the command: `-d ctc`
- **Prepare data:**
    1) Download the raw [LibriSpeech](https://www.openslr.org/12) corpus and unzip.
    ```bash=
    cd /path/to/put/data
    wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
    tar zxvf train-clean-100.tar.gz
    ```
    2) After extracting the file, you should have the file structure as following: 
    ```bash
    LibriSpeech
    ├── train-clean-100
    └── README.TXT
    ```
    3) Prepare the lengths for all utterances in LibriSpeech:
    ```bash=
    python preprocess/generate_len_for_bucket.py -i /path/to/LibriSpeech -o data/librispeech -a .flac --n_jobs 12
    ```
    4) Modify the root path of LibriSpeech in `downstream/asr/config.yaml`
- **Training:**
    ```bash=
    distributed="-m torch.distributed.launch --nproc_per_node 16";
    python $distributed run_downstream.py -m train -u baseline -d asr -c downstream/asr/config.yaml -n TrainExpName
    ```
- **Testing:**
    ```bash=
    python run_downstream.py -m evaluate -t "test-clean" -u baseline -d asr -c downstream/asr/config.yaml -i /path/to/ckpt -n TestExpName
    ```

**<details><summary>(Click to expand) Useful links for preparing decoding environmnet</summary><p>**

- Install wav2letter python bindings (not the entire wav2letter is needed to be installed):
    - https://github.com/facebookresearch/wav2letter/wiki/Building-Python-bindings
    - When installing KenLM, please follow the [official instruction](https://github.com/kpu/kenlm). This is a [known issue](https://github.com/facebookresearch/wav2letter/issues/875).
    - If you encounter issue when installing KenLM, you might need to install some [extra dependencies](https://medium.com/tekraze/install-kenlm-binaries-on-ubuntu-language-model-inference-tool-33507000f33).
- LM on fairseq:
    - https://github.com/facebookresearch/wav2letter/tree/v0.2/recipes/models/sota/2019
- LibriSpeech Official LM:
    - https://www.openslr.org/resources/11/4-gram.arpa.gz
- Lexicon:
    - https://dl.fbaipublicfiles.com/fairseq/wav2vec/librispeech_lexicon.lst
</p></details>

[Back to Top](#table-of-contents)

------------------------------------

## Detection

### (KS) Speech Commands

- Specified with the command: `-d speech_commands`
- **Prepare data:**
    1) You can use either v0.01 or v0.02, here **we use v0.01**:
        - http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
        - http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
        - http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz
        - http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz
    3) Download and unpack Speech Commands
        ```bash=
        mkdir -p /YOUR_CORPORA_DIR/speech_commands_v0.01
        tar zxf speech_commands_v0.01.tar.gz -C /YOUR_CORPORA_DIR/speech_commands_v0.01
        ```
    3) Download and unpack Speech Commands test set
        ```bash=
        mkdir -p /YOUR_CORPORA_DIR/speech_commands_test_set_v0.01
        tar zxf speech_commands_test_set_v0.01.tar.gz -C /YOUR_CORPORA_DIR/speech_commands_test_set_v0.01
        ```
    4) Modify the following path in `speech_commands/config.yaml`:
        ```yaml
        speech_commands_root: /YOUR_CORPORA_DIR/speech_commands_v0.01
        speech_commands_test_root: /YOUR_CORPORA_DIR/speech_commands_test_set_v0.01
        ```
- **Training:**
    ```bash=
    python3 run_downstream.py -m train -u baseline -d speech_commands -n MyExpName
    ```
- **Testing:**
Testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt:
    ```bash=
    python3 utility/get_best_dev.py result/downstream/MyExpName/log.log
    ```

[Back to Top](#table-of-contents)

------------------------------------

### (STD) Spoken Term Detection: SWS2013
- Specified with the command: `-d sws2013`
- **Prepare data:**
    1) Download the database
        - https://speech.fit.vutbr.cz/files/sws2013Database.tgz
    2) Specify the place to unpack the database
        ```bash=
        export CORPORA_DIR=/YOUR/CORPORA/DIR/PATH
        ```
    3) Unpack the tarball
        ```bash=
        tar zxf sws2013Database.tgz -C $CORPORA_DIR
        ```
    4) Further unpack the scoring script tarball
        ```bash=
        tar zxf $CORPORA_DIR/sws2013Database_dev_eval/scoring_atwv_sws2013_full.tgz -C $CORPORA_DIR/sws2013Database_dev_eval
        ```
    5) Modify the following path in `sws2013/config.yaml` to yours:
        ```yaml
        sws2013_root: /YOUR/CORPORA/DIR/PATH/sws2013Database_dev_eval
        sws2013_scoring_root: /YOUR/CORPORA/DIR/PATH/sws2013Database_dev_eval/scoring_atwv_sws201
        ```
- **Training:**
    ```bash=
    python run_downstream.py -m train -u baseline -d sws2013 -n MyExpName
    ```
- **Testing:**
Testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt:
    ```bash=
    python3 utility/get_best_dev.py result/downstream/MyExpName/log.log
    ```

[Back to Top](#table-of-contents)

------------------------------------

### (QBE) Spoken Term Detection: QUESST14
- Specified with the command: `-d quesst14_dtw`
- **Prepare data:**
    1) Download the database
        - https://speech.fit.vutbr.cz/files/quesst14Database.tgz
    2) Specify the place to unpack the database
        ```bash
        export CORPORA_DIR=/YOUR/CORPORA/DIR/PATH
        ```
    3) Unpack the tarball
        ```bash
        tar zxf quesst14Database.tgz -C $CORPORA_DIR
        ```
    4) Change the following path in `quesst14/config.yaml` to yours
        ```yaml
        dataset_root: /YOUR/CORPORA/DIR/PATH/quesst14Database
        ```
- **Run command:**
    1) **This task don't need training and only need at most 2 hours per model.**
    2) Run the downstream in **evaluate** mode:

    ```bash=
    # The default dist_fn if not specified is "cosine_exp" 
    # as it yields the best result for almost all upstream
    dist_fn=cosine;
    python3 run_downstream.py -m evaluate -t "test" -u baseline \
        -d quesst14_dtw -n baseline_dtw_test \
        -o "config.downstream_expert.dtwrc.dist_method=\"$dist_fn\""
    ```

    3) Score the result using the scoring script:

    ```bash=
    export S3PRL_DIR=/YOUR/S3PRL/PATH
    cd $CORPORA_DIR/quesst14Database/scoring
    ./score-TWV-Cnxe.sh $S3PRL_DIR/result/downstream/baseline_dtw_test \
        groundtruth_quesst14_eval -10
    ```

    4) Since we will test through 4 popular distance functions and also the standard dev/test set. You can script this by:

        ```bash=
        upstream=baseline
        for dist_fn in cosine cityblock euclidean cosine_exp;
        do
            for quesst_split in "test" "dev";
            do
                expdir=result/downstream/qbe/$upstream/$dist_fn/$quesst_split/

                mkdir -p $expdir
                if [ -f $expdir/done ]; then
                    continue
                fi

                python3 run_downstream.py -m evaluate -t $quesst_split \
                 -u $upstream -d quesst14_dtw -p $expdir \
                 -o "config.downstream_expert.dtwrc.dist_method=\"$dist_fn\""

                touch $expdir/done
            done
        done
        ```

[Back to Top](#table-of-contents)

------------------------------------

## Semantics

### (IC) Intent Classification: Fluent Speech Commands
- Specified with the command: `-d fluent_commands`
- **Prepare data:**
    1) Download and unzip data to the path you want:
    ```bash
    cd /path/to/put/data
    wget http://fluent.ai:2052/jf8398hf30f0381738rucj3828chfdnchs.tar.gz
    tar zxvf jf8398hf30f0381738rucj3828chfdnchs.tar.gz
    ```
    2) After extracting the file, you should have the file structure as following: 
    ```bash
    fluent_speech_commands_dataset
    ├── wavs
    │   └── speakers
    ├── data
    │   └── [*.csv]
    ├── readme.md
    └── Fluent Speech Commands Public License.pdf
    ```
    3) Modify the following paths under `fluent_commands/config.yaml` to your own:
    ```yaml
    file_path: /home/raytz/Disk/data/fluent_speech_commands_dataset
    ```
- **Training:**
    ```bash=
    python run_downstream.py -m train -u baseline -d fluent_commands -n MyExpName
    ```
- **Testing:**
    Testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt
    ```bash=
    python utility/get_best_dev.py result/downstream/MyExpName/log.log
    ```

[Back to Top](#table-of-contents)

------------------------------------

### (IC) Intent Classification: Audio SNIPS
- Specified with the command: `-d audio_snips`
- **Prepare data:**
    1) Prepare the Audio file:
    ```bash
    cd /path/to/put/data
    wget https://shangwel-asr-evaluation.s3-us-west-2.amazonaws.com/audio_slu_v3.zip
    unzip audio_slu_v3.zip
    ```
    2) Prepare the NLU annotation file:
    ```bash
    git clone https://github.com/aws-samples/aws-lex-noisy-spoken-language-understanding.git
    cp -r aws-lex-noisy-spoken-language-understanding/* audio_slu
    ```
    3) After extracting the file, you should have the file structure as following: 
    ```bash
    audio_slu
    ├── data
    │   └── nlu_annotation
    │       └── [*.csv]
    ├── license
    ├── audio_Aditi
    ...
    └── audio_Salli
    ```
    4) Change the following paths under `audio_snips/config.yaml` to your own and specify speakers you want in training set and test set:
    ```yaml
    file_path: /home/raytz/Disk/data/audio_slu
    train_speakers: 
      - Aditi
      ...
      - Salli
    test_speakers:
      - Aditi
      ...
      - Salli
    ```
- **Training:**
    ```bash=
    python run_downstream.py -m train -u baseline -d audio_snips -n MyExpName
    ```
- **Testing:**
    Testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt
    ```bash=
    python utility/get_best_dev.py result/downstream/MyExpName/log.log
    ```

[Back to Top](#table-of-contents)

------------------------------------

### (IC) Intent Classification: ATIS
- Specified with the command: `-d atis`
- **Prepare data:**
    1) Prepare the dataset (under the folder of /groups/public):
    ```bash
    //first sftp to the battleship
    lcd /path/to/put/data
    get -r /groups/public/atis
    ```
    2) After downloading the dataset, you should have the file structure as following: 
    ```bash
    atis
    ├── test
    ├── nlu_iob
    ├── train
    ├── dev
    ├── all.trans.txt
    ├── all.iob.trans.txt
    └── slot_vocabs.txt
    ```
    4) Change the following paths under `audio_snips/config.yaml` to your own:
    ```yaml
    file_path: /home/raytz/Disk/data/atis
    ```
- **Training**:
    ```bash=
    python run_downstream.py -m train -u baseline -d atis -n MyExpName
    ```
- **Testing:**
    Testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt
    ```bash=
    python utility/get_best_dev.py result/downstream/MyExpName/log.log
    ```
    
[Back to Top](#table-of-contents)

------------------------------------

### (SF) Slot Filling: SNIPS
- Specified with the command: `-d ctc`
- **Prepare data:**
    1) Prepare env for text normalization
    ```python=
    import nltk
    nltk.download('brown')
    nltk.download('names')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')
    ```
    2) Download data
    ```bash=
    git clone https://github.com/s3prl/s3prl.git
    ./s3prl/preprocess/snips_prepare_data.sh
    ```
    3) Modify the paths of `downstream_expert.corpus.path` and `downstream_expert.corpus.text.slots_file` in **downstream/ctc/snips.yaml**
- **Training:**
    ```bash=
    python run_downstream.py -m train -u baseline -d ctc -c downstream/ctc/snips.yaml -n MyExpName
    ```
- **Testing:**
    ```bash=
    expdir=result/downstream/[MyExpName];
    python run_downstream.py -m evaluate -e $expdir/dev-best.ckpt > $expdir/test.result
    ```

[Back to Top](#table-of-contents)

------------------------------------

### (SSA) Spoken Sentiment Analysis
- Specified with the command: `-d mosei`
- **Prepare data:**
    1) Download and unzip data to the path you want:
    ```bash=
    cd /path/to/put/data
    wget http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip
    unzip CMU_MOSEI.zip
    ```
    2) After extracting the file, you should have the file structure as following. You only need to keep the folder "Audio"
    ```bash
    CMU_MOSEI
    ├── Audio
    │   └── FULL
    ├── Videos
    │   └── ..
    ├── ..
    ```
    3) Change the following paths under `mosei/segment_audio.sh` to your own. 
    ```sh
    python3 ./utility/segment_audio.py **/home/godiclili/Audio**
    ```
    4) Segment Audio by running `mosei/segment_audio.sh`
    ```bash
    bash segment_audio.sh
    ```  
    5) Change the following paths under `mosei/config.yaml` to your own.
    ```yaml
    data_dir: **/home/godiclili/Audio**
    ```
    6) Specify number of classes (2/7, default 2) for classification under `mosei/config.yaml`
    ```yaml
    num_class: 2
    ```
- **Training**:
    ```bash=
    python run_downstream.py -m train -u baseline -d mosei -n MyExpName
    ```
- **Testing:**
    Testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt
    ```bash=
    python utility/get_best_dev.py result/downstream/MyExpName/log.log
    ```

[Back to Top](#table-of-contents)

------------------------------------

## Speaker

### (SID) Speaker Identification
- Specified with the command: `-d voxceleb1`
- **Prepare data:**
    1) Download dataset from [Voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and unzip them.
    2) put dev / test set into one folder (root)
        ```bash=
        Voxceleb1(root)
        ├── dev
        │   └── wav
                └──Speaker id folder
        ├── test
        │   └── wav
                └──Speaker id folder
        ```
    3) Modify the following paths under `./downstream/voxceleb1/config.yaml` to your own:
        ```yaml
          downstream_expert:
            datarc:
              file_path: /path/to/VoxCeleb1    
        ```
- **Training:**
    ```bash=
       python run_downstream.py -m train -d voxceleb1 -u baseline -n MyExpName
    ```
- **Testing:**
    Testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt
    ```bash=
    python utility/get_best_dev.py result/downstream/MyExpName/log.log
    ```

[Back to Top](#table-of-contents)

------------------------------------

### (ASV) Automatic Speaker Verification
- Specified with the command: `-d sv_voxceleb1`
- **Prepare data:**
    1) Download dataset from [Voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and unzip them.
    2) put dev / test set into one folder (root)
        ```bash=
        Voxceleb1(root)
        ├── dev
        │   └── wav
                └──Speaker id folder
        ├── test
        │   └── wav
                └──Speaker id folder
        ```
    3) Modify the following paths under `./downstream/voxceleb1/config.yaml` to your own:
        ```yaml
          downstream_expert:
            datarc:
              file_path: /path/to/VoxCeleb1    
        ```
- **Training**:
```bash=
python run_downstream.py -m train -u baseline -d sv_voxceleb1 -n MyExpName
```
- **Testing:**
    Testing stage of ASV takes a really long time, as we have to evaluate 10 checkpoints per upstream (there is not official validation set) and each of them need time (this depends on the speed of upstream inference, for **fbank** only 15 mins but for **pase** is 1.5 hour).

    Hence, we opt to save all checkpoints and test them parallely with another GPU during training. The following command will run a for-loop to monitor if any new checkpoints is saved, and evaluate it if any is found. The already evaluated checkpoints will be passed as they have the result loggings under their **expdir**.

    ```bash=
    ./run_while.sh "./downstream/sv_voxceleb1/test_expdirs.sh result/downstream/TrainExpName; sleep 1800;"
    ```
    - **Report numbers:**

    The lowest number should be reported, which should be at the bottom.

    ```bash=
    ./downstream/sv_voxceleb1/report.sh result/downstream/MyExpName
    ```

    - **Creating your own dev set:** (optional, not used in the paper)

    You also can generate your own dev set by running the code under the `./downstream/sv_voxceleb1/preprocess.py`

    ```bash=
    python ./downstream/sv_voxceleb1/preprocess.py \
    -r root_dir/dev/wav \
    -s speaker_num \ 
    -p sample_pair \ 
    -s seed

    # (select s speaker)
    # (generate p sample pair from s speaker)
    ```

[Back to Top](#table-of-contents)

------------------------------------

### (SD) Speaker Diarization
- Specified with the command: `-d diarization`
- Required packages:
    - soundfile
    - pysndfx
- **Prepare data:**
    1) Simulate Libri2Mix Data for Diarization:
    ```bash=
    git clone https://github.com/ftshijt/LibriMix.git
    cd LibriMix 
    ./generate_librimix.sh storage_dir
    python scripts/prepare_diarization.py --target_dir ../downstream/diarization/data
    ```
- **Training:**
    ```bash=
    python run_downstream.py -m train -c ./downstream/diarization/config.yaml -d diarization -u baseline -n MyExpName
    ```
- **Testing:**
```bash=
python run_downstream.py -m evaluate -t test -e result/downstream/MyExpName/best-states-dev.ckpt
```  
- Scoring with dscore (need pre-installzation of [dscore](https://github.com/ftshijt/dscore)):

    Change the dscore_dir (line 13) to your root directory of the cloned dscore in `downstream/diarization/score.sh` and then run:

    ```bash=
    ./downstream/diarization/score.sh result/downstream/libri2mix_diar downstream/diarization/data/test
    ```

    The scoring results will look like:

    ![](https://i.imgur.com/GnVlFlH.png)

    One should report the lowest number at the bottom, where the column represents DER and the most bottom row will always have the lowest DER which is the number we will report.

- **Re-check the scoring results**
    Running the above scoring script takes time. If you just want to re-check the scored results, use

    ```bash=
    ./downstream/diarization/report.sh result/downstream/libri2mix_diar
    ```

[Back to Top](#table-of-contents)

------------------------------------

## Emotion

### (EC) Emotion Classification
- Specified with the command: `-d emotion`
- **Prepare data:**
    1) Download data
    ```cd /path/to/put/data ```
Download dataset from https://sail.usc.edu/iemocap/:
    2) Preprocess by 
    ```bash=
    python IEMOCAP_preprocess.py /path/to/put/data/IEMOCAP
    ```
    3) Modify the root path of IEMOCAP in: `downstream/emotion/config.yaml`
- **Training:**
    ```bash=
    python run_downstream.py -m train -u baseline -d emotion -v fold1 -n MyExpNameFold1
    ```
    - There are five folds, namely: `-v fold1`, `-v fold2`, `-v fold3`, `-v fold4`, and `-v fold5`.
- **Testing:**
    Testing is done on-the-fly with training since it is not costly. Use the following command to get the testing result from the best-dev ckpt
    ```bash=
    python utility/get_best_dev.py result/downstream/MyExpNameFold1/log.log
    ```
- **Note**
    5-fold cross validation is required for the standard evaluation of this dataset. We will first explore the best lr, and then run 5-fold exps on the best lr. The final ACC is averaged over 5 folds. If `-v` option is not assigned durining training it is default to **fold 1**. Available options for `-v` on this task are: **fold 1**, **fold 2**, **fold 3**, **fold 4**, **fold 5**. So after exploring learning rate you will need to:
    ```bash=
    for test_fold in fold1 fold2 fold3 fold4 fold5;
    do
        python3 run_downstream.py -m train -u apc -d emotion \
            -v $test_fold -n emotion_lr1e-4_$test_fold
        python3 utility/get_best_dev.py \
            result/downstream/emotion_lr1e-4_$test_fold/log.log
    done
    ```

[Back to Top](#table-of-contents)

------------------------------------

## Adding new downstream tasks
* Please see this slide for a detailed tutorial: [link](https://docs.google.com/presentation/d/1QRau3NyuHM6KXa8j6Jnw_deFH6Y7s9_kO-_VjeS5LZs/edit?usp=sharing)

[Back to Top](#table-of-contents)

------------------------------------
