#!/usr/local/bin bash

# basic config
lxt_root=/ssd/1.1_distributed
lxt_split=train_5hr
audio_dir=$lxt_root/audio
lxt_script=$lxt_root/normalized_transcription.txt
translation_tsv=/home/sean/Desktop/s3prl/lxt_first_sec.tsv
src_lang=en
tgt_lang=de

# data config
dataset=lxt_${lxt_split}
data_root=../../../data

# key of tsv
id_key=id
path_key=path
src_key=src_text
tgt_key=tgt_text

# const
DATA_SRC_KEY=src_text
DATA_TGT_KEY=tgt_text

mkdir $data_root/$dataset

python combine_script_and_trans.py \
    --transcript-file $lxt_script \
    --translation-tsv $translation_tsv \
    --output-tsv $data_root/$dataset/full.tsv

for split in train dev test; do

    python select_with_id.py \
        --input-tsv $data_root/$dataset/full.tsv \
        --id-key $id_key \
        --id-file $lxt_root/$lxt_split/$split.txt \
        --output-tsv lxt_tmp.tsv \

    bash prepare_data.sh \
        lxt_tmp.tsv \
        --audio-dir $audio_dir \
        --data-root $data_root \
        --dataset $dataset \
        --path-key $path_key \
        --src-key $src_key \
        --tgt-key $tgt_key \
        -S $src_lang -T $tgt_lang \
        --output $split.tsv
    
    rm lxt_tmp.tsv
    
done

python prepare_gen_fairseq_vocab.py \
    ${data_root}/${dataset}/train.tsv \
    --src-key ${DATA_SRC_KEY} \
    --tgt-key ${DATA_TGT_KEY} \
    --output-dir ${data_root}/${dataset} \
    --model-type char

python prepare_create_config.py \
    --sp-model ${data_root}/${dataset}/spm-${DATA_TGT_KEY}.model \
    --vocab-file spm-${DATA_TGT_KEY}.txt \
    --audio-dir $audio_dir \
    --output $data_root/$dataset/config.yaml
