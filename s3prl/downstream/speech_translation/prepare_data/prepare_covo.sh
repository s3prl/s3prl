#!/usr/local/bin bash

# basic config
covo_root="root directory of covost (ex. /Drive/cv-corpus-6.1-2020-12-11)"
src_lang=en
tgt_lang=de

tsv_dir=covost_tsv

if [ ! -d covost ]; then
    git clone https://github.com/facebookresearch/covost.git
fi

mkdir $tsv_dir -p

if [ ! -f covost_v2.${src_lang}_${tgt_lang}.tsv.tar.gz ]; then
    wget https://dl.fbaipublicfiles.com/covost/covost_v2.${src_lang}_${tgt_lang}.tsv.tar.gz
fi 

tar -zxvf covost_v2.${src_lang}_${tgt_lang}.tsv.tar.gz -C $tsv_dir

python covost/get_covost_splits.py \
    --version 2 --src-lang $src_lang --tgt-lang $tgt_lang \
    --root $tsv_dir \
    --cv-tsv $covo_root/$src_lang/validated.tsv

# data config
dataset=covost_${src_lang}_${tgt_lang}
data_root=../../../data

# key of tsv
path_key=path
src_key=sentence
tgt_key=translation

# const
DATA_SRC_KEY=src_text
DATA_TGT_KEY=tgt_text

for split in train dev test; do
    bash prepare_data.sh \
        ${tsv_dir}/covost_v2.${src_lang}_${tgt_lang}.$split.tsv \
        --audio-dir ${covo_root}/${src_lang}/clips/ \
        --data-root ${data_root} \
        --dataset $dataset \
        --path-key ${path_key} \
        --src-key ${src_key} \
        --tgt-key ${tgt_key} \
        -S ${src_lang} -T ${tgt_lang} \
        --output $split.tsv
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
    --audio-dir $covo_root/${src_lang}/clips \
    --output $data_root/$dataset/config.yaml
