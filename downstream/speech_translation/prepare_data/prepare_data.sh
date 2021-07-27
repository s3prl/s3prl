#!/usr/local/bin bash

# const
DATA_SRC_KEY=src_text
DATA_TGT_KEY=tgt_text

usage() {
    echo "Usage: $0 [flag] [tsv_file]"
    echo "  -a, --audio-dir [audio dir] "
    echo "  -r, --data-root [data root] (default: data)"
    echo "  -d, --dataset [dataset] (default: tmp)"
    echo "  -o, --output [name] (default: test.tsv)"
    echo "  -p, --path-key [key] (default: path)"
    echo "  -s, --src-key [key] (default: src_text)"
    echo "  -t, --tgt-key [key] (default: tgt_text)"
    echo "  -S, --src-lang [lang] (default: en)"
    echo "  -T, --tgt-lang [lang] (default: de)"
    echo "  -h, --help print this help message"
    exit 1; 
}

# default argement
audio_dir=
data_root=data
dataset=tmp
out_tsv=test.tsv
path_key=path
src_key=src_text
tgt_key=tgt_text
src_lang=en
tgt_lang=de
help=false


OPTIONS=a:r:d:o:p:s:t:S:T:h
LONGOPTS=audio-dir:,data-root:,dataset:,output:,path-key:,src-key:,tgt-key:,src-lang:,tgt-lang:,help
PARSED=`getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@"`

eval set -- $PARSED

while true; do
  case "$1" in
    -a | --audio-dir ) audio_dir=$2; shift 2 ;;
    -r | --data-root ) data_root=$2; shift 2 ;;
    -d | --dataset ) dataset=$2; shift 2 ;;
    -o | --output ) out_tsv=$2; shift 2 ;;
    -p | --path-key ) path_key=$2; shift 2 ;;
    -s | --src-key ) src_key=$2; shift 2 ;;
    -t | --tgt-key ) tgt_key=$2; shift 2 ;;
    -S | --src-lang ) src_lang=$2; shift 2 ;;
    -T | --tgt-lang ) tgt_lang=$2; shift 2 ;;
    -h | --help ) help=true; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ $help = true ]; then
    usage
    exit 1;
fi

in_tsv=$1

data_dir=${data_root}/${dataset}
log=${data_dir}/prepare_data.log
out_tsv=${data_dir}/${out_tsv}

if [ -f ${out_tsv} ]; then
    echo "file \"${out_tsv}\" exists"
    exit 1
fi

mkdir -p ${data_dir}

echo "[prepare data]" | tee -a $log
python prepare_data.py ${in_tsv} ${out_tsv}.tmp \
    --verbose \
    -p ${path_key} \
    -s ${src_key} \
    -t ${tgt_key} \
    -d ${audio_dir} \
    | tee -a $log

echo "[clean paired corpus]" | tee -a $log
python prepare_clean_paired_corpus.py \
    ${out_tsv}.tmp ${out_tsv}.tmp \
    --verbose \
    --min 1 \
    --ratio 5 \
    --overwrite \
    | tee -a $log

echo "[clean source text]" | tee -a $log
python prepare_normalize_tsv.py \
    ${out_tsv}.tmp ${out_tsv}.tmp ${DATA_SRC_KEY}\
    --verbose \
    --normalize \
    --lowercase \
    --remove-punctuation \
    --lang ${src_lang} \
    --overwrite \
    | tee -a $log

echo "[clean target text]" | tee -a $log
python prepare_normalize_tsv.py \
    ${out_tsv}.tmp ${out_tsv} ${DATA_TGT_KEY}\
    --verbose \
    --normalize \
    --lang ${tgt_lang} \
    --overwrite \
    | tee -a $log

rm -f ${out_tsv}.tmp
