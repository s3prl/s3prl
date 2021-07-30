# We need this to use GPUs inside the container
FROM nvidia/cuda:11.2.2-base
# Using a multi-stage build simplifies the s3prl installation
# TODO: Find a slimmer base image that also "just works"
FROM tiangolo/uvicorn-gunicorn:python3.8


RUN apt-get update --fix-missing && apt-get install -y wget \
    libsndfile1 \
    sox \
    git \
    git-lfs

RUN python -m pip install --upgrade pip
RUN python -m pip --no-cache-dir install fairseq@git+https://github.com//pytorch/fairseq.git@f2146bdc7abf293186de9449bfa2272775e39e1d#egg=fairseq
RUN python -m pip --no-cache-dir install git+https://github.com/huggingface/s3prl.git@huggingface2#egg=s3prl

COPY s3prl/ /app/s3prl
COPY src/ /app/src

# Setup filesystem
RUN mkdir /app/data

# Configure Git
RUN git config --global user.email "lewis@huggingface.co"
RUN git config --global user.name "SUPERB Admin"

# Fine-tune!
ENV upstream_model osanseviero/hubert_base
ENV downstream_task asr


WORKDIR /app/s3prl
# Each task's config.yaml is used to set all the training parameters
# The results of each training run are stored in /app/s3prl/result/downstream/{downstream_task}
# and pushed to the Hugging Face Hub with name {hf_username}/superb-s3prl-{upstream_model}-{downstream_task}-uuid
CMD python run_downstream.py -n ${downstream_task} -m train -u ${upstream_model} -d ${downstream_task} --hub huggingface