# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ vc_evaluate.py ]
#   Synopsis     [ functions related to objective evaluation for voice conversion ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""


import logging
from fastdtw import fastdtw
import librosa
import numpy as np
import pysptk
import pyworld as pw
import scipy
from scipy.io import wavfile
from scipy.signal import firwin
from scipy.signal import lfilter
from torch._C import ErrorReport

SRCSPKS = ["SEF1", "SEF2", "SEM1", "SEM2"]
TRGSPKS_TASK1 = ["TEF1", "TEF2", "TEM1", "TEM2"]
TRGSPKS_TASK2 = ["TFF1", "TFM1", "TGF1", "TGM1", "TMF1", "TMM1"]

################################################################################

# The follow section is related to the calculation of MCD and F0-related metrics
# Reference: https://github.com/espnet/espnet/blob/master/utils/mcd_calculate.py

MCEP_DIM=39
MCEP_ALPHA=0.466
MCEP_SHIFT=5
MCEP_FFTL=1024

def low_cut_filter(x, fs, cutoff=70):
    """FUNCTION TO APPLY LOW CUT FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x

def spc2npow(spectrogram):
    """Calculate normalized power sequence from spectrogram

    Parameters
    ----------
    spectrogram : array, shape (T, `fftlen / 2 + 1`)
        Array of spectrum envelope

    Return
    ------
    npow : array, shape (`T`, `1`)
        Normalized power sequence

    """

    # frame based processing
    npow = np.apply_along_axis(_spvec2pow, 1, spectrogram)

    meanpow = np.mean(npow)
    npow = 10.0 * np.log10(npow / meanpow)

    return npow

def _spvec2pow(specvec):
    """Convert a spectrum envelope into a power

    Parameters
    ----------
    specvec : vector, shape (`fftlen / 2 + 1`)
        Vector of specturm envelope |H(w)|^2

    Return
    ------
    power : scala,
        Power of a frame

    """

    # set FFT length
    fftl2 = len(specvec) - 1
    fftl = fftl2 * 2

    # specvec is not amplitude spectral |H(w)| but power spectral |H(w)|^2
    power = specvec[0] + specvec[fftl2]
    for k in range(1, fftl2):
        power += 2.0 * specvec[k]
    power /= fftl

    return power

def extfrm(data, npow, power_threshold=-20):
    """Extract frame over the power threshold

    Parameters
    ----------
    data: array, shape (`T`, `dim`)
        Array of input data
    npow : array, shape (`T`)
        Vector of normalized power sequence.
    power_threshold : float, optional
        Value of power threshold [dB]
        Default set to -20

    Returns
    -------
    data: array, shape (`T_ext`, `dim`)
        Remaining data after extracting frame
        `T_ext` <= `T`

    """

    T = data.shape[0]
    if T != len(npow):
        raise("Length of two vectors is different.")

    valid_index = np.where(npow > power_threshold)
    extdata = data[valid_index]
    assert extdata.shape[0] <= T

    return extdata

def world_extract(x, fs, f0min, f0max):
    # scale from [-1, 1] to [-32768, 32767]
    x = x * np.iinfo(np.int16).max
    
    x = np.array(x, dtype=np.float64)
    x = low_cut_filter(x, fs)

    # extract features
    f0, time_axis = pw.harvest(
        x, fs, f0_floor=f0min, f0_ceil=f0max, frame_period=MCEP_SHIFT
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=MCEP_FFTL)
    ap = pw.d4c(x, f0, time_axis, fs, fft_size=MCEP_FFTL)
    mcep = pysptk.sp2mc(sp, MCEP_DIM, MCEP_ALPHA)
    npow = spc2npow(sp)

    return {
        "sp": sp,
        "mcep": mcep,
        "ap": ap,
        "f0": f0,
        "npow": npow,
    }

def calculate_mcd_f0(x, y, fs, f0min, f0max):
    """
    x and y must be in range [-1, 1]
    """

    # extract ground truth and converted features
    gt_feats = world_extract(x, fs, f0min, f0max)
    cvt_feats = world_extract(y, fs, f0min, f0max)

    # VAD & DTW based on power
    gt_mcep_nonsil_pow = extfrm(gt_feats["mcep"], gt_feats["npow"])
    cvt_mcep_nonsil_pow = extfrm(cvt_feats["mcep"], cvt_feats["npow"])
    _, path = fastdtw(cvt_mcep_nonsil_pow, gt_mcep_nonsil_pow, dist=scipy.spatial.distance.euclidean)
    twf_pow = np.array(path).T

    # MCD using power-based DTW
    cvt_mcep_dtw_pow = cvt_mcep_nonsil_pow[twf_pow[0]]
    gt_mcep_dtw_pow = gt_mcep_nonsil_pow[twf_pow[1]]
    diff2sum = np.sum((cvt_mcep_dtw_pow - gt_mcep_dtw_pow) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

    # VAD & DTW based on f0
    gt_nonsil_f0_idx = np.where(gt_feats["f0"] > 0)[0]
    cvt_nonsil_f0_idx = np.where(cvt_feats["f0"] > 0)[0]
    try:
        gt_mcep_nonsil_f0 = gt_feats["mcep"][gt_nonsil_f0_idx]
        cvt_mcep_nonsil_f0 = cvt_feats["mcep"][cvt_nonsil_f0_idx]
        _, path = fastdtw(cvt_mcep_nonsil_f0, gt_mcep_nonsil_f0, dist=scipy.spatial.distance.euclidean)
        twf_f0 = np.array(path).T

        # f0RMSE, f0CORR using f0-based DTW
        cvt_f0_dtw = cvt_feats["f0"][cvt_nonsil_f0_idx][twf_f0[0]]
        gt_f0_dtw = gt_feats["f0"][gt_nonsil_f0_idx][twf_f0[1]]
        f0rmse = np.sqrt(np.mean((cvt_f0_dtw - gt_f0_dtw) ** 2))
        f0corr = scipy.stats.pearsonr(cvt_f0_dtw, gt_f0_dtw)[0]
    except ValueError:
        logging.warning(
            "No nonzero f0 is found. Skip f0rmse f0corr computation and set them to NaN. "
            "This might due to unconverge training. Please tune the training time and hypers."
        )
        f0rmse = np.nan
        f0corr = np.nan

    # DDUR
    # energy-based VAD with librosa
    x_trim, _ = librosa.effects.trim(y=x)
    y_trim, _ = librosa.effects.trim(y=y)
    ddur = float(abs(len(x_trim) - len(y_trim)) / fs)

    return mcd, f0rmse, f0corr, ddur


################################################################################

# The follow section is related to the calculation of ASR-based metrics
# Reference: https://github.com/tzuhsien/Voice-conversion-evaluation/blob/master/metrics/character_error_rate/inference.py 

import editdistance as ed
import jiwer
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
    
ASR_PRETRAINED_MODEL = "facebook/wav2vec2-large-960h-lv60-self"

def load_asr_model(device):
    """Load model"""
    print(f"[INFO]: Load the pre-trained ASR by {ASR_PRETRAINED_MODEL}.")
    model = Wav2Vec2ForCTC.from_pretrained(ASR_PRETRAINED_MODEL).to(device)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(ASR_PRETRAINED_MODEL)
    models = {"model": model, "tokenizer": tokenizer}
    return models


def normalize_sentence(sentence):
    """Normalize sentence"""
    # Convert all characters to upper.
    sentence = sentence.upper()
    # Delete punctuations.
    sentence = jiwer.RemovePunctuation()(sentence)
    # Remove \n, \t, \r, \x0c.
    sentence = jiwer.RemoveWhiteSpace(replace_by_space=True)(sentence)
    # Remove multiple spaces.
    sentence = jiwer.RemoveMultipleSpaces()(sentence)
    # Remove white space in two end of string.
    sentence = jiwer.Strip()(sentence)

    # Convert all characters to upper.
    sentence = sentence.upper()

    return sentence


def calculate_measures(groundtruth, transcription):
    """Calculate character/word measures (hits, subs, inserts, deletes) for one given sentence"""
    groundtruth = normalize_sentence(groundtruth)
    transcription = normalize_sentence(transcription)

    #cer = ed.eval(transcription, groundtruth) / len(groundtruth)
    # c_result = jiwer.compute_measures([c for c in groundtruth if c != " "], [c for c in transcription if c != " "])
    c_result = jiwer.cer(groundtruth, transcription, return_dict=True)
    w_result = jiwer.compute_measures(groundtruth, transcription)

    return c_result, w_result, groundtruth, transcription


def transcribe(model, device, wav):
    """Calculate score on one single waveform"""
    # preparation
    inputs = model["tokenizer"](
        wav, sampling_rate=16000, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # forward
    logits = model["model"](
        input_values, attention_mask=attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = model["tokenizer"].batch_decode(predicted_ids)[0]

    return transcription

################################################################################

# The follow section is related to the calculation of ASV-based metrics
# Reference: https://github.com/tzuhsien/Voice-conversion-evaluation/blob/master/metrics/speaker_verification
# Reference: https://github.com/yistLin/dvector/blob/master/equal_error_rate.pyi

from collections import defaultdict
from itertools import chain
import os

#from utils import find_files, write_hdf5, read_hdf5

from resemblyzer import preprocess_wav, VoiceEncoder
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

def load_asv_model(device):
    model = VoiceEncoder().to(device)
    return model

def get_embedding(wav_path, encoder):
    wav = preprocess_wav(wav_path)
    embedding = encoder.embed_utterance(wav)
    return embedding

def get_cosine_similarity(x_emb, y_emb):
    return np.inner(x_emb, y_emb) / (np.linalg.norm(x_emb) * np.linalg.norm(y_emb))

def generate_sample(embeddings, this_spk, other_spks, label):
    """
    Calculate cosine similarity.
    Generate positive or negative samples with the label.
    """
    this_spk_embs = embeddings[this_spk]
    other_spk_embs = list(chain(*[embeddings[spk] for spk in other_spks]))

    samples = []
    for this_spk_emb in this_spk_embs:
        for other_spk_emb in other_spk_embs:
            cosine_similarity = get_cosine_similarity(this_spk_emb, other_spk_emb)
            samples.append((cosine_similarity, label))

    return samples

def calculate_equal_error_rate(labels, scores):
    """
    labels: (N,1) value: 0,1

    scores: (N,1) value: -1 ~ 1

    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    a = lambda x: 1.0 - x - interp1d(fpr, tpr)(x)
    equal_error_rate = brentq(a, 0.0, 1.0)
    threshold = interp1d(fpr, thresholds)(equal_error_rate)
    return equal_error_rate, threshold

def calculate_threshold(data_root, task, device, query="E3*.wav"):

    if task == "task1":
        spks = SRCSPKS + TRGSPKS_TASK1
    if task == "task2":
        spks = SRCSPKS + TRGSPKS_TASK2
    else:
        raise NotImplementedError
    
    encoder = load_asv_model(device)

    # 1. extract all embeddings
    embeddings = defaultdict(list)
    for spk in spks:
        wav_list = find_files(os.path.join(data_root, spk), query)
        print(f"Extracting spekaer embedding for {len(wav_list)} files of {spk}")
        for wav_path in wav_list:
            embedding = get_embedding(wav_path, encoder)
            embeddings[spk].append(embedding)

    # 2. generate samples
    samples = []
    for spk in spks:
        negative_spks = [_spk for _spk in spks if _spk != spk]
        samples += generate_sample(embeddings, spk, [spk], 1)
        samples += generate_sample(embeddings, spk, negative_spks, 0)
            
    # 3. Calculate EER and threshold
    print(f"[INFO]: Number of samples: {len(samples)}")
    scores = [x[0] for x in samples]
    labels = [x[1] for x in samples]
    equal_error_rate, threshold = calculate_equal_error_rate(labels, scores)

    return float(equal_error_rate), float(threshold) # remember to convert from np.array to float

def calculate_accept(x_path, y_path, encoder, threshold):
    x_emb = get_embedding(x_path, encoder)
    y_emb = get_embedding(y_path, encoder)
    cosine_similarity = get_cosine_similarity(x_emb, y_emb)
    return cosine_similarity > threshold
