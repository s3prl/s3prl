# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utils/audio.py ]
#   Synopsis     [ audio processing functions ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference 0  [ https://github.com/andi611/TTS-Tacotron-Pytorch ]
#   Reference 1  [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
#   Reference 2  [ https://groups.google.com/forum/#!msg/librosa/V4Z1HpTKn8Q/1-sMpjxjCSoJ ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import random
import numpy as np
#-------------#
import librosa
from scipy import signal
#-------------#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
from matplotlib.colors import SymLogNorm
#-------------#
import warnings
warnings.filterwarnings("ignore")
# NOTE: there are warnings for MFCC extraction due to librosa's issue


##################
# AUDIO SETTINGS #
##################
sample_rate = 16000
"""
For feature == 'fbank' or 'mfcc'
"""
num_mels = 80 # int, dimension of feature
num_mfcc = 13 # int, number of MFCCs
window_size = 25 # int, window size for FFT (ms)
stride = 10 # int, window stride for FFT
"""
For feature == 'mel' or 'linear'
"""
num_freq = 1025
frame_length_ms = 50
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
hop_length = 250
griffin_lim_iters = 16
power = 1.5 # Power to raise magnitudes to prior to Griffin-Lim


#############################
# SPECTROGRAM UTILS FORWARD #
#############################
def _stft_parameters(sample_rate):
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    return n_fft, hop_length, win_length

def _linear_to_mel(spectrogram, sample_rate):
    _mel_basis = _build_mel_basis(sample_rate)
    return np.dot(_mel_basis, spectrogram)

def _build_mel_basis(sample_rate):
    n_fft = (num_freq - 1) * 2
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels)

def _preemphasis(x):
    return signal.lfilter([1, -preemphasis], [1], x)

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def _stft(y, sr):
    n_fft, hop_length, win_length = _stft_parameters(sr)
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


#############################
# SPECTROGRAM UTILS BACKWARD #
#############################
def _denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def inv_preemphasis(x):
    return signal.lfilter([1], [1, -preemphasis], x)

def _griffin_lim(S, sr):
    """
        librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, sr)
    for _ in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y ,sr)))
        y = _istft(S_complex * angles, sr)
    return y

def _istft(y, sr):
    _, hop_length, win_length = _stft_parameters(sr)
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


###################
# MEL SPECTROGRAM #
###################
"""
Compute the mel-scale spectrogram from the wav.
"""
def melspectrogram(y, sr):
    D = _stft(_preemphasis(y), sr)
    S = _amp_to_db(_linear_to_mel(np.abs(D), sr))
    return _normalize(S)


###############
# SPECTROGRAM #
###############
"""
Compute the linear-scale spectrogram from the wav.
"""
def spectrogram(y, sr):
    D = _stft(_preemphasis(y), sr)
    S = _amp_to_db(np.abs(D)) - ref_level_db
    return _normalize(S)


###################
# INV SPECTROGRAM #
###################
"""
Converts spectrogram to waveform using librosa
"""
def inv_spectrogram(spectrogram, sr=16000):
    S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** power, sr))          # Reconstruct phase


###################
# EXTRACT FEATURE #
###################
# Acoustic Feature Extraction
# Parameters
#     - input file  : str, audio file path
#     - feature     : str, fbank or mfcc
#     - delta       : bool, append delta
#     - delta_delta : bool, append delta delta
#     - cmvn        : bool, apply CMVN on feature
#     - save_feature: str, if given, store feature to the path and return len(feature)
# Return
#     acoustic features with shape (time step, dim)
def extract_feature(input_file, feature='fbank', delta=False, delta_delta=False, cmvn=True, save_feature=None):
    y, sr = librosa.load(input_file, sr=sample_rate)

    if feature == 'fbank':
        ws = int(sr*0.001*window_size)
        st = int(sr*0.001*stride)
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=num_mels,
                                              n_fft=ws, hop_length=st)
        feat = np.log(feat + 1e-6) # log-scaled
    elif feature == 'mfcc':
        ws = int(sr*0.001*window_size)
        st = int(sr*0.001*stride)
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc,
                                    n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rms(y, hop_length=st, frame_length=ws)
    elif feature == 'mel':
        # feat = melspectrogram(y, sr) # deprecated
        n_fft, hop_length, win_length = _stft_parameters(sr)
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=num_mels,
                                              n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        feat = np.log(feat + 1e-6) # log-scaled
    elif feature == 'linear':
        feat = spectrogram(y, sr)
    else:
        raise ValueError('Unsupported Acoustic Feature: ' + feature)

    # Apply delta
    feat = [feat]
    if delta and feature != 'linear':
        feat.append(librosa.feature.delta(feat[0]))

    if delta_delta and feature != 'linear':
        feat.append(librosa.feature.delta(feat[0], order=2))
    feat = np.concatenate(feat, axis=0)
    if feature == 'linear': assert(np.shape(feat)[0] == num_freq)

    if cmvn:
        feat = (feat - feat.mean(axis=1)[:,np.newaxis]) / (feat.std(axis=1)+1e-16)[:,np.newaxis]
    if save_feature is not None:
        tmp = np.swapaxes(feat, 0, 1).astype('float32')
        np.save(save_feature,tmp)
        return len(tmp)
    else:
        return np.swapaxes(feat, 0, 1).astype('float32')


#####################
# SAVE FIG TO NUMPY #
#####################
def _save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data.transpose(2, 0, 1) # (Channel, Height, Width)


#############################
# PLOT SPECTROGRAM TO NUMPY #
#############################
def plot_spectrogram_to_numpy(spectrogram):
    spectrogram = spectrogram.transpose(1, 0)
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = _save_figure_to_numpy(fig)
    plt.close()
    return data


####################
# PLOT SPECTROGRAM #
####################
def plot_spectrogram(spec, path):
    spec = spec.transpose(1, 0) # (seq_len, feature_dim) -> (feature_dim, seq_len)
    plt.gcf().clear()
    plt.figure(figsize=(12, 3))
    plt.imshow(spec, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300, format="png")
    plt.close() 


####################
# PLOT EMBEDDING #
####################
def plot_embedding(spec, path):
    spec = spec.transpose(1, 0) # (seq_len, feature_dim) -> (feature_dim, seq_len)
    plt.gcf().clear()
    plt.figure(figsize=(12, 3))
    plt.pcolormesh(spec, norm=SymLogNorm(linthresh=1e-3))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300, format="png")
    plt.close()


def plot_attention(attn, path):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(attn)
    plt.savefig(path, format='png')
    plt.close()
