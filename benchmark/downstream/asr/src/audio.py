import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
#from torchaudio.transforms import TimeMasking, FrequencyMasking
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, FrequencyMask, TimeMask
from scipy import signal, ndimage
from librosa import feature
import numpy as np

GRIFFIN_LIM_ITER = 50
SAMPLE_RATE = 16000

class CMVN(torch.jit.ScriptModule):

    __constants__ = ["mode", "dim", "eps"]

    def __init__(self, mode="global", dim=2, eps=1e-10):
        # `torchaudio.load()` loads audio with shape [channel, feature_dim, time]
        # so perform normalization on dim=2 by default
        super(CMVN, self).__init__()

        if mode != "global":
            raise NotImplementedError(
                "Only support global mean variance normalization.")

        self.mode = mode
        self.dim = dim
        self.eps = eps

    @torch.jit.script_method
    def forward(self, x):
        if self.mode == "global":
            return (x - x.mean(self.dim, keepdim=True)) / (self.eps + x.std(self.dim, keepdim=True))

    def extra_repr(self):
        return "mode={}, dim={}, eps={}".format(self.mode, self.dim, self.eps)


class Delta(torch.jit.ScriptModule):

    __constants__ = ["order", "window_size", "padding", "device", "batch"]

    def __init__(self, order=2, window_size=2, device=None, batch=False):
        # Reference:
        # https://kaldi-asr.org/doc/feature-functions_8cc_source.html
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_audio.py
        super(Delta, self).__init__()

        self.order = order
        self.window_size = window_size
        self.device = device if device is not None else 'cpu'
        self.batch = batch

        filters = self._create_filters(order, window_size).to(self.device)
        self.register_buffer("filters", filters)
        self.padding = (0, (filters.shape[-1] - 1) // 2)

    @torch.jit.script_method
    def forward(self, x):
        # Unsqueeze batch dim
        if self.batch:
            x = x.unsqueeze(1)
            # tmp = F.conv2d(x, weight=self.filters, padding=self.padding)
            # print(tmp.shape)
            # return tmp
            return F.conv2d(x, weight=self.filters, padding=self.padding)
        else:
            x = x.unsqueeze(0) # 1 x 1 x T x MEL
            return F.conv2d(x, weight=self.filters, padding=self.padding)[0]

    # TODO(WindQAQ): find more elegant way to create `scales`
    def _create_filters(self, order, window_size):
        scales = [[1.0]]
        for i in range(1, order + 1):
            prev_offset = (len(scales[i-1]) - 1) // 2
            curr_offset = prev_offset + window_size

            curr = [0] * (len(scales[i-1]) + 2 * window_size)
            normalizer = 0.0
            for j in range(-window_size, window_size + 1):
                normalizer += j * j
                for k in range(-prev_offset, prev_offset + 1):
                    curr[j+k+curr_offset] += (j * scales[i-1][k+prev_offset])
            curr = [x / normalizer for x in curr]
            scales.append(curr)

        max_len = len(scales[-1])
        for i, scale in enumerate(scales[:-1]):
            padding = (max_len - len(scale)) // 2
            scales[i] = [0] * padding + scale + [0] * padding

        return torch.tensor(scales).unsqueeze(1).unsqueeze(1)

    def extra_repr(self):
        return "order={}, window_size={}".format(self.order, self.window_size)


class Postprocess(torch.jit.ScriptModule):

    __constants__ = ["detach", "batch"]

    def __init__(self, detach=True, batch=False):
        super(Postprocess, self).__init__()
        self.detach = detach
        self.batch  = batch

    @torch.jit.script_method
    def forward(self, x):
        if self.batch:
            # [batch, channel, feature_dim, time] -> [batch, time, channel, feature_dim]
            x = x.permute(0, 2, 3, 1)
        else:
            # [channel, feature_dim, time] -> [time, channel, feature_dim]
            x = x.permute(2, 0, 1)
        if self.detach:
            # [time, channel, feature_dim] -> [time, feature_dim * channel]
            return x.reshape(x.size(0), -1).detach()
        else:
            # [batch, time, channel, feature_dim] -> [time, feature_dim * channel]
            return x.reshape(x.shape[0], x.shape[1], -1)

# TODO(Windqaq): make this scriptable
class ExtractAudioFeature(nn.Module):
    def __init__(self, mode, num_mel_bins, frame_length, frame_shift, ref_level_db, 
                 min_level_db, preemphasis_coeff, sample_rate=16000):
        super(ExtractAudioFeature, self).__init__()
        self.mode = mode # which feature type
        self.sr = sample_rate
        self.n_fft = 1025
        self.window = None
        # Wave 2 spec
        self.hop_length = int(frame_shift  / 1000 * sample_rate)
        self.win_length = int(frame_length / 1000 * sample_rate)
        self.to_specgram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, 
            win_length=self.win_length, 
            hop_length=self.hop_length,
            # [NOTICE] 
            # What we want is power=1, but this is a HACK for the bug of torchaudio's spectrogram (power=1)
            power=2 
        )
        self.num_mel_bins = num_mel_bins
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db
        self.preemphasis_coeff = preemphasis_coeff

        # HACK : replace torch audios filter bank
        self.to_melspecgram = torchaudio.transforms.MelScale(
            n_mels=self.num_mel_bins, 
            sample_rate=sample_rate
        )
        _mel_basis = create_mel_filterbank(self.sr, self.n_fft, n_mels=self.num_mel_bins).T
        _mel_basis = torch.from_numpy(_mel_basis)
        self.to_melspecgram.fb.resize_(_mel_basis.size())
        self.to_melspecgram.fb.copy_(_mel_basis)

    def forward(self, waveform, channel=0):
        # waveform: B x T = 1 x T
        with torch.no_grad():
            # Use pre-emphasis
            waveform = self._preemphasis(waveform)
            # sqrt(): HACK for the bug of torchaudio's spectrogram (power=1)
            specgram = self.to_specgram(waveform).sqrt() # CH x FREQ x T
            melspecgram = self.to_melspecgram(specgram) # CH x MEL x T
            specgram = self._amp_to_db(specgram) - self.ref_level_db 
            specgram = self._normalize(specgram)
            melspecgram = self._amp_to_db(melspecgram) - self.ref_level_db
            melspecgram = self._normalize(melspecgram)
            msp = melspecgram[channel] # T x MEL
        return msp.unsqueeze(0) # 1 x T x MEL

    def extra_repr(self):
        return "mode={}, num_mel_bins={}".format(self.mode, self.num_mel_bins)

    #------ New functions I added -------#
    def feat_to_wave(self, mel_spec):
        # Convert sigle mel-spec (shape TxD) to waveform
        mel_spec = mel_spec.transpose(0, 1)
        spec = self.melspecgram_to_specgram(mel_spec)
        # Spectrogram -> wave
        wave = self.specgram_to_waveform(spec)
        return wave, self.sr

    def melspecgram_to_specgram(self, melspecgram):
        """
        Arg:
            melspecgram: torch.Tensor of shape (freq[mel], time)
        Return:
            approximate spectrogram: numpy array of shape (freq[spectrogram], time)
        """
        # (freq[mel], )
        fb_pinv = torch.pinverse(self.to_melspecgram.fb).transpose(0, 1)
        melspecgram = self._db_to_amp(self._denormalize(melspecgram) + self.ref_level_db)
        specgram = torch.matmul(fb_pinv, melspecgram)
        return specgram

    def specgram_to_waveform(self, specgram, power=1.0, inv_preemphasis=True):
        """
        Arg:
            specgram: torch.Tensor of shape (freq, time)
        Return:
            approximate waveform: numpy array of shape (samples)
        """
        wav = self._griffin_lim(specgram).detach().cpu().numpy()
        if inv_preemphasis:
            wav = self._inv_preemphasis(wav)
        return np.clip(wav, -1, 1)

    def _griffin_lim(self, specgram):
        """
        Arg:
            specgram: torch.Tensor of shape (freq, time)
        Return:
            approximate waveform of shape (samples)
        """
        phases = np.angle(np.exp(2j * np.pi * np.random.rand(*specgram.shape)))
        phases = phases.astype(np.float32)
        phases = torch.from_numpy(phases)
        magnitude = specgram.abs()
        # Spectrum with random phases
        y = self._to_complex(magnitude, phases)
        x = self._istft(y)
        for _ in range(30): # fixed 30 iter
            y = self._stft(x)
            phases = self._get_phase(y)
            y = self._to_complex(magnitude, phases)
            x = self._istft(y)
        return x

    def _preemphasis(self, waveform):
        waveform = torch.cat([
            waveform[:, :1], 
            waveform[:, 1:] - self.preemphasis_coeff * waveform[:, :-1]], dim=-1)
        return waveform 
    def _inv_preemphasis(self, wav):
        """Note this is implemented in 'scipy' but not 'torch'!!"""
        return signal.lfilter([1], [1, -self.preemphasis_coeff], wav)
    def _amp_to_db(self, x, minimum=1e-5):
        return 20 * torch.log10(torch.clamp(x, min=minimum))
    def _db_to_amp(self, x):
        return 10 ** (0.05 * x)
    def _normalize(self, feat):
        return torch.clamp((feat - self.min_level_db) / -self.min_level_db, min=0, max=1)
    def _denormalize(self, feat):
        return self.min_level_db + torch.clamp(feat, min=0, max=1) * -self.min_level_db
    def _to_complex(self, magnitude, phase):
        """To make a fake complex number in torch"""
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        complx = torch.stack([real, imag], dim=-1)
        return complx
    def _get_phase(self, complx):
        return torchaudio.functional.angle(complx)
    def _stft(self, x):
        # `x` for time-domain signal and `y` for frequency-domain signal
        y = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=self.window, 
            center=True, 
            pad_mode='reflect', 
            normalized=False, 
            onesided=True)
        return y
    
    def _istft(self, y):
        # `x` for time-domain signal and `y` for frequency-domain signal
        x = torchaudio.functional.istft(
            y, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=self.window,
            center=True, 
            pad_mode='reflect', 
            normalized=False, 
            onesided=True)
        return x

class ReadAudio(nn.Module):
    # Read audio files and downsample to specified sample rate
    def __init__(self, desired_sr, mode, time_aug=False):
        super(ReadAudio, self).__init__()
        self.desired_sr = desired_sr
        self.augmenter = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        #Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ])
        self.mode = mode
        self.time_aug = time_aug

    def forward(self, filepath):
        if type(filepath) is not str:
            return filepath
        
        waveform, sample_rate = librosa.load(filepath)
        
        '''if want to use time domain augmentation, do it just after reading raw waveform'''
        if self.mode=="train" and self.time_aug:
            waveform = self.augmenter(samples=waveform, sample_rate=self.desired_sr)
            
        waveform = torch.tensor(waveform.reshape(1, len(waveform)))

        return waveform

#################################################
# ## THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
# ################################################
# file to edit: dev_nb/SpecAugment.ipynb

from collections import namedtuple
import random
from torchaudio import transforms
#from nb_SparseImageWarp import sparse_image_warp
import matplotlib.pyplot as plt

#from IPython.display import Audio
import librosa.util
#from torchaudio.transforms import freq_mask, time_mask, time_warp
'''
Time domain augmentataion
'''
import numpy as np
class Augment_Time(nn.Module):
    def __init__(self):
        super(Augment_Time, self).__init__()
        self.p = 0.5
        self.augmenter = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        FrequencyMask(),
        TimeMask()
        #Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ])
    def forward(self, waveform):
        SAMPLE_RATE = 16000
        samples = self.augmenter(samples=waveform.numpy(), sample_rate=SAMPLE_RATE)
        
        return torch.tensor(samples)






'''
new Augment function
'''
class Augment(nn.Module):
    def __init__(self, T=40, num_masks=1, replace_with_zero=False, F=27):#ori: T = 40
        super(Augment, self).__init__()    
        self.T=T
        self.num_masks=num_masks
        self.replace_with_zero=replace_with_zero
        self.F=F
        self.spec=None
    #@torch.jit.script_method
    def forward(self, spec):
        spec = spec.permute(1, 0)

        spec = self.time_mask(spec, T=self.T, num_masks=self.num_masks, replace_with_zero=self.replace_with_zero)
        spec = self.freq_mask(spec, F=self.F, num_masks=self.num_masks, replace_with_zero=self.replace_with_zero)
        spec = spec.permute(1, 0)
        
        return spec
    def normalize(self, spec):
        spec = (spec-spec.mean())/spec.std()
        return spec        

    def time_mask(self, spec, T=100, num_masks=1, replace_with_zero=False):
        cloned = spec
        len_spectro = cloned.shape[1]
        
        for i in range(0, num_masks):
            t = torch.randint(0, self.T, (1,)).item()
            t_zero = torch.randint(0, len_spectro-t, (1,)).item()
            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned
            mask_end = torch.randint(t_zero, t_zero+t, (1, )).item()

            if (replace_with_zero): cloned[:,t_zero:mask_end] = 0
            else: cloned[:,t_zero:mask_end] = cloned.mean()
        return cloned

    def freq_mask(self, spec, F=27, num_masks=1, replace_with_zero=False):
        cloned = spec
        num_mel_channels = cloned.shape[0]

        for i in range(0, num_masks):
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f)
            if (replace_with_zero): cloned[f_zero:mask_end, :] = 0
            else: cloned[f_zero:mask_end, :] = cloned.mean()

        return cloned

    def tensor_to_img(self, spectrogram, filename):
        spectrogram = self.to_db_scale(spectrogram)
        spectrogram = spectrogram.detach().numpy()
        #print(spectrogram[0])
        plt.figure(figsize=(14,1)) # arbitrary, looks good on my screen.
        plt.imshow(spectrogram)
        plt.savefig(filename)
        #plt.show()
        #display(spectrogram.shape)
    def to_db_scale(self, spectrogram):
        db_sp = 20*torch.log10(spectrogram)
        return db_sp

    def time_warp(self, spec, W=5):
        num_rows = spec.shape[1]
        spec_len = spec.shape[2]
        device = spec.device

        y = num_rows//2
        horizontal_line_at_ctr = spec[0][y]
        assert len(horizontal_line_at_ctr) == spec_len

        point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
        assert isinstance(point_to_warp, torch.Tensor)

        # Uniform distribution from (0,W) with chance to be up to W negative
        dist_to_warp = random.randrange(-W, W)
        src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device),
                            torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
        warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
        return warped_spectro.squeeze(3)
""
def pop_audio_config(audio_config):
    # Delta
    delta_order = audio_config.pop("delta_order", 0)
    delta_window_size = audio_config.pop("delta_window_size", 2)
    apply_cmvn = audio_config.pop("apply_cmvn")

    # Extract Feature
    feat_type = audio_config.pop("feat_type")
    feat_dim = audio_config.pop("feat_dim")

    return audio_config, feat_type, feat_dim


def create_transform(audio_config, post_process=True, mode='train'):
    # Delta
    delta_order = audio_config.pop("delta_order", 0)
    delta_window_size = audio_config.pop("delta_window_size", 2)
    apply_cmvn = audio_config.pop("apply_cmvn", False)

    # Extract Feature
    feat_type = audio_config.pop("feat_type")
    
    feat_dim = audio_config.pop("feat_dim")
    '''specaug'''
    augment = audio_config.pop("augment")
    '''time domain augment'''
    time_aug = audio_config.pop("time_aug")

    transforms = [ReadAudio(SAMPLE_RATE, mode, time_aug)]


    transforms.append(ExtractAudioFeature(mode=feat_type, num_mel_bins=feat_dim, sample_rate=SAMPLE_RATE, **audio_config))

    if delta_order >= 1:
        transforms.append(Delta(delta_order, delta_window_size))

    if apply_cmvn:
        transforms.append(CMVN())
    
    if post_process:
        transforms.append(Postprocess())
    if augment and mode=='train':
        transforms.append(Augment())
        

    return nn.Sequential(*transforms), feat_dim * (delta_order + 1) # 80 *2 feature D


# Filters from librosa, you may ignore this

def create_mel_filterbank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
        norm=1, dtype=np.float32):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft     : int > 0 [scalar]
        number of FFT components

    n_mels    : int > 0 [scalar]
        number of Mel bands to generate

    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)

    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`

    htk       : bool [scalar]
        use HTK formula instead of Slaney

    norm : {None, 1, np.inf} [scalar]
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization).  Otherwise, leave all the triangles aiming for
        a peak value of 1.0

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    >>> melfb = librosa.filters.mel(22050, 2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])


    Clip the maximum frequency to 8KHz

    >>> librosa.filters.mel(22050, 2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])


    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(melfb, x_axis='linear')
    >>> plt.ylabel('Mel filter')
    >>> plt.title('Mel filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    >>> plt.show()
    """

    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn('Empty filters detected in mel frequency basis. '
                      'Some channels will produce empty responses. '
                      'Try increasing your sampling rate (and fmax) or '
                      'reducing n_mels.')

    return weights


def fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of `np.fft.fftfreq`

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    n_fft : int > 0 [scalar]
        FFT window size


    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies `(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)`


    Examples
    --------
    >>> librosa.fft_frequencies(sr=22050, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])

    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
                       endpoint=True)


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """Compute an array of acoustic frequencies tuned to the mel scale.

    The mel scale is a quasi-logarithmic function of acoustic frequency
    designed such that perceptually similar pitch intervals (e.g. octaves)
    appear equal in width over the full hearing range.

    Because the definition of the mel scale is conditioned by a finite number
    of subjective psychoaoustical experiments, several implementations coexist
    in the audio signal processing literature [1]_. By default, librosa replicates
    the behavior of the well-established MATLAB Auditory Toolbox of Slaney [2]_.
    According to this default implementation,  the conversion from Hertz to mel is
    linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
    replicates the Hidden Markov Toolkit [3]_ (HTK) according to the following formula:

    `mel = 2595.0 * np.log10(1.0 + f / 700.0).`

    The choice of implementation is determined by the `htk` keyword argument: setting
    `htk=False` leads to the Auditory toolbox implementation, whereas setting it `htk=True`
    leads to the HTK implementation.

    .. [1] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
        In Proc. International Conference on Acoustics, Speech, and Signal Processing
        (ICASSP), vol. 1, pp. 217-220, 1998.

    .. [2] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
        Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.

    .. [3] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., Liu, X.,
        Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
        The HTK book, version 3.4. Cambridge University, March 2009.


    See Also
    --------
    hz_to_mel
    mel_to_hz
    librosa.feature.melspectrogram
    librosa.feature.mfcc


    Parameters
    ----------
    n_mels    : int > 0 [scalar]
        Number of mel bins.

    fmin      : float >= 0 [scalar]
        Minimum frequency (Hz).

    fmax      : float >= 0 [scalar]
        Maximum frequency (Hz).

    htk       : bool
        If True, use HTK formula to convert Hz to mel.
        Otherwise (False), use Slaney's Auditory Toolbox.

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of n_mels frequencies in Hz which are uniformly spaced on the Mel
        axis.

    Examples
    --------
    >>> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])

    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)


def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels

    Examples
    --------
    >>> librosa.hz_to_mel(60)
    0.9
    >>> librosa.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])

    Parameters
    ----------
    frequencies   : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk           : bool
        use HTK formula instead of Slaney

    Returns
    -------
    mels        : number or np.ndarray [shape=(n,)]
        input frequencies in Mels

    See Also
    --------
    mel_to_hz
    """

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies

    Examples
    --------
    >>> librosa.mel_to_hz(3)
    200.

    >>> librosa.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])

    Parameters
    ----------
    mels          : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk           : bool
        use HTK formula instead of Slaney

    Returns
    -------
    frequencies   : np.ndarray [shape=(n,)]
        input mels in Hz

    See Also
    --------
    hz_to_mel
    """

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs
