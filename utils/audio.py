# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ utils/audio.py ]
#   Synopsis     [ audio processing functions ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference 1  [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
#   Reference 2  [ https://groups.google.com/forum/#!msg/librosa/V4Z1HpTKn8Q/1-sMpjxjCSoJ ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import librosa
import numpy as np
import matplotlib
import matplotlib.pylab as plt
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")
# NOTE: there are warnings for MFCC extraction due to librosa's issue


###################
# EXTRACT FEATURE #
###################
# Acoustic Feature Extraction
# Parameters
#     - input file  : str, audio file path
#     - feature     : str, fbank or mfcc
#     - dim         : int, dimension of feature
#     - cmvn        : bool, apply CMVN on feature
#     - window_size : int, window size for FFT (ms)
#     - stride      : int, window stride for FFT
#     - save_feature: str, if given, store feature to the path and return len(feature)
# Return
#     acoustic features with shape (time step, dim)
def extract_feature(input_file,feature='fbank',dim=40, cmvn=True, delta=False, delta_delta=False,
					window_size=25, stride=10,save_feature=None):
	y, sr = librosa.load(input_file,sr=None)
	ws = int(sr*0.001*window_size)
	st = int(sr*0.001*stride)
	if feature == 'fbank': # log-scaled
		feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=dim,
									n_fft=ws, hop_length=st)
		feat = np.log(feat+1e-6)
	elif feature == 'mfcc':
		feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim, n_mels=26,
									n_fft=ws, hop_length=st)
		feat[0] = librosa.feature.rmse(y, hop_length=st, frame_length=ws) 
		
	else:
		raise ValueError('Unsupported Acoustic Feature: '+feature)

	feat = [feat]
	if delta:
		feat.append(librosa.feature.delta(feat[0]))

	if delta_delta:
		feat.append(librosa.feature.delta(feat[0],order=2))
	feat = np.concatenate(feat,axis=0)
	if cmvn:
		feat = (feat - feat.mean(axis=1)[:,np.newaxis]) / (feat.std(axis=1)+1e-16)[:,np.newaxis]
	if save_feature is not None:
		tmp = np.swapaxes(feat,0,1).astype('float32')
		np.save(save_feature,tmp)
		return len(tmp)
	else:
		return np.swapaxes(feat,0,1).astype('float32')


def plot_spectrogram_to_numpy(spectrogram):
	
	def save_figure_to_numpy(fig):
		# save it to a numpy array.
		data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
		return data.transpose(2, 0, 1)
	
	fig, ax = plt.subplots(figsize=(12, 3))
	im = ax.imshow(spectrogram, aspect="auto", origin="lower",
				   interpolation='none')
	plt.colorbar(im, ax=ax)
	plt.xlabel("Frames")
	plt.ylabel("Channels")
	plt.tight_layout()

	fig.canvas.draw()
	data = save_figure_to_numpy(fig)
	plt.close()
	return data