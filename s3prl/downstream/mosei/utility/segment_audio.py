import os
import pandas as pd
from pydub import AudioSegment
import sys

audio_path_root = sys.argv[1]

audio_path = os.path.join(audio_path_root, "Full/WAV_16000")
label_path = "./utility/CMU_MOSEI_Labels.csv"
out_path = os.path.join(audio_path_root, "Segmented_Audio")

if not os.path.exists(out_path):
    os.makedirs(out_path)
    os.makedirs(os.path.join(out_path, "train"))
    os.makedirs(os.path.join(out_path, "dev"))
    os.makedirs(os.path.join(out_path, "test"))

df = pd.read_csv(label_path)

for row in df.itertuples():
    unsegmented = AudioSegment.from_wav(os.path.join(audio_path, 
		row.file + ".wav"))
    segment = unsegmented[max(0, row.start * 1000) : row.end * 1000]
    if row.split == 0:
        segment.export(
            os.path.join(out_path, 
						 "train/" + row.file + "_" + str(row.index) + ".wav"),
            format="wav",
            bitrate="256k",
        )
    elif row.split == 1:
        segment.export(
            os.path.join(out_path, 
						 "dev/" + row.file + "_" + str(row.index) + ".wav"),
            format="wav",
            bitrate="256k",
        )
    elif row.split == 2:
        segment.export(
            os.path.join(out_path, 
						 "test/" + row.file + "_" + str(row.index) + ".wav"),
            format="wav",
            bitrate="256k",
        )
