from random import shuffle 
import os
from glob import glob
import shutil
import re
import tqdm

def sox_mp3_to_wav(in_root, out_root):
    
    os.makedirs(out_root, exist_ok=True)
    for root, dirs, files in os.walk(in_root):
            print('[Processing] enter directory %s'%root)
            if not len(files):
                continue
            speaker = root.split('/')[-2].split('_')[1]
            print('[Processing] process %d audio files from speaker %s'%(len(files), speaker))
            for name in tqdm.tqdm(files):
                if name.endswith(".mp3"):
                    split = name.split('-')[1]
                    out_dir = os.path.join(out_root, split)
                    os.makedirs(out_dir, exist_ok=True)
                    orig_file = os.path.join(root, name)
                    new_file  = os.path.join(out_dir, speaker+'-'+name.split('/')[-1].split('.')[0] + '.wav')
                    bashCommand = "sox " + orig_file + " -t wav -c 1 -r 16000 -b 16 -e signed-integer " + new_file
                    r = os.popen(bashCommand).read()


if __name__ == '__main__':

    import sys
    audio_dir = sys.argv[1]
    dump_dir = sys.argv[2]
    # Step: sox the snips *.mp3 to the correct format 
    sox_mp3_to_wav(audio_dir, dump_dir)
