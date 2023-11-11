import numpy as np
# import tensorflow as tf
from scipy.io import wavfile
import os

root_folder = "./data/audio_dataset"

# create new folders if not there already
for subdir, dirs, files in os.walk(root_folder):
    if not os.path.exists(subdir.replace('audio_dataset', 'short_audio_dataset')):
        os.mkdir(subdir.replace('audio_dataset', 'short_audio_dataset'))
    else:
        for short_subdir, short_dirs, short_files in os.walk(root_folder.replace('audio_dataset', 'short_audio_dataset')):
            for file_name in short_files:
                os.remove(os.path.join(short_subdir, file_name))



for subdir, dirs, files in os.walk(root_folder):
    idx = 0
    for file_name in files:
        file_path = os.path.join(subdir, file_name)
        sample_rate, wav = wavfile.read(file_path)
        duration = wav.shape[0] / sample_rate
        sub_wavs = np.array_split(wav, round(duration))
        for sub_wav in sub_wavs:
            new_subdir = subdir.replace('audio_dataset', 'short_audio_dataset')
            wavfile.write(os.path.join(new_subdir, "{:04d}.wav".format(idx)), sample_rate, sub_wav)
            idx += 1
        