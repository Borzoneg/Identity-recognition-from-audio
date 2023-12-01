import numpy as np
from scipy.io import wavfile
import os
import torch
from torch.utils.data import DataLoader
from audio_dataset import AudioDataset
import matplotlib.pyplot as plt

audio_dataset = AudioDataset(root_path="data/", drop_both=True, use_short=False, normalize=True)
data_len = len(audio_dataset)
train_size, test_size, valid_size = int(data_len * 0.8), int(data_len * 0.1), int(data_len * 0.1)

data_train, data_test, data_valid = torch.utils.data.random_split(audio_dataset, (train_size, test_size, valid_size))

kwargs = {'batch_size': 5, 'num_workers': 2}
loader_train = torch.utils.data.DataLoader(data_train, **kwargs, shuffle=True)
loader_test = torch.utils.data.DataLoader(data_test, **kwargs, shuffle=True)
loader_valid = torch.utils.data.DataLoader(data_valid, **kwargs, shuffle=True)

plt.plot(np.arange(audio_dataset.max_length), audio_dataset.wavs[0])
plt.show()