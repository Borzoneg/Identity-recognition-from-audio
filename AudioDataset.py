
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
import os
import torch
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, root_folder="./audio_dataset", read_from_file=False):        
        self.class_map = {"both": 0, "esben" : 1, "peter": 2}
        self.data = []
        for subdir, dirs, files in os.walk('./audio_dataset'):
            for file_name in files:
                file_path = os.path.join(subdir, file_name)
                sample_rate, data = wavfile.read(file_path)
                label = file_path.split('/')[2][2:]
                self.data.append([data, label])
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        wav, label = self.data[idx]
        class_id = self.class_map[label]
        wav_tensor = torch.from_numpy(wav)
        class_id = torch.tensor([class_id])
        return wav_tensor, class_id

dataset = AudioDataset()

for wav, label in dataset:
    print(wav, label)