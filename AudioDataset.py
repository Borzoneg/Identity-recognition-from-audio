
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
import os
import torch
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, root_folder="./audio_dataset", max_length=80249):        
        self.class_map = {"both": 0, "esben" : 1, "peter": 2}
        self.data = []
        for subdir, dirs, files in os.walk(root_folder):
            for file_name in files:
                file_path = os.path.join(subdir, file_name)
                sample_rate, wav = wavfile.read(file_path)
                if wav.shape[0] > max_length:
                    print("Found wav with more length than specified max one, new max is:", wav.shape[0])
                wav = np.pad(wav, (0, max_length-wav.shape[0]))
                label = file_path.split('/')[2][2:]
                self.data.append([wav, label])
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
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
for wav, label in data_loader:
    print(wav.shape, label.shape)