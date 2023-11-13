
import numpy as np
from scipy.io import wavfile
import os
import torch
from torch.utils.data import Dataset, DataLoader

sliced_dataset = "../data/short_audio_dataset"
sliced_dataset_lenght = 16050
original_dataset = "../data/audio_dataset"
original_dataset_lenght = 80249

class AudioDataset(Dataset):
    def __init__(self, drop_both=False):
        root_folder = original_dataset
        max_length = original_dataset_lenght
        self.class_map = {"both": 0, "esben" : 1, "peter": 2}
        self.data = []
        self.labels = []
        for subdir, dirs, files in os.walk(root_folder):
            for file_name in files:
                if "both" in subdir and drop_both:
                   continue
                file_path = os.path.join(subdir, file_name)
                _, wav = wavfile.read(file_path)
                if wav.shape[0] > max_length:
                    max_length = wav.shape[0]
                    print("Found wav with more length than specified max one, new max is:", wav.shape[0])
                wav = np.pad(wav, (0, max_length-wav.shape[0]))
                label = file_path.split('/')[3][2:]
                self.labels.append(label)
                self.data.append(wav)
        print("Max length of wav files:", max_length)
        #self.sample_rate = sample_rate
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        wav = self.data[idx]
        label = self.labels[idx]
        class_id = self.class_map[label]
        wav_tensor = torch.from_numpy(wav)
        class_id = torch.tensor([class_id])
        return wav_tensor, class_id
