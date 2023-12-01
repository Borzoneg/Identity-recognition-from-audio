import numpy as np
from scipy.io import wavfile
import os
import torch
from torch.utils.data import Dataset, DataLoader

sliced_dataset = "short_audio_dataset"
sliced_dataset_lenght = 16050
original_dataset = "audio_dataset"
original_dataset_lenght = 80249

class AudioDataset(Dataset):
    def __init__(self, root_path="./data/", drop_both=False, use_short=False, normalize=False):
        root_folder = root_path + original_dataset if not use_short else root_path + sliced_dataset
        self.max_length = original_dataset_lenght if not use_short else sliced_dataset_lenght
        self.class_map = {"esben" : 0, "peter": 1, "both": 2}
        self.data = []
        self.wavs = []
        self.labels = []
        min_val = 10e10
        max_val = 0
        print("Start reading files")
        for subdir, dirs, files in os.walk(root_folder):
            for file_name in files:
                if drop_both and "both" in subdir:
                   continue
        
                file_path = os.path.join(subdir, file_name)
                _, wav = wavfile.read(file_path)
                wav = wav.astype(np.float32)
                
                if wav.shape[0] > self.max_length:
                    self.max_length = wav.shape[0]
                    print("Found wav with more length than specified max one, new max is:", wav.shape[0])
                
                wav = np.pad(wav, (0, self.max_length-wav.shape[0]))
                label_str = file_path.split('/')[-3][2:]
                label = (np.int64(self.class_map[label_str]))
                
                self.max_val = max(wav) if max(wav) > max_val else max_val
                self.min_val = min(wav) if min(wav) < min_val else min_val
                
                self.wavs.append(wav)
                self.labels.append(label)

        self.wavs = np.array(self.wavs)
        self.mu  = self.wavs.mean()
        self.std = np.std(self.wavs)
        if normalize:
            print("Normalizin with min: {} and max: {}".format(self.min_val, self.max_val))
            # self.wavs = (self.wavs - self.mu) / self.std
            self.wavs = (self.wavs - np.abs(self.min_val)) / (np.abs(self.max_val) - np.abs(self.min_val))
        print("="*40)
        print("Loaded DATABASE from {}\n{} total file\nLongest file is {} long\nMean: {}\nStandard deviation: {}\nNormalization: {}".
              format(root_folder, len(self.wavs), self.max_length, self.mu, self.std, normalize))
        print("="*40)
    
    def __len__(self):
        return len(self.wavs)
    
    def __getitem__(self, idx):
        wav = self.wavs[idx]
        class_id = self.labels[idx]
        wav_tensor = torch.from_numpy(wav)
        class_id = torch.tensor(class_id)
        return wav_tensor, class_id
