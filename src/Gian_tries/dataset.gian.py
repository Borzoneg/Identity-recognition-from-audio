import glob
import os
from pathlib import Path
from scipy.io import wavfile

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
# Dataset
from torch.utils.data import DataLoader, Dataset, random_split

import librosa
import librosa.display
import IPython.display as ipd

from tqdm.notebook import tqdm


sliced_dataset = "short_audio_dataset"
sliced_dataset_lenght = 16050
# sliced_dataset = "shorter_audio_dataset"
# sliced_dataset_lenght = 4013
original_dataset = "audio_dataset"
original_dataset_lenght = 80249

class AudioDataset(Dataset):
    def __init__(self, root_path="./data/", drop_both=False, use_short=False, normalize=False, use_features=False):
        root_folder = root_path + original_dataset if not use_short else root_path + sliced_dataset
        self.use_features = use_features
        self.max_length = original_dataset_lenght if not use_short else sliced_dataset_lenght
        self.class_map = {"esben" : 0, "peter": 1, "both": 2}
        self.data = []
        self.wavs = []
        self.labels = []
        self.features_list = []
        self.min_val = 10e10
        self.max_val = 0
        print("Start reading files")
        for subdir, dirs, files in os.walk(root_folder):
            for file_name in files:
                if drop_both and "both" in subdir:
                   continue
        
                file_path = os.path.join(subdir, file_name)
                self.sample_rate, wav = wavfile.read(file_path)
                wav = wav.astype(np.float32)
                
                if wav.shape[0] > self.max_length:
                    self.max_length = wav.shape[0]
                    print("Found wav with more length than specified max one, new max is:", wav.shape[0])
                
                wav = np.pad(wav, (0, self.max_length-wav.shape[0]))
                #feature_type_list = ["mfcc", "chroma", "contrast", "centroid", "bandwidth"]
                feature_type = "mfcc"
                features = np.transpose(self.feature_extraction(wav, self.sample_rate, feature_type, normalize=normalize))
                label_str = file_path.split('/')[-3][2:]
                label = (np.int64(self.class_map[label_str]))
                
                self.max_val = np.max(wav) if np.max(wav) > self.max_val else self.max_val
                self.min_val = np.min(wav) if np.min(wav) < self.min_val else self.min_val
                
                self.wavs.append(wav)
                self.features_list.append(features)
                self.labels.append(label)
               
        self.wavs = np.array(self.wavs)
        self.mu  = self.wavs.mean()
        self.std = np.std(self.wavs)
        # self.wavs = torch.Tensor(self.wavs)
        if normalize:
            self.wavs = (self.wavs + np.abs(self.min_val)) / (np.abs(self.min_val) + self.max_val)
            # self.wavs = torch.nn.functional.normalize(self.wavs, dim=1)
        
        print("="*40)
        print("Loaded DATABASE from {}\n{} total file\nLongest file is {} long\nMean: {}\nStandard deviation: {}\nNormalization: {}".
              format(root_folder, len(self.wavs), self.max_length, self.mu, self.std, normalize))
        print("="*40)
    
    def feature_extraction(self, wav, sample_rate, feature, normalize=False):
        if feature == "mfcc":
            mfcc = np.mean(librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=128).T, axis=0)
            if normalize:
                mfcc = (mfcc + np.abs(np.min(mfcc))) / (np.abs(np.min(mfcc)) + np.max(mfcc))
            chosen_feature = mfcc
        if feature == "chroma":
            chroma = librosa.feature.chroma_stft(y=wav, sr=sample_rate)
            if normalize: #idk if its correct
                chroma = (chroma + np.abs(np.min(chroma))) / (np.abs(np.min(chroma)) + np.max(chroma))
            chosen_feature = chroma
        if feature == "contrast":
            contrast = librosa.feature.spectral_contrast(y=wav, sr=sample_rate)
            if normalize:
                contrast = (contrast + np.abs(np.min(contrast))) / (np.abs(np.min(contrast)) + np.max(contrast))
            chosen_feature = contrast
        if feature == "centroid":
            centroid = librosa.feature.spectral_centroid(y=wav, sr=sample_rate)
            if normalize:
                centroid = (centroid + np.abs(np.min(centroid))) / (np.abs(np.min(centroid)) + np.max(centroid))
            chosen_feature = centroid
        if feature == "bandwidth":
            bandwidth = librosa.feature.spectral_bandwidth(y=wav, sr=sample_rate)
            if normalize:
                bandwidth = (bandwidth + np.abs(np.min(bandwidth))) / (np.abs(np.min(bandwidth)) + np.max(bandwidth))
            chosen_feature = bandwidth
        
        return chosen_feature

    def __len__(self):
        return len(self.wavs)
    
    def __getitem__(self, idx):
        wav = self.wavs[idx]
        label = self.labels[idx]
        wav_tensor = torch.from_numpy(wav)
        # label_tensor = torch.Tensor(label)
        if self.use_features:
            features = self.features_list[idx]
            features_tensor = torch.Tensor(features)
            return features_tensor, label
        return wav_tensor, label
