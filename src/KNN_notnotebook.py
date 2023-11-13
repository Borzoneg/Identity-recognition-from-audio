import time
import numpy as np

# sklearn
from sklearn import neighbors
from sklearn.metrics import confusion_matrix

# visualization
import matplotlib.pyplot as plt
import seaborn

# torch
import torch
from torchvision import datasets, transforms

# Dataset
from audio_dataset import AudioDataset
from torch.utils.data import DataLoader

#Sklearn
from sklearn.model_selection import train_test_split


dataset = AudioDataset()

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for wav, label in dataloader:
    print(wav.shape)
    print(label.shape)
    break

train_data, val_data, train_labels, val_labels = train_test_split(dataset.data, dataset.labels, test_size=0.2, random_state=42)

