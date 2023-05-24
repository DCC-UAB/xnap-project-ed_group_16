import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display

import utils


import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os
import copy
import librosa
import librosa.display
import shutil
import random
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

          
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
def get_melspectrogram_db_2(filename):
    x, sr = librosa.load(filename, sr=None, mono=True)

    #volem mostres de nomes 5 segons
    if x.shape[0]<5*sr:
        x=np.pad(x,int(np.ceil((5*sr-x.shape[0])/2)),mode='reflect')
    else:
        x=x[:5*sr]

    #transformada rapida de fourier per fer l'espectograma
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    #fem spectograma
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    log_mel = librosa.amplitude_to_db(mel)
    return log_mel
def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled
class GenerateDataloader(Dataset):
  def __init__(self, categories, mp3_f):
    self.mp3_f = mp3_f
    self.data = []
    self.labels = []
    self.category2index={}
    self.index2category={}
    self.categories = categories
    for i, category in enumerate(self.categories):
      self.category2index[category]=i
      self.index2category[i]=category
    for ind in range(len(mp3_f)):
      file_path=mp3_f[ind]
      print(file_path)
      self.data.append(spec_to_image(get_melspectrogram_db_2(file_path))[np.newaxis,...])
      self.labels.append(file_path[10:13])
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]