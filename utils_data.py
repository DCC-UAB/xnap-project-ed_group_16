import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

          
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

# Funció per passade de file mp3 a espectograma amb volum

def get_melspectrogram_db_volum(filename, volum):
    x, sr = librosa.load(filename, sr=None, mono=True)

    # Volem mostres de nomes 10 segons
    if x.shape[0]<10*sr:
        x=np.pad(x,int(np.ceil((10*sr-x.shape[0])/2)),mode='reflect')
    else:
        x=x[:10*sr]
        
    # Resamplejar l'audio a una freqüència de mostreig comú
    target_sr = 22050
    audio = librosa.resample(x, orig_sr = sr, target_sr = target_sr)

    # Transformada ràpida de fourier per fer l'espectograma
    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    # Fem l'espectograma
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    if volum == "up":
        log_mel = librosa.amplitude_to_db(mel*2)  # Multipliquem per l'amplificador
    else:
        log_mel = librosa.amplitude_to_db(mel*0.5)  # Multipliquem per l'amplificador
    return log_mel
  

# Funció per passade de file mp3 a espectograma

def get_melspectrogram_db(filename):
    x, sr = librosa.load(filename, sr=None, mono=True)

    # Volem mostres de nomes 10 segons
    if x.shape[0]<10*sr:
        x=np.pad(x,int(np.ceil((10*sr-x.shape[0])/2)),mode='reflect')
    else:
        x=x[:10*sr]
        
    # Resamplejar l'audio a una freqüència de mostreig comú
    target_sr = 22050
    audio = librosa.resample(x, orig_sr = sr, target_sr = target_sr)

    # Transformada ràpida de fourier per fer l'espectograma
    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    # Fem l'espectograma
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    log_mel = librosa.amplitude_to_db(mel)
    return log_mel
  

# Funció per passar d'espectograma a imatge

def spec_to_image(spec, eps=1e-6):
  mean = spec.mean() 
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps) # Normalitzem l'esepctograma
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled


# Funció per passar d'espectograma a imatge

def spec_to_image_noise(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps) # Normalitzem l'esepctograma
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  noise = np.random.normal(0, 0.1, spec_scaled.shape)
  noisy_image = np.clip(spec_scaled + noise, 0, 255).astype(np.uint8)   # Afegir soroll gaussià amb la mateixa forma
  return noisy_image


class GenerateDataloader(Dataset):
  def __init__(self, mp3_f, tipus, train_labels, dict_class, data_aug):
    self.mp3_f = mp3_f
    self.data = []
    self.labels = []
    
    # Afegir dades en el cas d'indicar-ho 
    if data_aug == "True": 
      mp3_file_aug = random.sample(mp3_f, int(len(mp3_f)*0.30))
      self.mp3_f += mp3_file_aug
    
    for ind in range(len(self.mp3_f)):
      file_path=self.mp3_f[ind]
      print(file_path)
      if ind < len(mp3_f):
        im=spec_to_image(get_melspectrogram_db(file_path))[np.newaxis,...]
      elif len(mp3_f) < ind < len(mp3_f)*1.20:  # 20% de les noves dades a aprtir d'espectogramas amb diferent volum
        if ind % 2 == 0:
          im = spec_to_image(get_melspectrogram_db_volum(file_path, "up"))[np.newaxis,...]    # 10% augmenant volum
        else:
          im = spec_to_image(get_melspectrogram_db_volum(file_path, "down"))[np.newaxis,...]   # 10% disminuint volum
      else:
        im = spec_to_image_noise(get_melspectrogram_db(file_path))[np.newaxis,...]   # 10% afegint soroll a les imatges
      
      self.data.append(im)


      if tipus=="train":
        trackid=file_path[26:32]
        label_track=train_labels.loc[train_labels['track_id'] == int(trackid), 'genre'].values[0] # Obtenim el gènere corresponent
        label_track=dict_class[label_track]  # Guardem el valor corresponent en comptes del nom del gènere
        self.labels.append(int(label_track))
      else:
        trackid=file_path[25:31]
        label_track=train_labels.loc[train_labels['track_id'] == int(trackid), 'genre'].values[0] # Obtenim el gènere corresponent 
        label_track=dict_class[label_track] # Guardem el valor corresponent en comptes del nom del gènere
        self.labels.append(int(label_track))
        
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]