#IMPORTS

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os
import copy


import pickle
from utils_data import GenerateDataloader,spec_to_image,get_melspectrogram_db_2
import wandb
import random
from efficientnet_pytorch import EfficientNet
import main_model as mn

# Nova execució del wandb
wandb.init(

    project="MusicClassification",

    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# Detectar si tenim una GPU disponible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Carreguem el dataloader
with open('dataloaderAugmentation.pkl', 'rb') as f:
    loaded_dataloader = pickle.load(f)
    
def initialize_model(num_classes):
    model = EfficientNet.from_pretrained('efficientnet-b0')  # Carga el modelo preentrenado EfficientNet-B0
    num_ftrs = model._fc.in_features
    model._fc = nn.Sequential(
       nn.Linear(num_ftrs, num_classes),  # Añade una capa fully connected intermedia
       nn.Dropout(0.5),  # Añade la capa de Dropout
    )
    model._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False) # Canviem la primera capa convolucional a un canal
    
    input_size = 224  # Tamaño de entrada requerido por EfficientNet
    return model, input_size


num_classes = 8

# Inicialitza el model
model, input_size = initialize_model(num_classes)

# Envia el model a la GPU
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Nombre d'èpoques per entrenament 
num_epochs = 20

# Funció d'optimització 
learning_rate = 0.001
optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate)

# Entrena i evalua
model_feature, histFeature, lossesFeature = mn.train_model(model, loaded_dataloader, criterion, optimizer_ft, num_epochs=num_epochs, change_lr = True, lr = learning_rate)
torch.save(model_feature,"EfficientNet_NO_overfitting.pt")
gt,pr=mn.predict(model_feature,loaded_dataloader,criterion)