
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
,
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
    model = models.resnet50(pretrained=True) # Inicialitzem la ariqutectura i els paramatres pre entrenats
    mn.set_parameter_requires_grad(model,True) # Congelem el model perque no convvi el model
    model.fc = nn.Linear(in_features=2048, out_features=num_classes) # Creem despres de congelar la ultima capa que sera la fully conectet
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Canviem perque accepit 1 chanel que es el que li passem

    input_size=224
    return model,input_size


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

# Funció d'optimització 
model_feature, histFeature, lossesFeature = mn.train_model(model, loaded_dataloader, criterion, optimizer_ft, num_epochs=num_epochs, change_lr = True, lr = learning_rate)
torch.save(model_feature,"Resnet_Millorat.pt")


    