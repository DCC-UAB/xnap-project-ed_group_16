#IMPORTS
from __future__ import print_function 
from __future__ import division
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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import sys
    # caution: path[0] is reserved for script path (or '' in REPL)

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

import pickle
from utils_data import GenerateDataloader,spec_to_image,get_melspectrogram_db_2
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import wandb
import random

import main_model as mn
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# Load the saved dataloader
with open('dataloaderClasse.pkl', 'rb') as f:
    loaded_dataloader = pickle.load(f)
    print(loaded_dataloader)

num_classes = 8
# Initialize the model
def initialize_model(num_classes):
  model = models.resnet18(pretrained=True) # Inicialitzem la ariqutectura i els paramatres pre entrenats
  mn.set_parameter_requires_grad(model,True) #Congelem el model perque no convvi el model
  model.fc = nn.Linear(in_features=512, out_features=num_classes) #Creem despres de congelar la ultima capa que sera la fully conectet
  model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # canviem perque accepit 1 chanel que es el que li passem

  input_size=224
  return model,input_size

model, input_size = initialize_model(num_classes)



  # Send the model to GPU
model = model.to(device)


# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Number of epochs to train for 
num_epochs = 3

optimizer_ft = optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate
model_feature, histFeature, lossesFeature = mn.train_model(model, loaded_dataloader, criterion, optimizer_ft, num_epochs=num_epochs)
gt,pr=mn.predict(model_feature,loaded_dataloader,criterion)
