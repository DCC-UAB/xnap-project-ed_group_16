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
model, input_size = mn.initialize_model(num_classes)


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
