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
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

import pickle
from utils_data import GenerateDataloader,spec_to_image,get_melspectrogram_db_2
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import wandb
import random

# start a new wandb run to track this script
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


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    acc_history = {"train": [], "val": []}
    losses = {"train": [], "val": []}

    # we will keep a copy of the best weights so far according to validation accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device,dtype=torch.float32)
                #inputs= inputs.to(torch.cuda.FloatTensor)
                labels = torch.tensor(labels)
                labels = labels.to(device,dtype=torch.long)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    losses[phase].append(loss.item())

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == "train":

                wandb.log({"acc": epoch_acc, "loss": epoch_loss})

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            acc_history[phase].append(epoch_acc.item())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, acc_history, losses


#CONGELAR PESOS PERL FEATRURE
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

#FEATURE EXTRACTION
def initialize_model(num_classes):
  model = models.resnet18(pretrained=True) # Inicialitzem la ariqutectura i els paramatres pre entrenats
  set_parameter_requires_grad(model,True) #Congelem el model perque no convvi el model
  model.fc = nn.Linear(in_features=512, out_features=num_classes) #Creem despres de congelar la ultima capa que sera la fully conectet
  model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # canviem perque accepit 1 chanel que es el que li passem

  input_size=224
  return model,input_size

num_classes = 8
# Initialize the model
model, input_size = initialize_model(num_classes)


  # Send the model to GPU
model = model.to(device)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Number of epochs to train for 
num_epochs = 25

optimizer_ft = optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate
model_feature, histFeature, lossesFeature = train_model(model, loaded_dataloader, criterion, optimizer_ft, num_epochs=num_epochs)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(lossesFeature["train"], label="training loss")
ax1.plot(lossesFeature["val"], label="validation loss")
ax1.legend()

ax2.plot(histFeature["train"],label="training accuracy")
ax2.plot(histFeature["val"],label="val accuracy")
ax2.legend()

plt.show()  

wandb.finish()