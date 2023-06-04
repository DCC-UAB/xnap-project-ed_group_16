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
from sklearn.metrics import confusion_matrix

# Detectar si tenim una GPU disponible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, num_epochs, change_lr, lr):
    since = time.time()

    acc_history = {"train": [], "val": []}
    losses = {"train": [], "val": []}

    # Mantenir una copia dels millors pesos fins al moment d'acord amb la precisió de la validació
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Mirar d'actualitzar l'optimitzador si es vol utilitzar learning rate decay
        if change_lr:
          optimizer = lr_decay(optimizer, epoch, lr, model)
          
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Cada època té una fase d'entrenament i validació
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Establir el model en mode entrenament
            else:
                model.eval()   # Establir el model en mode validació

            running_loss = 0.0
            running_corrects = 0

            # Iterar sobre les dades
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device, dtype = torch.float32)
                labels = torch.tensor(labels)
                labels = labels.to(device, dtype = torch.long)

                # Establir a zero els gradients dels paràmetres
                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == 'train'):
                    # Obtenir resultats del model i calcular la pèrdua
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    losses[phase].append(loss.item())

                    _, preds = torch.max(outputs, 1)

                    # Backward + optimizar només si estem a la fase d'entrenament
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Estadístiques
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == "train":
                # Registrar mètriques al Wandb
                wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc})
            else:
                wandb.log({"val_loss": epoch_loss, "val_acc": epoch_acc})
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Còpia profunda del model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            acc_history[phase].append(epoch_acc.item())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Carregar els pesos del millor model
    model.load_state_dict(best_model_wts)
    return model, acc_history, losses


# Congelar pesos pel Features extraction
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Adaptar el learning rate 
def lr_decay(optimizer, epoch, lr, model):
    if epoch%10==0:  # Cada deu èpoques es canvia
        new_lr = lr / (10**(epoch//10)) 
        optimizer = optim.Adam(model.parameters(), lr=new_lr)  # Generar el nou optimitzador amb el new_lr
        print(f'Changed learning rate to {new_lr}')
    return optimizer

def predict(model, dataloaders, criterion):
    
    # Establir el model en mode validació
    model.eval()
    ground_truth = []
    predictions = []
    for inputs, labels in dataloaders["val"]: # unes 1600 mp3

        inputs = inputs.to(device,dtype=torch.float32)
        labels = torch.tensor(labels)
        labels = labels.to(device,dtype=torch.long)

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        ground_truth.extend(labels.tolist())
        predictions.extend(preds.tolist())

    # Generar la confusion matrix al wandb
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=ground_truth, preds=predictions, class_names=["Hip-Hop","Pop","Folk","Experimental","Rock","International","Electronic","Instrumental"])})
    predictions=np.array(predictions)
    cm=confusion_matrix(ground_truth,predictions.ravel())

    # Obtenir el nombre de classes
    num_classes = cm.shape[0]

    # Calcular l'accuracy per cada classe
    class_acc = []
    for i in range(8):
        class_correct = cm[i,i]
        class_total = sum(cm[i,:])
        class_acc.append(class_correct / class_total)
        print("Accuracy for class ",i," :", (class_correct / class_total)," Total samples class ",class_total)
    return ground_truth,predictions


