from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import copy
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, models, transforms
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from torchvision.io import read_image

from dataclasses import dataclass
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    accuracy_score,
)


@dataclass
class RetinaModel:
    label_num: int
    condition_name: str
    model_name: str
    learning_rate: float
    momentum: float
    feature_extract: bool
    num_epochs: int
    is_inception: bool

    # The following code is taken from the pytorch tutorial on finetuning models
    # LINK: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    def train_model(self, model, dataloaders, criterion, optimizer, device):
        """
        The train_model function handles the training and validation of a given model.
        As input, it takes a PyTorch model, a dictionary of dataloaders, a loss function, an optimizer,
        a specified number of epochs to train and validate for, and a boolean flag for when the model is
        an Inception model.
        The is_inception flag is used to accomodate the Inception v3 model,
        as that architecture uses an auxiliary output and the overall model loss respects both
        the auxiliary output and the final output, as described here.
        The function trains for the specified number of epochs and after each epoch runs
        a full validation step.
        It also keeps track of the best performing model (in terms of validation accuracy),
        and at the end of training returns the best performing model. After each epoch,
        the training and validation accuracies are printed.
        """
        val_acc_history = []
        best_model_wts = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0
        since = time.time()
        for epoch in tqdm(range(self.num_epochs)):
            # Each epoch has a training and validation phase
            for phase in ["Training", "Validation"]:
                if phase == "Training":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                all_preds = []  # all predictions made over the batches
                all_labels = []  # all labels throug the batches
                all_outputs = []  # Actual output of the models TODO IS THIS RIGHT???/

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.float()
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "Training"):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if self.is_inception and phase == "Training":
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)
                        all_preds += list(preds.cpu().numpy())
                        all_labels += list(labels.cpu().numpy())
                        all_outputs += list(outputs.cpu().detach().numpy())

                        # backward + optimize only if in training phase
                        if phase == "Training":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = (
                    running_loss / len(dataloaders[phase].dataset)
                    if len(dataloaders[phase].dataset)
                    else running_loss
                )
                epoch_acc = (
                    running_corrects.double() / len(dataloaders[phase].dataset)
                    if len(dataloaders[phase].dataset)
                    else running_loss
                )

                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

                # deep copy the model
                if phase == "Validation" and epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == "Validation":
                    val_acc_history.append(epoch_acc)

        self.time_elapsed = time.time() - since
        print(
            "\nTraining complete in {:.0f}m {:.0f}s".format(
                self.time_elapsed // 60, self.time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(self.best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        # Record Performance Features
        self.save_performance(all_preds, all_labels, all_outputs, model)
        return model, val_acc_history

    def save_performance(self, preds, labels, outputs, model):
        self.preds = preds
        self.labels = labels
        self.outputs = outputs
        self.best_acc = self.best_acc
        self.model = model

        self.f1_score = f1_score(y_true=labels, y_pred=preds)
        self.precision_score = precision_score(y_true=labels, y_pred=preds)
        self.recall_score = recall_score(y_true=labels, y_pred=preds)
        self.accuracy = accuracy_score(y_true=labels, y_pred=preds)
        self.confusion_matrix = confusion_matrix(labels, preds)
