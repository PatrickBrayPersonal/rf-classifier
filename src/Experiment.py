import sys
sys.path.append('/content/drive/MyDrive/CDA_Final_Project/Notebooks')

from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import copy
import time
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from torchvision import datasets, models, transforms
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from torchvision.io import read_image

from src.data.RetinaDataset import RetinaDataset
from src.models.RetinaModel import RetinaModel

class Experiment():
    def __init__(self, model_names, data_dir, label_file, results_dir, batch_size, num_epochs, feature_extract, learning_rates, experiment_name, DEBUG):
        if DEBUG:
            model_names = ['resnet']
            batch_size = 64
            num_epochs = 1
            learning_rates = [0.1]
        self.model_names = model_names
        self.data_dir = data_dir
        self.label_file = label_file
        self.results_dir = results_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.feature_extract = feature_extract
        self.learning_rates = learning_rates
        self.experiment_name = experiment_name
        self.DEBUG = DEBUG
        self.results_dict = defaultdict(list)
        self.num_debug_imgs = 10

    def read_labels(self, name):
        '''
        returns the labels files as a dataframe
        the files that failed to upload to the google drive are removed from the df
        '''
        label_file = pd.read_csv(self.label_file.format(name, name.lower()))
        # Remove non-existent files
        no_exist = []
        for i in range(1, label_file.ID.max()+1):
            file_exists = os.path.exists(self.data_dir.format(name, name.lower()) + str(i) + '.png')
            if not file_exists:
                no_exist.append(i)
        print(f'Images {no_exist} missing from {name} dataset')
        if self.DEBUG:
            for i in range(1, 28):
                label_file.iloc[:, i] = np.random.choice([0, 1], size=len(label_file), p=[.5, .5])
            label_file = label_file.head(self.num_debug_imgs)
        return label_file.loc[~label_file.ID.isin(no_exist), :]

    def make_weights_for_balanced_classes(self, images, nclasses): 
        '''
        accepts a set of labels and returns the "weight" of each image because of its label
        Seems overly complicated but there is probably good reason for it
        SOURCE: https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703?page=2
        '''                       
        count = [0] * nclasses                                                   
        for item in images:                                                         
            count[item] += 1                                                     
        weight_per_class = [0.] * nclasses                                      
        N = float(sum(count))                                                  
        for i in range(nclasses):                                                   
            weight_per_class[i] = N/float(count[i]) if count[i] else 0                                 
        weight = [0] * len(images)                                              
        for idx, val in enumerate(images):                                          
            weight[idx] = weight_per_class[val]                                  
        return weight   

    def run(self, num_labels):
        '''
        Executes the full eperiment that will be reported on for CDA Project
        '''
        dataset_names = ['Training', 'Validation']
        dataloaders_dict = {}
        
        for model_name in tqdm(self.model_names): # run models
            print(f'\n\tDownloading "{model_name}"')
            # Initialize the model for this run
            otb_model, device, input_size = self.initialize_model(model_name=model_name, 
                                                                num_classes=2, 
                                                                feature_extract=self.feature_extract, 
                                                                use_pretrained=True)
            # for label_num in range(1, num_labels+1):
            for label_num in range(10, 15):
                for name in dataset_names: # initialize dataloaders
                    rd, sampler, condition_name = self.gen_dataloader(name, label_num)
                    dataloaders_dict[name] = DataLoader(rd, batch_size=self.batch_size, shuffle=False, sampler=sampler)

                self.dataloaders_dict = dataloaders_dict # TODO: Clean up
                print(f'Building models for {condition_name}, condition {label_num}/{num_labels}')

                for learning_rate in self.learning_rates:
                    optimizer = self.create_optimizer(otb_model, learning_rate)
                    criterion = nn.CrossEntropyLoss()
                    rm = RetinaModel(label_num=label_num,
                                    condition_name=condition_name,
                                    model_name=model_name,
                                    learning_rate=learning_rate,
                                    momentum=0.9,
                                    feature_extract=self.feature_extract,
                                    num_epochs=self.num_epochs,
                                    is_inception=(model_name=='inception'))
                    rm.train_model(otb_model, dataloaders_dict, criterion, optimizer, device)
                    self.record_results(rm)
                    self.save_results() # save down to long term every time in case of issues
        print('COMPLETE!')
        return rm

    def record_results(self, instance):
        for key, value in instance.__dict__.items():
            self.results_dict[key].append(value)

    def save_results(self):
        # self.results_dict = {key, val.cpu() for key, val in self.results_dict.items()}
        pd.DataFrame(self.results_dict).to_pickle(os.path.join(self.results_dir, f'Experiment_{self.experiment_name}.pckl'))

    def set_parameter_requires_grad(self, model, feature_extracting):
        '''
        This helper function sets the .requires_grad attribute of the parameters in the 
        model to False when we are feature extracting. 
        By default, when we load a pretrained model all of the parameters have .requires_grad=True,
        which is fine if we are training from scratch or finetuning. 
        However, if we are feature extracting and only want to compute gradients for 
        the newly initialized layer then we want all of the other parameters to not require gradients. 
        This will make more sense later.
        '''
        if feature_extracting:
                for param in model.parameters():
                        param.requires_grad = False

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #     variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs,num_classes)
            input_size = 299

        else:
                print("Invalid model name, exiting...")
                exit()

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Send the model to GPU
        model_ft = model_ft.to(device)

        return model_ft, device, input_size

    def create_optimizer(self, model_ft, learning_rate, momentum=0.9):
        # Gather the parameters to be optimized/updated in this run. If we are
        #    finetuning we will be updating all parameters. However, if we are
        #    doing feature extract method, we will only update the parameters
        #    that we have just initialized, i.e. the parameters with requires_grad
        #    is True.
        params_to_update = model_ft.parameters()
        if self.feature_extract:
                params_to_update = []
                for name,param in model_ft.named_parameters():
                        if param.requires_grad == True:
                                params_to_update.append(param)
        optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
        return optimizer_ft

    def gen_dataloader(self, name, label_num):
        transform = transforms.Compose([transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # Read Labels
        label_file = self.read_labels(name)
        # Create weighted random sampler for imbalanced dataset
        weights = self.make_weights_for_balanced_classes(label_file.iloc[:, label_num], 2)                                                                
        weights = torch.DoubleTensor(weights)
        try:
            sampler = WeightedRandomSampler(weights, len(weights))  
        except:
            sampler = None
        # Instantiate Datset
        rd = RetinaDataset(labels=label_file,
                            data_dir=self.data_dir.format(name, name.lower()),
                            transform=transform,
                            label_num=label_num)
        condition_name = label_file.columns[label_num]
        return rd, sampler, condition_name

    def testing_preds(self, model, dataloader, device):
        all_preds =[]
        all_labels = []
        all_outputs = []
        for inputs, labels in dataloader:
            inputs = inputs.float()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds += list(preds.cpu().numpy())
            all_labels += list(labels.cpu().numpy())
            all_outputs += list(outputs.cpu().detach().numpy())
        return all_preds, all_labels, all_outputs