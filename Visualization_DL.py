import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import reduce
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import sys
import pickle
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import functools
import numpy.ma as ma # for masked arrays
import glob
import random
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tqdm.notebook import tqdm
from itertools import groupby
import sklearn



from sklearn.model_selection import train_test_split
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import Saliency
import torchvision

# Loading the data
all_corr = pickle.load(open('./data/SFC_CC200.pkl', 'rb'))
flist = np.array(list(all_corr.keys()))
labels = np.array([all_corr[f][1] for f in flist], dtype = 'uint8')
print('Length of Input subjects : ', len(flist))
print('Length of Output subjects : ', len(labels))
print('Distribution of Labels : ', np.unique(labels, return_counts = True))


# Auto Encoder and Classifier
class Network(nn.Module):
    def __init__(self, num_inputs = 19900):
        super(Network, self).__init__()
        
        self.num_inputs = num_inputs
        
        self.fc_encoder = nn.Sequential (
                nn.Linear(self.num_inputs,2048),
                nn.Tanh(),
                nn.Linear(2048,512),
                nn.Tanh())
        
        self.fc_decoder = nn.Sequential (
                nn.Linear(512,2048),
                nn.Tanh(),
                nn.Linear(2048,self.num_inputs),
                nn.Tanh())
         
        self.classifier = nn.Sequential (
            nn.Dropout(p=0.25),
            nn.Linear(512, 1),
        )

        self.sigmoid = nn.Sigmoid()           
         
    def forward(self, x, eval_classifier = True):

        x = self.fc_encoder(x)
        if eval_classifier:
            x_logit = self.classifier(x)   #   .squeeze(1)
            # x_logit = self.sigmoid(x_logit)
            return x_logit 

        x = self.fc_decoder(x)        
        return x


def attribute_image_features(algorithm, inputs, target = 1):
    # model.zero_grad()
    # model.eval()
    tensor_attributions = algorithm.attribute(inputs = inputs, target = target, return_convergence_delta=True)  
    return tensor_attributions

dl_attributions = []
all_folds = pickle.load(open('./data/AllFoldssubjects.pkl', 'rb'))
for fold in range(10):
    fold_weights = torch.load(f'data/Weights/Fold_{fold+1}.pth', map_location=torch.device('cpu'))
    best_clf_model = Network(num_inputs = 19900) 
    best_clf_model.load_state_dict(fold_weights)
    best_clf_model = best_clf_model.to('cpu')

    test_subjects = all_folds[fold]['test']
    x_asd, y_asd = [], []
    for sample in test_subjects : 
        if(all_corr[sample][1] == 1):
          x_asd.append(all_corr[sample][0])
          y_asd.append(all_corr[sample][1])
    print('Number of ASD subjects in test set : ', len(x_asd))

    x_asd = torch.tensor(x_asd, dtype=torch.float)
    y_asd = np.array(y_asd)  

    y_asd_pred = best_clf_model(x_asd)
    y_asd_pred = y_asd_pred.detach().cpu().numpy()
    y_asd_pred = torch.sigmoid(y_asd_pred)
    y_asd_pred = np.round(y_asd_pred)
    y_asd_pred = np.squeeze(y_asd_pred, axis = 1)

    right_indices = np.where(y_asd_pred == 1)
    x_asd_dl = x_asd[right_indices]
    print('Number of correctly predicted ASD subjects : ', len(x_asd_dl))

    dl_asd = DeepLift(best_clf_model)        
    grads_asd, delta_asd = attribute_image_features(dl_asd, inputs = x_asd_dl, target = 0)                     
    grads_asd = torch.mean(grads_asd, axis = 0)
    grads_asd = grads_asd.detach().cpu().numpy()
    dl_attributions.append(grads_asd)
                            
dl_attributions = np.array(dl_attributions)
print("Attributions shape : ", dl_attributions.shape)

rois_count = {}    # {roi : number of times it is repeated in all 10 folds}

for grads_asd in dl_attributions :
  
    attr_vals_asd = grads_asd.copy()
    thresh = np.percentile(attr_vals_asd, 99)
    attr_vals_asd = np.where(attr_vals_asd > thresh,  1 , 0) # check1
    corr_matrix_asd = np.zeros((200,200))
    corr_matrix_asd[np.triu_indices(200, 1)] = attr_vals_asd
    print('Number of unique elements in corr_matrix : ', np.unique(corr_matrix_asd, return_counts = True))

    max_sum_rows = np.sum(corr_matrix_asd, axis = 1)      # check 2
    top_indices = np.argsort(max_sum_rows)            # Max value indices
    top_values = max_sum_rows[top_indices]      # Max values

    top20_indices  = top_indices[-20 : ]
    top20_values = top_values[-20 : ]

    print('Most repeated ROIS in ASD (Not index values): ', top20_indices + 1)
    print('Number of times ROIS repeated in ASD : ', top20_values)
    
    for index, roi in enumerate(top_indices): 
        if(roi in rois_count):
            rois_count[roi] += top_values[index]
        else : 
          rois_count[roi] = top_values[index]        

rois_count_sorted = dict(sorted(rois_count.items(), key = lambda x : x[1], reverse = True))
print(rois_count_sorted)  # This will give the ROIs in sorted order of its frequency in all the 10 folds. 