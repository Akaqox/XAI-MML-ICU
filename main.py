# %%

# -*- coding: utf-8 -*-

"""
Created on Sun Nov  7 19:21:09   2024

@authors: Akaqox(S.Kizilisik) (kzlsksalih@gmail.com) S.Candemir (cand2emirsema@gmail.com) Ayşegül Terzi


This script is experimental codes of paper "Explainable Multimodal Machine Learning Model
for Predicting Intensive Care Unit Admission " by S.Kizilisik et al. You may use the codes only for research. 
Plese cite the paper if you use any part of the codes.

"""
import gc
from utils.utils import readConfig
from dataset.augmentation import Augment
from utils.segmentation import Segment
from dataset.constructDataset import constructDataset
from train.train import Train

#Taking hyperparameters
config = readConfig()
runningConfig = config["runningConfig"]
root = config["paths"]["PATH"]

#%%
#Processing and Augmentation Stage 
#This stage consist loading data, calculate mean resolution, augment the data
gc.collect()
#augment
if runningConfig["augmentation"] == True:
    aug = Augment(config)
    aug()
    del aug

#%%

if runningConfig["segmentation"] == True:
    seg = Segment()
    seg()   
    gc.collect()
#%%
if runningConfig["construct_dataset"] == True:
    seg = Segment()
# %%
    constructDataset(seg)

    del seg
    gc.collect() 
#%%
if runningConfig["train"] == True:
    for i in range(50):
        train = Train()
        train.fit()
        gc.collect()