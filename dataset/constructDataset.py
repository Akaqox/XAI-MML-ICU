"""
Created on Tue Jun 25 21:41:16 2024

@author: Akaqox(Salih KIZILIŞIK)

This script is experimental codes of paper "Explainable Multimodal Machine Learning Model
for Predicting Intensive Care Unit Admission " by S.Kizilisik et al. You may use the codes only for research. 
Plese cite the paper if you use any part of the codes.

"""


from utils.utils import readConfig, create_destroy_dic
import os
import shutil
import glob
import numpy as np
import tensorflow as tf
from dataset.augmentation import readAndCategorize, create_pn

config = readConfig()
paths = config["paths"]
root = os.path.join(paths["PATH"], paths["dataset"])
ready = os.path.join(paths["PATH"], paths["ready"])


def createLabel(files):
    for path in files:
        path = os.path.join(root,path)
        arr = np.load(path)
        print(arr.shape)
        if "0" in path:
            label0 = np.zeros((arr.shape[0], 1), dtype=int)   
        if "1" in path:   
            label1 = np.ones((arr.shape[0], 1), dtype=int)   
    
    label = np.concatenate((label0, label1), axis=0)        
    label = tf.keras.utils.to_categorical(label, 2).astype(int)
    print(label.shape)
    return label
            

def construct_subset(seg, src, dest, dataset_name):

    if dataset_name == "train-augmented":
        print(dest)
        for i in range(2):
            
            train_src = src +"/" + str(i)+"/" 
            train_dest = dest.split("-")[0] + "/" + str(i) + "/" 
            print(train_src)
            print(train_dest)
            allfiles = glob.glob(train_src + '/*.npy')
            print(allfiles)
            for f in allfiles:
                
                src_path = f
                dst_path = os.path.join(train_dest, os.path.basename(f))
                
                print(src_path)
                print(dst_path)
                shutil.move(src_path, dst_path)
            
    else:
        #Construct numpy array for both Validation and Test
            root_folder = src
            
            lists = []
            lists.append(src + "/0") 
            lists.append(src + "/1")
            print(root_folder)
            
            create_pn(root_folder, lists[0], lists[1])
            
            readAndCategorize(root_folder, lists[0], lists[1])
            
            file_lists = []
            file_lists.append(glob.glob(src + "/0" +'/*.dcm'))
            file_lists.append(glob.glob(src + "/1" +'/*.dcm'))
            
            create_destroy_dic(dest + "/0/")
            create_destroy_dic(dest + "/1/")
            
            for i, npy_list in enumerate(file_lists):
                seg.cropAll(npy_list, i, dest, dicom=True)
            
    return
    
#Construct train dataset from augmented, segmented numpy file    
def constructDataset(seg): 
    dataset = {
        "train" : paths["train"],
        "train-augmented" : paths["segmented"],
        "val" : paths["val"],
        "test" : paths["test"],
        }
    for key, input_path in dataset.items():
        
        #if not already create ready_to_train folder
        save_path = ready + "/" + key
        create_destroy_dic(save_path)
        #Construct Validation and Test files    
        construct_subset(seg, input_path, save_path, key)