# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 21:41:16 2021

@author: Akaqox(S.Kizilisik)

This script is experimental codes of paper "Explainable Multimodal Machine Learning Model
for Predicting Intensive Care Unit Admission " by S.Kizilisik et al. You may use the codes only for research. 
Plese cite the paper if you use any part of the codes.

"""

import os
import glob
import cv2
import gc
from utils.utils import readXray, create_destroy_dic, processMetadata, save_numpy
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



class Augment():
    def __init__(self, config):
        
        self.config = config
        paths = self.config["paths"]
        root = paths["PATH"]
        self.save_path = os.path.join(root, paths["augmented"])
        self.img_path = os.path.join(root, paths["dataset"])
        self.root = root
        self.paths = paths
        augmentation = config["augmentation"]
        augmentation_params = augmentation["augmentation_params"]
        self.datagen = ImageDataGenerator(**augmentation_params)
        self.target_num = augmentation["target_num"]
        self.noise = augmentation["noise"]
        self.kernel = (augmentation["kernel_size"], augmentation["kernel_size"])
        self.visualization = config["runningConfig"]["visualization_on"]

    def __call__(self):

        #define paths
        train_path = os.path.join(self.root, self.paths["train"])
        print(train_path)
        
        #Define or if needed create folders
        path0 = os.path.join(train_path,"0")
        path1 = os.path.join(train_path,"1")
        
        create_pn(train_path, path0, path1)
        
        #Categorize data from metadata    
        npy_list = readAndCategorize(train_path, path0, path1)
        
        #Start the augmentation process
        self.augment(npy_list)
        return npy_list
    
            
    def _apply_augment(self, npylist, label):
        print("merhaba")
        gc.collect()
        
        number_of_img = len(npylist)
        num_additional_img = self.target_num - number_of_img
        print("Number of needed additional img is " + str(num_additional_img)+ " for label:" + str(label))
        
        if num_additional_img > 0:
            save_path= os.path.join(self.save_path, str(label))
            #if not create folder
            create_destroy_dic(save_path)
                
            i = 0
            while num_additional_img > 0:
                
                if(i>=len(npylist)):
                    i = 0
                print(npylist[i])
                arr = readXray(npylist[i])
                print(arr.shape)
                arr = cv2.resize(arr, (1525, 1270)) 
                arr = np.expand_dims(arr, axis=0)
                aug_iter = self.datagen.flow(np.expand_dims(arr, axis=-1), batch_size=1)
                img = next(aug_iter)
                print(np.std(img))
                
                #Gaussian Noise
                img_blurred = cv2.GaussianBlur(img[0,:,:,0], self.kernel, np.std(img) * self.noise)
                img[0,:, :, 0] = img_blurred
                print(img.shape)
                aug = img
                
                if self.visualization:
                    plt.imshow(aug[0,:,:,0], )
                    plt.show()
                    print("Needed Image Count is " + str(num_additional_img))
                basename = os.path.basename(npylist[i]).split(".")[0]
                save = os.path.join(save_path, basename) 
                print(save)
                save_numpy(aug, save, 2)
                i = i + 1
                num_additional_img -= 1
                print(num_additional_img)
                
                    
    def augment(self, lists):
        for i, npy_list in enumerate(lists):
            print(i)
            self._apply_augment(npy_list, i)
            
def readAndCategorize(root, path0, path1):
    
    #Read the Metadata and Categorize
    df = processMetadata()
    df = df[["to_patient_id", "is_icu"]]
    #♥ Categorizing
    for path in glob.glob(root + '/*.dcm'):
        img_name = os.path.basename(path)
        img_id = img_name[:7]
        temp_df = df[df["to_patient_id"] == img_id]
        
        img_label = temp_df.iloc[0]["is_icu"]
        
        if img_label == 0:
            os.rename(path, os.path.join(path0 , img_name))
        else:
            os.rename(path, os.path.join(path1 , img_name))
            
    print("labeling Completed \n")
    list0 = glob.glob(path0 + '/*.dcm')
    list1 = glob.glob(path1 + '/*.dcm')
    return list0, list1            
    
def create_pn(root, path0, path1):
    print(os.path.isdir(path0))
    if os.path.isdir(path0) or os.path.isdir(path1):
        
        for f in os.listdir(path0):
            os.rename(os.path.join(path0, f), os.path.join(root, f))
            
        for f in os.listdir(path1):
            os.rename(os.path.join(path1, f), os.path.join(root, f))
                      
    create_destroy_dic(path0)
    create_destroy_dic(path1)
    gc.collect()
 

    
        