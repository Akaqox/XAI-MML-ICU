# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 22:33:54 2024

@author: Akaqox

This script is experimental codes of paper "Explainable Multimodal Machine Learning Model
for Predicting Intensive Care Unit Admission " by S.Kizilisik et al. You may use the codes only for research. 
Plese cite the paper if you use any part of the codes.

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np_arr = np.load("../resources/train_0.npy")

for i in range(10):
    mpl.rcParams['figure.dpi'] = 420

    arr = np_arr[i, :, :]
    
    # Display the original image
    plt.figure(figsize=(16, 3))

    # Normalize the original image
    img = arr.astype('float32')
    img /= 4096

    # Create a gridspec layout
    gs = plt.GridSpec(1, 6, width_ratios=[1, 0.1, 1, 1, 1, 1], wspace=0.01)

    # Display original image
    plt.subplot(gs[0, 0])
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.gray)

    # Augmentation: Width Shift
    datagen_trans = ImageDataGenerator(width_shift_range=[-0.09,-0.1], fill_mode="nearest")
    aug_trans = datagen_trans.flow(np.expand_dims(np.expand_dims(arr, axis=-1), axis=0), batch_size=1)
    img_trans = next(aug_trans)
    plt.subplot(gs[0, 2])
    plt.axis('off')
    plt.imshow(img_trans[0, :, :, 0], cmap=plt.cm.gray)

    # Augmentation: Height Shift
    datagen_transh = ImageDataGenerator(height_shift_range=[-0.09,-0.1], fill_mode="nearest")
    aug_transh = datagen_transh.flow(np.expand_dims(np.expand_dims(arr, axis=-1), axis=0), batch_size=1)
    img_transh = next(aug_transh)
    plt.subplot(gs[0, 3])
    plt.axis('off')
    plt.imshow(img_transh[0, :, :, 0], cmap=plt.cm.gray)

    # Augmentation: Rotation
    datagen_rot = ImageDataGenerator(rotation_range=10, fill_mode="nearest")
    aug_rot = datagen_rot.flow(np.expand_dims(np.expand_dims(arr, axis=-1), axis=0), batch_size=1)
    img_rot = next(aug_rot)
    plt.subplot(gs[0, 4])
    plt.axis('off')
    plt.imshow(img_rot[0, :, :, 0], cmap=plt.cm.gray)

    # Augmentation: Zoom
    datagen_zoom = ImageDataGenerator(zoom_range=[1.2,1.2], fill_mode="nearest")
    aug_zoom = datagen_zoom.flow(np.expand_dims(np.expand_dims(arr, axis=-1), axis=0), batch_size=1)
    img_zoom = next(aug_zoom)
    plt.subplot(gs[0, 5])
    plt.axis('off')
    plt.imshow(img_zoom[0, :, :, 0], cmap=plt.cm.gray)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("../out/" + str(i) + "_aug.png")
    plt.show()