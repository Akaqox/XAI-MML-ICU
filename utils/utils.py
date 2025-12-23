# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 21:41:16 2021

@author: cand07, Akaqox

This script is experimental codes of paper "Explainable Multimodal Machine Learning Model
for Predicting Intensive Care Unit Admission " by S.Kizilisik et al. You may use the codes only for research. 
Plese cite the paper if you use any part of the codes.

"""

import os
import shutil
import cv2
import numpy as np
import SimpleITK as sitk
import h5py
from skimage import io, exposure, img_as_float, transform, morphology
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
import json
import pandas as pd

from keras.models import model_from_json
from keras.optimizers import Adam
from datetime import datetime


def readConfig():

    
    # Opening JSON file
    with open('config.json', 'r') as openfile:
     
        # Reading from json file
        json_object = json.load(openfile)
     
    return json_object
config = readConfig()

paths = config["paths"]
    
hypp = config["parameters"]
        
dropout = hypp["dropout"]
# def VGG16():
#     model = Sequential()
#     model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
#     model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#     model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#     model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#     model.add(Flatten())
#     model.add(Dense(units=4096,activation="relu"))
#     model.add(Dense(units=4096,activation="relu"))
#     model.add(Dense(units=2, activation="softmax"))

#     return model


def base_model(input_shape):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape= input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    return model

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Compile(model, l_rate):
   
    # opt = tf.keras.optimizers.SGD(lr=l_rate, decay=1e-6, momentum=0.9, nesterov=False)
    opt = Adam(lr=l_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
    # opt = Adam(lr=l_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def segmentLung(seg_model, img):
    
    mask_size = img.shape
    
    "input shape..." 
    # seg_w = seg_model.input.shape[1]         
    # seg_h = seg_model.input.shape[2]    
    # im_shape = (seg_w,seg_h)  
    im_shape = (224,224,1)
   
    img = transform.resize(img, im_shape)
    
    " Predict the lung region "
    img = np.expand_dims(img, -1)
    img = np.expand_dims(img,axis = 0)
    pred = seg_model.predict(img, verbose = None)[..., 0].reshape(im_shape)
    img = np.squeeze(img)
    
    " 0-1 mask conversion "
    pr = pred > 0.5
    pr = morphology.remove_small_objects(pr, int(0.1*im_shape[1]))    
  
    " show predicted results "  
    # show_segment(img, pred, pr)   
    
    "mask"
    pr = pr.astype(np.float32) 
    mask = transform.resize(pr, mask_size) 
    # show_frame(mask)
    
    return mask


def loadSegmentModel():
    """
    Robust loader that handles:
    1. Legacy split format (JSON architecture + H5 weights)
    2. Modern full-model format (H5 or Keras file)
    """
    
    # PATHS
    path_json = 'model/segment_model.json'
    path_weights = 'model/segment_model.h5'
    path_full_model = 'model/segment_model.h5' # Often full models use the same name
    
    # --- ATTEMPT 1: Legacy (JSON + Weights) ---
    try:
        if os.path.exists(path_json):
            print("Found JSON architecture. Attempting legacy load...")
            
            with open(path_json, 'r') as json_file:
                loaded_model_json = json_file.read()
            
            # Create model from JSON
            model = model_from_json(loaded_model_json)
            
            # Load weights
            if os.path.exists(path_weights):
                model.load_weights(path_weights)
                print("Success: Loaded legacy model (JSON + Weights)")
                return model
            else:
                print(f"Warning: JSON found but weights file ({path_weights}) is missing.")
    
    except Exception as e:
        print(f"Legacy load failed ({e}). This is expected in newer TF versions.")

    # --- ATTEMPT 2: Modern (Full Model Load) ---
    print("Attempting to load as a full Keras model...")
    try:
        # Try loading the .h5 file directly as a full model
        if os.path.exists(path_full_model):
            model = load_model(path_full_model)
            print(f"Success: Loaded full model from {path_full_model}")
            return model
            
        # Optional: Check for .keras format (newer standard)
        path_keras = 'model/segment_model.keras'
        if os.path.exists(path_keras):
            model = load_model(path_keras)
            print(f"Success: Loaded full model from {path_keras}")
            return model
            
    except Exception as e:
        print(f"Full model load failed: {e}")

    raise RuntimeError("Could not load segmentation model. Ensure 'model/segment_model.h5' is a valid Keras model.")

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# def cropImg(img,mask):

#    vertical_sum = np.sum(mask, axis = 0)
#    horizontal_sum = np.sum(mask, axis = 1)
#    
#    indexes = np.nonzero(vertical_sum)[0]
#    border_l = indexes[0]
#    border_r = indexes[len(indexes)-1]
#    
#    indexes = np.nonzero(horizontal_sum)[0]
#    border_up = indexes[0]
#    border_down = indexes[len(indexes)-1]
#    
#    crop = img[ border_up:border_down, border_l:border_r]
#    # show_frame(crop)
#    
#    return crop

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

 
def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


def load_hdf5(infile): 
    with h5py.File(infile,"r") as f: 
        return f["image"][()]
      
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""        

" plot history for accuracy " 
def plot_acc(history, title = None):
    fig = plt.figure()
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)

    plt.grid(True) 
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc=0)
    plt.savefig('Acc.png')


" plot history for loss " 
def plot_loss(history, title = None):
    fig = plt.figure()
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.grid(True) 
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc=0)
    plt.savefig('Loss.png')

        
def confusion_matrix_binary(y_true, y_pred, threshold):
    """
    Given the threshold, it calculates the Confusion Matrix. 
    :params y_true
    :params y_pred 
    :params threshold
    """

    n_shape = y_pred.shape[0]
    _y_pred = np.argmax(y_pred, axis =1)
    _y_true = np.zeros([n_shape, 1])
    _y_pred = np.zeros([n_shape, 1])
    for i in range(n_shape):
        if y_true[i,0]:
            _y_true[i] = 0
        else:
            _y_true[i] = 1

        if y_pred[i,0] > threshold:
            _y_pred[i] = 0
        else:
            _y_pred[i] = 1

    cm = confusion_matrix(_y_true, _y_pred)
    np.savetxt('Confusion_matrix.txt', cm, fmt='%.1f')
    
    TN = cm[0][0];
    FP = cm[0][1]; 
    FN = cm[1][0];  
    TP = cm[1][1];
    
    return cm 
        

def plot_roc_curve_binary(nb_classes, Y_test, y_pred):

    """
    Takes the true and the predicted probabilities and generates the ROC curve
    :param y_true true values 
    :param y_pred predicted values 
    """
    # compute ROC-AUC
    fpr = dict()
    tpr = dict()
    thr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
 
    
    plt.figure()  
    plt.plot(fpr[1], tpr[1], color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid(True) 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc.png')
    
    return roc_auc

    
def show_frame(img): 
    
    " show X-ray  "
    # plt.title('input', fontsize = 40)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.gray)
            
    plt.show()
    
config = readConfig()

s_config = config["segmentation_level"]


def cropImg(img, mask):
    vertical_sum = np.sum(mask, axis=0)
    horizontal_sum = np.sum(mask, axis=1)

    # Find non-zero elements (i.e., where the mask is active)
    indexes_v = np.nonzero(vertical_sum)[0]
    indexes_h = np.nonzero(horizontal_sum)[0]

    if len(indexes_v) == 0 or len(indexes_h) == 0:
        # If there are no non-zero elements in either dimension, return the original image
        return img

    border_l = indexes_v[0]
    border_r = indexes_v[-1]
    border_u = indexes_h[0]
    border_d = indexes_h[-1]
    
    if(len(indexes_v) > len(indexes_h)):
        size = len(indexes_v)
    else:
        size = len(indexes_h)
        
    horizontal_mid =border_l+int((border_r - border_l)/2)
    vertical_mid = border_u+int((border_d - border_u)/2)
    #if size is too small
    if size < 750:
        size = 750
    
    size = int(size*s_config)     
    horizontal_start = max(0, horizontal_mid - size)
    vertical_start = max(0, vertical_mid - size)    
        
    horizontal_end = min(img.shape[1], horizontal_mid + size)
    vertical_end = min(img.shape[0], vertical_mid + size)

    # Crop the image based on the calculated borders
    cropped_img = img[vertical_start:vertical_end, horizontal_start:horizontal_end]

    return cropped_img



def draw_rectangle(img, mask):
    " computes the non-zero borders, and draws a rectangle around the predicted lung area"
    " the area inside the rectangle can be sent to neural network to process"

    
    vertical_sum = np.sum(mask, axis = 0)
    horizontal_sum = np.sum(mask, axis = 1)
    
    # Find non-zero elements (i.e., where the mask is active)
    indexes_v = np.nonzero(vertical_sum)[0]
    indexes_h = np.nonzero(horizontal_sum)[0]

    border_l = indexes_v[0]
    border_r = indexes_v[-1]
    border_u = indexes_h[0]
    border_d = indexes_h[-1]
   
    
    if(len(indexes_v) > len(indexes_h)):
        size = len(indexes_v)
    else:
        size = len(indexes_h)
        
    horizontal_mid =border_l+int((border_r - border_l)/2)
    vertical_mid = border_u+int((border_d - border_u)/2)
    
    #if size is too small
    if size < 750:
        size = 750
        
        
    size = int(size*s_config)    
    horizontal_start = max(0, horizontal_mid - size)
    vertical_start = max(0, vertical_mid - size)    
        
    horizontal_end = min(img.shape[1], horizontal_mid + size)
    vertical_end = min(img.shape[0], vertical_mid + size)
    img = img*255
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.merge([img, img, img])
    
    start_point = (horizontal_start, vertical_start)
    end_point = (horizontal_end, vertical_end)
    
    color = (255, 255, 0)  # green color in BGR
    thickness = 5
    img = cv2.rectangle(img, start_point, end_point, color, thickness)
    return img

def show_segment(img, pred):
    # --- STEP 1: Standardize to 0-255 (uint8) ---
    img = img.astype('float32')
    
    img -= img.min()
    img /= (img.max() - img.min())
    
    # Now scale comfortably to 0-255 integers
    # This prevents "blank" images caused by float values being interpreted as 0
    img_disp = (img * 255).astype(np.uint8)
    
    # Setup Figure
    mpl.rcParams['figure.dpi'] = 150 
    plt.figure(figsize=(5,5))
    
    # 1. Original Image
    plt.subplot(131)
    plt.axis('off')
    plt.imshow(img_disp, cmap=plt.cm.gray)
    plt.title("Original")
    
    # 2. Mask
    plt.subplot(132)
    plt.axis('off')
    plt.imshow(pred, cmap='jet')
    plt.title("Mask")

    # 3. Crop Area
    # We pass the uint8 image to draw_rectangle. 
    # This ensures OpenCV functions inside it see values 0-255, not 0-1.
    try:
        img_rect = draw_rectangle(img_disp.copy(), pred)
        plt.subplot(133)
        plt.axis('off')
        plt.imshow(img_rect, cmap=plt.cm.gray)
        plt.title("Crop Area")
    except Exception as e:
        print(f"Drawing rectangle failed: {e}")

    # --- STEP 2: Non-Blocking Render ---
    plt.show()
    
    return

   
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def readXray(file):

    itk_image = sitk.ReadImage(file)
    img = sitk.GetArrayFromImage(itk_image)
    img = np.squeeze(img)

    # # if rgb, convert to grayscale
    # if(len(img.shape) == 3): 
    #     img = img[:,:,0]

    # # histogram equalization
    # img = exposure.equalize_hist(img)
    
        # # Median filter
        # img = median_filter(img, size=3)  # 3x3 median filter (you can adjust the filter size)

        # # Sharpening (Gaussian filter)
        # img = gaussian_filter(img, sigma=1)  # Adjust the sigma parameter for the desired sharpness

    # # Normalize
    # img -= img.mean()
    # img /= img.std()

    return img

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# Function to generate a unique name based on date, number of epochs, and number of training images
def generate_unique_name(base_name, epochs, train_images):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_name = f"{base_name}_{current_time}_{epochs}ep_{train_images}img"
    return unique_name

def create_out_folder(date, model_type):
    folder = "results/" + date + "/"+ model_type
    if not os.path.isdir(folder):
        os.makedirs(folder)
    models = os.path.join(folder, "models/")
    if not os.path.isdir(models):
        os.makedirs(models)
    plots = os.path.join(folder, "plots/")
    if not os.path.isdir(plots):
        os.makedirs(plots)
    report = folder

    return models, plots, report
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#Process Metadata.csv file for creating all_images.npz
def processMetadata():
    config_m = config["paths"]
    csv_file = os.path.join(config_m["PATH"],config_m["csv_df"])
    
    df = pd.read_csv(csv_file, encoding='utf-8')
    df['is_icu'] = df['is_icu'].replace(['FALSE'],int(0))
    df['is_icu'] = df['is_icu'].replace(['TRUE'],int(1))
    return df

def standardization(npy):
    npy = ((npy-npy.mean())/ npy.std())
    return npy

def normalize(npy):
    npy = ((npy-np.amin(npy))/(np.amax(npy)-np.amin(npy)))
    return npy
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def create_destroy_dic(path):
    if (not os.path.isdir(path)):
        os.makedirs(path, exist_ok=True)
    else:
        shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        
def save_numpy(data ,path ,i = None):
        if i == None:
            if not os.path.isfile(path + '.npy'):
                np.save(path + '.npy', data)
                return
            else:
                i = 1
                save_numpy(data ,path, i)
        else:
            name = path + "(" + str(i) + ")" + '.npy'
            if not os.path.isfile(name):
                np.save(name, data)
                return
            else:
                i = i + 1
                save_numpy(data ,path, i)