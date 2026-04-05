"""
Created on Tue Jun 25 21:41:16 2024

@author: Akaqox(Salih KIZILIŞIK), Ayşegül Terzi
"""

from utils.utils import readConfig, readXray, cropImg, segmentLung, loadSegmentModel, show_segment, create_destroy_dic
import os
import matplotlib.pyplot as plt
import numpy as np

from skimage import exposure
from scipy.ndimage import median_filter, gaussian_filter
import cv2
import glob

config = readConfig()
paths = config["paths"]


process_size = (224,224) 
visualization = config["runningConfig"]["visualization_on"]
        
class Segment():
    def __init__(self):
        self.seg_model = loadSegmentModel()
        
    def __call__(self):
        lists = []
        path_augmented = os.path.join(paths["PATH"], paths["augmented"])
        path_segmented = os.path.join(paths["PATH"], paths["segmented"])
        
        lists.append(glob.glob(path_augmented + "/0" + '/*.npy'))
        lists.append(glob.glob(path_augmented + "/1" + '/*.npy'))
        
        create_destroy_dic(path_segmented + "/0")
        create_destroy_dic(path_segmented + "/1")
        
        for i, npy_list in enumerate(lists):
            print(f"Segmentation started for class {i}")
            self.cropAll(npy_list, i, path_segmented)
            print("Segmentation finished")
            
    def cropAll(self, npy_list, i, save_path, dicom=False):
        for npy_path in npy_list:
            
            # 1. Define/Load the image
            if dicom == False:
                npy = np.load(npy_path)
                # Handle potential shape mismatch (e.g. if npy is (1, H, W, 1))
                if len(npy.shape) == 4:
                    img = npy[0, :, :, 0]
                elif len(npy.shape) == 3:
                    img = npy[0, :, :]
                else:
                    img = npy
            else:
                img = readXray(npy_path)
    
            # If rgb/channel dim exists, convert to grayscale 2D
            if len(img.shape) == 3: 
                img = img[:,:,0]
            
            # 2. Create Mask
            mask = self.makeMask(img)
            
            # --- VISUALIZATION FIX PART 1 ---
            # Use a non-blocking check for your custom show_segment function
            # (Ensure utils.show_segment does not contain plt.show() either)
            if visualization:
                show_segment(img, mask)

    
            # 3. Crop and Resize
            img = cropImg(img, mask)
            mask = cropImg(mask, mask)
            img = cv2.resize(img, process_size)
            mask = cv2.resize(mask, process_size)
            # --- VISUALIZATION FIX PART 2 ---
            if visualization:
                plt.figure(figsize=(3, 5))
                plt.imshow(img, cmap=plt.cm.gray)
                plt.title(os.path.basename(npy_path))
                plt.axis('off')
                
                # Instead of plt.show(), use pause so code continues
                plt.show()
            
            # 4. Save
            basename = os.path.basename(npy_path)
            if len(basename) < 10:
                basename = basename.split(".")[0]
            
            
            save = save_path + "/" + str(i) + "/" + str(basename)
            save_mask = save_path + "/" + str(i) + "/" +"mask_" + str(basename)
            print(save)
            np.save(save, img)
            np.save(save_mask, mask)
            #print(save)
            
            
            
    def makeMask(self, img):
        # Create a copy so we don't modify the original image used for cropping later
        img_proc = img.copy()
        
        # if rgb, convert to grayscale
        if len(img_proc.shape) == 3: 
            img_proc = img_proc[:,:,0]
    
        # histogram equalization
        img_proc = exposure.equalize_hist(img_proc)
        img_proc = median_filter(img_proc, size=3) 
        img_proc = gaussian_filter(img_proc, sigma=1)
        
        # Normalize
        if img_proc.std() != 0:
            img_proc -= img_proc.mean()
            img_proc /= img_proc.std()
    
        mask = segmentLung(self.seg_model, img_proc)
        
        # Ensure mask is binary uint8
        mask = (mask > 0.5).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
    
        # Label objects
        num_labels, labels = cv2.connectedComponents(mask)
    
        # Calculate object sizes
        if num_labels > 1:
            object_sizes = np.bincount(labels.flatten())
            # Ignore background (label 0) and get largest 2
            object_sizes[0] = 0 
            
            # Select largest two objects
            largest_objects_indices = np.argsort(object_sizes)[::-1][:2]
            
            # Create a mask for the largest objects
            largest_objects_mask = np.isin(labels, largest_objects_indices)
            
            # Apply filter
            mask[~largest_objects_mask] = 0
        
        return mask

