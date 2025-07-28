"""
Created on Tue Jun 25 21:41:16 2024

@author: Akaqox(Salih KIZILIŞIK), Ayşegül Terzi
"""

from utils.utils import readConfig, readXray, cropImg, segmentLung, loadSegmentModel, show_segment, create_destroy_dic
import os
import matplotlib.pyplot as plt
import numpy as np

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
        for i ,npy_list in enumerate(lists):
            
            print("Segmentation started")
            self.cropAll(npy_list, i, path_segmented)
            print("Segmentation finished")
            
    def cropAll(self, npy_list, i, save_path, dicom=False):
        for npy_path in npy_list:
            #define the image
            if dicom == False:
                npy = np.load(npy_path)
                img = npy[0, : ,: , 0]
            else:
                img = readXray(npy_path)
    
            # if rgb, convert to grayscale
            if(len(img.shape) == 3): 
                img = img[:,:,0]
            
            mask = self.makeMask(img)
            
            if visualization:
                show_segment(img, mask)
    
            img = cropImg(img, mask)
            img = cv2.resize(img, process_size)
     
            # print(np.amax(img))
            # print(np.amin(img))
            
            if visualization:
                plt.imshow(img, cmap=plt.cm.gray)
                plt.show()
                plt.clf()
            basename = os.path.basename(npy_path)
            basename = basename.split(".")[0]
            save = save_path + "/"+ str(i) + "/" + basename
            np.save(save, img)
            
        
    def makeMask(self, img):
    
        # if rgb, convert to grayscale
        if(len(img.shape) == 3): 
            img = img[:,:,0]
    
        # histogram equalization
        img = exposure.equalize_hist(img)
        img = median_filter(img, size=3)  # 3x3 median filter (you can adjust the filter size)
        # Sharpening (Gaussian filter)
        img = gaussian_filter(img, sigma=1)  # Adjust the sigma parameter for the desired sharpness
        # # Normalize
        img -= img.mean()
        img /= img.std()
    
        mask = segmentLung(self.seg_model, img)
        
        # Assuming `mask` is a numpy array representing your binary image
        mask = mask.astype(np.uint8)
    
        # Step 1: Label objects
        num_labels, labels = cv2.connectedComponents(mask)
    
        # Step 2: Calculate object sizes
        object_sizes = np.bincount(labels.flatten())
    
        # Step 3: Select largest two objects
        largest_objects_indices = np.argsort(object_sizes)[::-1][1:3]  # Indices of largest two objects
    
        # Step 4: Create a mask for the largest objects
        largest_objects_mask = np.isin(labels, largest_objects_indices)
    
        # Step 5: Apply the mask to the original image
        result_image = mask.copy()
        result_image[~largest_objects_mask] = 0  # Keep only the largest objects, set others to 0
    
        # If you want to keep `mask` variable updated as well, do:
        mask[~largest_objects_mask] = 0
        
        return mask

