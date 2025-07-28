
"""
Created on 17 Mar 2025

@author: Akaqox(Salih KIZILIŞIK)

This script is experimental codes of paper "Explainable Multimodal Machine Learning Model
for Predicting Intensive Care Unit Admission " by S.Kizilisik et al. You may use the codes only for research. 
Plese cite the paper if you use any part of the codes.

"""

import cv2
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
# Display
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import create_destroy_dic, normalize


class XAI():
    def __init__(self, model, save_path, acc):
        self.save_path = save_path +"/gradcam/"
        create_destroy_dic(self.save_path)
        print(save_path)
        self.acc = acc
        self.pred = None
        self.true = None
        self.sampimg = cv2.imread("xr.png")
        self.model = model
        self.l_count = 0
        self.h_count = 0
        self.sampheatmap = None
        self.heatmap_l = np.zeros((7, 7, 3), dtype=np.float32)  
        self.heatmap_h = np.zeros((7, 7, 3), dtype=np.float32)
        self.last_conv_name = search_last_layer(self.model.layers)
        # self.last_conv_name = "block5_conv3"
        
    def __call__(self, img, i, true):
        if self.acc > 0.85:
            # print(img.shape)
            # data_uint8 = cv2.normalize(img[0,:,:,:], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # # Remove last layer's softmax
            # cv2.imwrite(str(i) + ".png", data_uint8)
            self.model.layers[-1].activation = None
            
            # Print what the top predicted class is
            self.pred = self.model.predict(img, verbose=0)
            
            heatmap = self.make_gradcam_heatmap(img)
            self.true = true
            self.overlay_heatmaps(heatmap)
            
            if isinstance(img, list):
                img = img[0]  # Only if it's a batch
    
                
            self.save_and_display_gradcam(img, heatmap, i)
        else:
            pass

    def overlay_heatmaps(self, heatmap):
        if self.pred < 0.20:
            self.l_count = self.l_count + 1
            if self.l_count == 1:
                self.heatmap_l = heatmap
                
            else:
                self.heatmap_l = self.heatmap_l * ((self.l_count )/(self.l_count + 1))
                heatmap = heatmap / (self.l_count + 1)
                self.heatmap_l = self.heatmap_l + heatmap
                
        elif self.pred > 0.80:
            self.h_count = self.h_count + 1
            if self.h_count == 1:
                self.heatmap_h = heatmap
                
            else:
                self.heatmap_h = self.heatmap_h * ((self.h_count)/(self.h_count + 1))
                heatmap = heatmap / (self.h_count + 1)
                self.heatmap_h = self.heatmap_h + heatmap
                
    def make_gradcam_heatmap(self, img_array, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        
        
        grad_model = keras.models.Model(
            self.model.inputs, [self.model.get_layer(self.last_conv_name).output, self.model.output]
        )
    
        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
    
        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)
    
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = np.nan_to_num(heatmap, nan=0, posinf=1, neginf=0)
        
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
    
        # Use jet colormap to colorize heatmap
        jet = mpl.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]  # Extract RGB colors
        jet_heatmap = jet_colors[heatmap]  # Apply colormap
    
        # Ensure heatmap has shape (height, width, 3)
        heatmap = np.squeeze(jet_heatmap)  
        
        return heatmap
    
    def plot_overlayed_heatmap(self, save, alpha = 0.35):
        if self.acc > 0.85:
            print("-")
            print(self.heatmap_h.shape)
            print(self.heatmap_l.shape)
            # Convert heatmap to image
            jet_heatmap_l = keras.utils.array_to_img(self.heatmap_l)
            jet_heatmap_h = keras.utils.array_to_img(self.heatmap_h)
            jet_heatmap_l = jet_heatmap_l.resize((self.sampimg.shape[1], self.sampimg.shape[0]))
            jet_heatmap_h = jet_heatmap_h.resize((self.sampimg.shape[1], self.sampimg.shape[0]))
            heatmap_l = keras.utils.img_to_array(jet_heatmap_l)
            heatmap_h = keras.utils.img_to_array(jet_heatmap_h)
            
            # Blend the heatmap with the original image
            heatmap_l = cv2.addWeighted(normalize(self.sampimg).astype(np.float32), 
                                        1 - alpha, 
                                        normalize(heatmap_l).astype(np.float32), 
                                        alpha, 0)
            
            heatmap_h = cv2.addWeighted(normalize(self.sampimg).astype(np.float32), 
                                        1 - alpha, 
                                        normalize(heatmap_h).astype(np.float32), 
                                        alpha, 0)
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            axes[0].imshow(heatmap_l)
            axes[0].set_title("Low Probability")
            axes[0].axis("off")
            
            axes[1].imshow(heatmap_h)
            axes[1].set_title("High Probability")
            axes[1].axis("off")
            
            # Show the plots
            plt.tight_layout()
            plt.savefig(save)
            plt.show()
            
        
        
    def save_and_display_gradcam(self, img, heatmap, cam_path, alpha=0.4):
        if (cam_path%20) == 0:
            img = np.squeeze(img)
        
            # Convert heatmap to image
            jet_heatmap = keras.utils.array_to_img(heatmap)
            jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
            jet_heatmap = keras.utils.img_to_array(jet_heatmap)
            
            # Blend the heatmap with the original image
            superimposed_img = cv2.addWeighted(normalize(img).astype(np.float32), 1 - alpha, normalize(jet_heatmap).astype(np.float32), alpha, 0)
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            axes[0].imshow(keras.utils.array_to_img(img))
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            
            axes[1].imshow(superimposed_img)
            axes[1].set_title("Original + Heatmap")
            axes[1].axis("off")
            
            # Save figure without showing it
            save_path = f"{self.save_path}/{cam_path}_pred{self.pred}_true_{self.true}.jpg"
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)  # Close the figure to free memory

def search_last_layer(layer_list):
    for layer in reversed(layer_list):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print("Found Last Conv2D Layer:", layer.name)
            return layer.name  # Return the layer name directly
        elif isinstance(layer, tf.keras.Model):  # If a nested model exists
            last_layer_name = search_last_layer(layer.layers)  # Recursively search inside
            if last_layer_name:  # If found inside, return it
                return last_layer_name
    return None  # Return None if no Conv2D layer is found


