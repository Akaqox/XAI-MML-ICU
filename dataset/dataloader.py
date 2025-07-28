"""
Created on Tue Jan 7 11:40 2025

@authors: S.Kizilisik (kzlsksalih@gmail.com)


This script is experimental codes of paper " Explainable Multimodal Machine Learning Model
for Predicting Intensive Care Unit Admission" by S.Candemir et al. You may use the codes only for research. 
Plese cite the paper if you use any part of the codes.

"""

import os
import numpy as np
import glob
import tensorflow as tf
from utils.utils import standardization, normalize
from sklearn.preprocessing import StandardScaler

class DataGenerator(tf.keras.utils.Sequence):
    '''Generates data for Keras models'''

    def __init__(self, dataset, df, model_name, IMG_SIZE, batch_size=1, mask=False, n_channels=1, shuffle=True):
        '''Initialization'''
        super().__init__()
        self.IMG_SIZE = IMG_SIZE  # Resized volume dimensions (e.g., 244 x 244 x 3)
        self.batch_size = batch_size
        self.dataset = dataset
        self.df = df
        if model_name == "clinic" or model_name == "multi":
            
            self.exclude_cols = ["to_patient_id", "covid19_statuses", "is_icu"]
                
            numeric_cols = self.df.drop(columns=self.exclude_cols).select_dtypes(include=["number"]).columns
            self.tabular_continuous_cols = [col for col in numeric_cols if self.df[col].nunique() > 5]
            self.tabular_categorical_cols = [col for col in numeric_cols if col not in self.tabular_continuous_cols]
        
                
            self.scaler = StandardScaler()
            self.scaler.fit(self.df[self.tabular_continuous_cols])
            self.feature_means = self.scaler.mean_
            self.feature_stds = self.scaler.scale_
            
        
        if model_name == "multi":
            self.img_list = glob.glob("dataset/features/" + dataset + "/*/*.npy")
            
        else:
            self.img_list = glob.glob("dataset/ready_to_train/" + dataset + "/*/*.npy")
        
        

                
        self.model_name = model_name
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.img_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)  # Shuffle the indexes during initialization

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.img_list) / self.batch_size))  # Ensure integer division

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes for the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Get the corresponding file names for the batch
        batch_list = [self.img_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_list)
        
        
        return X, y

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.img_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def add_gaussian_noise(self, data: np.ndarray, noise_factor: float = 0.07, debug: bool = False) -> np.ndarray:
        """
        Adds Gaussian noise scaled by each feature's std.
        If debug=True, prints before/after comparison.
        """
        noise = np.random.randn(*data.shape) * (self.feature_stds * noise_factor)
        noisy_data = data + noise
    
        if debug:
            import pandas as pd
            original_df = pd.DataFrame(data, columns=self.tabular_continuous_cols)
            noisy_df = pd.DataFrame(noisy_data, columns=self.tabular_continuous_cols)
    
            print("\n--- Gaussian Noise Injection ---")
            print("Original:")
            print(original_df.round(3).T)
            print("Noisy:")
            print(noisy_df.round(3).T)
            print("--------------------------------")
    
        return noisy_data         
    def __data_generation(self, batch_list):
        X = []
        tabular = []
        y = []
        for path in batch_list:
        
            patient_id = os.path.basename(path)[:7]
            img = np.load(path)
            if self.model_name == "multi":
                img = img
            else:
                # plt.imshow(img, cmap="gray")
                # plt.show()
                
                # histogram equalization
                
                # Normalize
                img = standardization(img)
                
                img = np.repeat(img[..., np.newaxis], 3, -1)

            patient_data = self.df[self.df["to_patient_id"] == patient_id]
                
            # Get label
            label = int(patient_data["is_icu"].values)
                
                
            if self.model_name == "clinic" or self.model_name == "multi":
                    # Continuous columns (for noise)
                    # Get the original numeric columns in order (excluding dropped ones)
                    numeric_cols = self.df.drop(columns=self.exclude_cols).select_dtypes(include=["number"]).columns
                    
                    # Prepare empty array to hold the processed row
                    ordered_processed = np.zeros((1, len(numeric_cols)))  # shape: (1, total numeric columns)
                    
                    # --- Process continuous data ---
                    cont_data = patient_data[self.tabular_continuous_cols].values
                    if self.dataset not in ("test", "val"):
                        cont_data = self.add_gaussian_noise(cont_data, noise_factor=0.1)
                        
                    #standartization
                    cont_data = (cont_data - self.feature_means) / self.feature_stds
                    
                    # --- Process categorical data ---
                    cat_data = patient_data[self.tabular_categorical_cols].values
                    
                    # --- Insert processed values back in correct order ---
                    for i, col in enumerate(numeric_cols):
                        if col in self.tabular_continuous_cols:
                            idx = self.tabular_continuous_cols.index(col)
                            ordered_processed[0, i] = cont_data[0, idx]
                        else:
                            idx = self.tabular_categorical_cols.index(col)
                            ordered_processed[0, i] = cat_data[0, idx]
                    
                    # Optional: further standardization
                    #ordered_processed = standardization(ordered_processed)
                    
                    # Append to final list
                    tabular.append(np.squeeze(ordered_processed))
                    
            # if label == 1:
            #     onehot_label = [0, 1] 
            # else:
            #     onehot_label = [1, 0]
            
            
           
            X.append(img)
            # y.append(label)
            y.append(label)
            
        X = np.array(X)
        tabular = np.array(tabular)
        y = np.array(y)
        
        if self.model_name == "fine_tune" :
            return X, y
        elif self.model_name == "mobilenet" :
            return X, y
        elif self.model_name == "clinic":
            return tabular, y
        else:
            return [X, tabular], y
