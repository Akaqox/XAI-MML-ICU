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
from skimage.exposure import rescale_intensity, match_histograms
from utils.utils import standardization, normalize
from sklearn.preprocessing import StandardScaler

def fourier_domain_adaptation(source_img, target_img, beta=0.05):
                        """
                        Applies Fourier Domain Adaptation.
                        Transfers the low-frequency style of target_img to source_img.
                        
                        Args:
                            source_img: 2D numpy array (External dataset / Structure).
                            target_img: 2D numpy array (Training dataset / Style).
                            beta: Float [0, 1]. Ratio of low frequencies to swap. 
                                Higher means more style transfer; typical values are 0.01 to 0.1.
                            
                        Returns:
                            adapted_img: 2D numpy array.
                        """
                        # 1. Apply Fast Fourier Transform and shift zero-frequency to the center
                        fft_src = np.fft.fftshift(np.fft.fft2(source_img))
                        fft_trg = np.fft.fftshift(np.fft.fft2(target_img))
                        
                        # 2. Extract Amplitude and Phase
                        amp_src, phase_src = np.abs(fft_src), np.angle(fft_src)
                        amp_trg = np.abs(fft_trg)
                        
                        # 3. Create the mask for low frequencies
                        h, w = source_img.shape
                        b = int(np.floor(min(h, w) * beta))
                        c_h, c_w = h // 2, w // 2
                        
                        mask = np.zeros((h, w))
                        mask[c_h - b : c_h + b + 1, c_w - b : c_w + b + 1] = 1
                        
                        # 4. Swap low-frequency amplitudes
                        amp_adapted = (1 - mask) * amp_src + mask * amp_trg
                        
                        # 5. Reconstruct the complex representation (Swapped Amplitude + Original Phase)
                        fft_adapted = amp_adapted * np.exp(1j * phase_src)
                        
                        # 6. Apply inverse shift and inverse FFT
                        fft_adapted_ishift = np.fft.ifftshift(fft_adapted)
                        adapted_img = np.fft.ifft2(fft_adapted_ishift)
                        
                        # Return the real part
                        return np.real(adapted_img)
                    
                    
class DataGenerator(tf.keras.utils.Sequence):
    '''Generates data for Keras models'''

    def __init__(self, dataset, df, model_name, IMG_SIZE, batch_size=1, mask=False, n_channels=1, shuffle=True, inference_mode=False, reference= None):
        '''Initialization'''
        super().__init__()
        self.inference_mode = inference_mode
        self.IMG_SIZE = IMG_SIZE
        self.batch_size = batch_size
        self.dataset = dataset
        self.df = df
        if reference is None and os.path.isfile("dataset/reference.npy"):
            self.reference_img = standardization(np.load("dataset/reference.npy"))

        
        if model_name == "clinic" or model_name == "multi":
            self.exclude_cols = ["to_patient_id", "is_icu"]
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
        if inference_mode == True:
            self.img_list = glob.glob(dataset + "/*/*.npy")
            self.img_list = [f for f in self.img_list if "mask" not in f]
        self.model_name = model_name
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.img_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.img_list) / self.batch_size))

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
            
    def add_gaussian_noise(self, data: np.ndarray, noise_factor: float = 0.05, debug: bool = False) -> np.ndarray:
        """
        Adds Gaussian noise scaled by each feature's std.
        """
        noise = np.random.randn(*data.shape) * (self.feature_stds * noise_factor)
        noisy_data = data + noise
        return noisy_data         

    def __data_generation(self, batch_list):
        X = []
        tabular = []
        y = []
        for path in batch_list:
            
            # --- FIX: Robust ID Extraction ---
            if self.inference_mode:
                # Use full filename (minus .npy) as ID
                patient_id = os.path.basename(path)

            else:
                # Keep original logic
                patient_id = os.path.basename(path)[:7]

            img = np.load(path)
            
            if self.model_name == "multi":
                pass
            else:
                # Normalize
                img = standardization(img)
                
            
                if self.inference_mode:

                    og_img = img.copy()
                    
                    mask = np.load(os.path.dirname(path) + "/mask_"+ os.path.basename(path))
                    
                    img_min, img_max = img.min(), img.max()
                    
                    
                    # 2. Swap arguments so 'img' provides the structure. Increase beta.
                    img = fourier_domain_adaptation(img, self.reference_img, beta=0.005)

                    # 3. Clip the overshoot to prevent visual distortion
                    img = np.clip(img, img_min, img_max)
                    
                    img = match_histograms(img, self.reference_img)
                img = np.repeat(img[..., np.newaxis], 3, -1)

            # Retrieve patient metadata
            patient_data = self.df[self.df["to_patient_id"] == patient_id]
                
            # --- FIX: Safe Label Extraction ---
            vals = patient_data["is_icu"].values
            if len(vals) > 0:
                label = int(vals[0])
            else:
                label = 0 # Default fallback
                
            if self.model_name == "clinic" or self.model_name == "multi":
                numeric_cols = self.df.drop(columns=self.exclude_cols).select_dtypes(include=["number"]).columns
                ordered_processed = np.zeros((1, len(numeric_cols)))
                
                cont_data = patient_data[self.tabular_continuous_cols].values
                # Handle duplicates
                if cont_data.shape[0] > 1: cont_data = cont_data[:1] 

                if self.dataset not in ("test", "val") :
                    cont_data = self.add_gaussian_noise(cont_data, noise_factor=0.1)
                
                cont_data = (cont_data - self.feature_means) / self.feature_stds
                
                cat_data = patient_data[self.tabular_categorical_cols].values
                if cat_data.shape[0] > 1: cat_data = cat_data[:1] 
                
                for i, col in enumerate(numeric_cols):
                    if col in self.tabular_continuous_cols:
                        idx = self.tabular_continuous_cols.index(col)
                        ordered_processed[0, i] = cont_data[0, idx]
                    else:
                        idx = self.tabular_categorical_cols.index(col)
                        ordered_processed[0, i] = cat_data[0, idx]
                
                tabular.append(np.squeeze(ordered_processed))
            
            X.append(img)
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