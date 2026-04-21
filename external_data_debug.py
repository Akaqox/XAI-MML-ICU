"""

@author: Akaqox(Salih KIZILIŞIK)

This script is experimental codes of paper "Explainable Multimodal Machine Learning Model
for Predicting Intensive Care Unit Admission " by S.Kizilisik et al. You may use the codes only for research. 
Plese cite the paper if you use any part of the codes.
"""

import os
import cv2
import json
import glob
import pandas as pd
import numpy as np
import SimpleITK as sitk
from utils.utils import readXray
from utils.segmentation import Segment

# --- CONFIGURATION ---
IMAGE_FOLDER = "/home/ai1/Desktop/manifest-1610656454899/MIDRC-RICORD-1C"
JSON_FOLDER = "/home/ai1/Desktop/manifest-1610656454899/midrc-ricord-1c-da-other-json(1)"
OUTPUT_CSV = "dataset/dataset_icu.csv"
OUTPUT_DIR = "dataset/mdrc/"
TARGET_SIZE = (1525, 1270)  # (Width, Height)
ref_dataset = "dataset/Train"
seg = Segment()

import matplotlib.pyplot as plt

def ensure_directories():
    """Creates the folder structure: processed_dataset/0 and processed_dataset/1"""
    if os.path.exists(OUTPUT_DIR):
        print(f"Warning: '{OUTPUT_DIR}' already exists. Files might be overwritten.")
    else:
        os.makedirs(os.path.join(OUTPUT_DIR, "0")) # Non-ICU
        os.makedirs(os.path.join(OUTPUT_DIR, "1")) # ICU
    print(f"Created output directories in {OUTPUT_DIR}")

def get_study_map(image_dir):
    """Indexes DICOM files by Study UID using SimpleITK."""
    print(f"Indexing DICOM files in {image_dir}...")
    study_map = {}
    
    # Check both lowercase and uppercase extensions
    files = glob.glob(os.path.join(image_dir, '**', '*.dcm'), recursive=True)
    files += glob.glob(os.path.join(image_dir, '**', '*.DCM'), recursive=True)
    
    print(f"DEBUG: Found {len(files)} DICOM files.")
    
    reader = sitk.ImageFileReader()
    # 0020|000d is the DICOM tag for Study Instance UID
    study_uid_tag = "0020|000d" 

    for f_path in files:
        try:
            reader.SetFileName(f_path)
            reader.ReadImageInformation() # fast; reads header only

            if reader.HasMetaDataKey(study_uid_tag):
                uid = reader.GetMetaData(study_uid_tag).strip()
                
                if uid not in study_map: 
                    study_map[uid] = []
                study_map[uid].append(f_path)
        except Exception:
            # Skip files that aren't valid DICOMs
            pass
            
    print(f"DEBUG: Indexed {len(study_map)} unique studies.")
    return study_map

def get_label_dictionary(json_data):
    """Maps L_codes to text names."""
    label_map = {}
    for group in json_data.get('labelGroups', []):
        for label in group.get('labels', []):
            label_map[label['id']] = label['name']
    return label_map

def determine_icu_class(label_text):
    """
    Severe -> 1 (ICU)
    Mild/Moderate/Negative -> 0 (Non-ICU)
    """
    text = label_text.lower()
    if "severe" in text or "mild" in text:
        return 1
    elif "moderate" in text or "negative" in text:
        return 0
    return None

def apply_nyul_udupa(external_img, external_mask, ref_landmarks, percentiles):
    # Ensure both arrays are strictly 2D (H, W) by stripping any dummy channel dimensions
    external_img = np.squeeze(external_img)
    external_mask = np.squeeze(external_mask)

    # Ensure 0 and 100 anchors exist
    if percentiles[0] != 0:
        percentiles = np.insert(percentiles, 0, 0)
        ref_landmarks = np.insert(ref_landmarks, 0, ref_landmarks[0]) 
    if percentiles[-1] != 100:
        percentiles = np.append(percentiles, 100)
        ref_landmarks = np.append(ref_landmarks, ref_landmarks[-1])

    # Now the shapes match, so this line will work perfectly
    tissue_vals = external_img[external_mask > 0]
    
    if len(tissue_vals) == 0:
        print("DIAGNOSTIC: Mask is empty! No tissue found.")
        return external_img
        
    ext_landmarks = np.percentile(tissue_vals, percentiles)
    standardized_vals = np.interp(tissue_vals, ext_landmarks, ref_landmarks)

    standardized_img = external_img.astype(np.float32)
    standardized_img[external_mask > 0] = standardized_vals
    
    return standardized_img

def process_dataset():
    ensure_directories()
    study_map = get_study_map(IMAGE_FOLDER)
    json_files = glob.glob(os.path.join(JSON_FOLDER, '*.json'))
    if not json_files:
        print("No JSON found.")
        return
    
    count = 0
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        label_map = get_label_dictionary(data)
        
        for ds in data.get('datasets', []):
            for annot in ds.get('annotations', []):
                study_uid = annot.get('StudyInstanceUID') or annot.get('studyInstanceUid')
                
                if study_uid and study_uid in study_map:
                    # Avoid duplicates
                    save_name = f"{study_uid}.npy"
                    if os.path.exists(os.path.join(OUTPUT_DIR, "0", save_name)) or \
                       os.path.exists(os.path.join(OUTPUT_DIR, "1", save_name)):
                        continue

                    # Get Label
                    raw_id = annot.get('labelId')
                    label_text = label_map.get(raw_id, "Unknown")
                    icu_class = determine_icu_class(label_text)
                    
                    if icu_class is not None:
                        # Get Image Path (Middle of stack)
                        images = study_map[study_uid]
                        img_path = images[len(images)//2]

                        arr = readXray(img_path)
                        
                        if arr is not None:
                            # Resize exactly as requested
                            arr = cv2.resize(arr, TARGET_SIZE)
                            arr = arr.astype(np.float32)
                            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 4095.0
                            # --------------------------------------

                            mask = seg.makeMask(arr)
                            mask = np.squeeze(mask)

                            # --- VISUALIZATION QA BLOCK ---
                            if count < 3:  # Adjust to see more or fewer images
                                tissue_vals_original = arr[mask > 0]
                                tissue_vals_harmonized = standardized_img[mask > 0]
                                
                                fig, axes = plt.subplots(1, 4, figsize=(20, 4))
                                
                                axes[0].imshow(np.squeeze(arr), cmap='gray')
                                axes[0].set_title("Original External Image")
                                axes[0].axis('off')
                                
                                axes[1].imshow(np.squeeze(mask), cmap='bone')
                                axes[1].set_title("Segmentor Mask")
                                axes[1].axis('off')
                                
                                axes[2].imshow(np.squeeze(standardized_img), cmap='gray')
                                axes[2].set_title("Harmonized Image")
                                axes[2].axis('off')
                                
                                axes[3].hist(tissue_vals_original.flatten(), bins=50, color='gray', alpha=0.5, label='Original')
                                axes[3].hist(tissue_vals_harmonized.flatten(), bins=50, color='teal', alpha=0.7, label='Harmonized')
                                axes[3].set_title("Tissue Distributions")
                                axes[3].legend()
                                
                                plt.tight_layout()
                                plt.show()
                            # ------------------------------

                            
                            arr = standardized_img
                        
                            # Save
                            save_path = os.path.join(OUTPUT_DIR, str(icu_class), save_name)
                            np.save(save_path, arr)
                            
                            count += 1
                            print(f"[{count}] Saved Class {icu_class}: {save_name} | Shape: {arr.shape}")

    print(f"\nDone. Processed {count} images.")

if __name__ == "__main__":
    # paths = glob.glob(ref_dataset + "/*/*.dcm")
    # create_mean_reference(paths, 2000)
    process_dataset()