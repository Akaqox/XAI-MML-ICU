import os
import cv2
import json
import glob
import pandas as pd
import numpy as np
import SimpleITK as sitk
from utils.utils import readXray

# --- CONFIGURATION ---
IMAGE_FOLDER = "/home/ai1/Desktop/manifest-1610656454899/MIDRC-RICORD-1C"
JSON_FOLDER = "/home/ai1/Desktop/manifest-1610656454899/midrc-ricord-1c-da-other-json(1)"
OUTPUT_CSV = "dataset/dataset_icu.csv"
OUTPUT_DIR = "dataset/mdrc/"
TARGET_SIZE = (1525, 1270)  # (Width, Height)

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
                        
                        # --- YOUR EXACT LOGIC START ---
                        arr = readXray(img_path)
                        
                        if arr is not None:
                            # Resize exactly as requested
                            arr = cv2.resize(arr, TARGET_SIZE)
                            # --- YOUR EXACT LOGIC END ---
                            
                            # Save
                            save_path = os.path.join(OUTPUT_DIR, str(icu_class), save_name)
                            np.save(save_path, arr)
                            
                            count += 1
                            print(f"[{count}] Saved Class {icu_class}: {save_name} | Shape: {arr.shape}")

    print(f"\nDone. Processed {count} images.")

if __name__ == "__main__":
    process_dataset()