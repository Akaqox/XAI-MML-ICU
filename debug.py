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
landmarks_path = "dataset/mean_landmarks.npy"
percentiles_path = "dataset/percentiles.npy"
import matplotlib.pyplot as plt

def create_mean_reference(paths, sample_size=None, num_visualize=3):
    """
    Calculates the Nyul-Udupa statistical reference (average percentiles)
    from the training set, strictly using lung tissue. 
    Also extracts a single 'Golden Image' for Fourier Domain Adaptation.
    Tracks and prints the intensity domain of the dataset.
    """
    total_files = len(paths)
    if sample_size is None or sample_size > total_files:
        sample_size = total_files

    subset_paths = np.copy(paths)
    np.random.seed(42) 
    np.random.shuffle(subset_paths)
    subset_paths = subset_paths[:sample_size]
    
    print(f"Computing Nyul-Udupa landmarks from {len(subset_paths)} training images...")

    all_landmarks = []
    ref_percentiles = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]) 
    
    golden_img = None
    golden_mask = None
    count = 0

    # --- DOMAIN TRACKING VARIABLES ---
    global_img_min = float('inf')
    global_img_max = float('-inf')
    global_tissue_min = float('inf')
    global_tissue_max = float('-inf')
    # ---------------------------------

    for i, path in enumerate(subset_paths):
        try:
            arr = readXray(path)
            if arr is not None:
                arr = cv2.resize(arr, TARGET_SIZE)
                
            img = np.repeat(arr[..., np.newaxis], 1, -1)
            
            # --- TRACK RAW IMAGE DOMAIN ---
            global_img_min = min(global_img_min, img.min())
            global_img_max = max(global_img_max, img.max())
            
            mask = seg.makeMask(img)
            tissue_vals = img[mask > 0]
            
            if len(tissue_vals) == 0:
                continue
                
            # --- TRACK TISSUE DOMAIN ---
            global_tissue_min = min(global_tissue_min, tissue_vals.min())
            global_tissue_max = max(global_tissue_max, tissue_vals.max())
                
            if count < num_visualize:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                axes[0].imshow(np.squeeze(img), cmap='gray')
                axes[0].set_title(f"Raw Image {count+1}")
                axes[0].axis('off')
                
                axes[1].imshow(np.squeeze(mask), cmap='bone')
                axes[1].set_title("Segmentor Mask")
                axes[1].axis('off')
                
                axes[2].hist(tissue_vals.flatten(), bins=50, color='teal', alpha=0.7)
                axes[2].set_title("Isolated Tissue Distribution")
                axes[2].set_xlabel("Pixel Intensity")
                axes[2].set_ylabel("Frequency")
                
                plt.tight_layout()
                plt.show() 

            if golden_img is None:
                golden_img = img.copy()
                golden_mask = mask.copy()
            
            landmarks = np.percentile(tissue_vals, ref_percentiles)
            all_landmarks.append(landmarks)
            count += 1
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(subset_paths)}...", end='\r')
                
        except Exception as e:
            print(f"Skipped bad file {path}: {e}")

    if count == 0:
        raise ValueError("No images were loaded or segmented successfully!")
        
    ref_landmarks = np.mean(all_landmarks, axis=0)
    
    # --- PRINT DOMAIN RESULTS ---
    print(f"\nDone. Computed statistical reference from {count} images.")
    print("-" * 40)
    print("DATASET INTENSITY DOMAIN:")
    print(f"Raw Image Bounds : [{global_img_min:.2f}, {global_img_max:.2f}]")
    print(f"Lung Tissue Bounds: [{global_tissue_min:.2f}, {global_tissue_max:.2f}]")
    print("-" * 40)
    print("Mean Landmarks (Percentiles):")
    print(np.round(ref_landmarks, 2))
    print("-" * 40)
    
    np.save(landmarks_path, ref_landmarks)
    np.save(percentiles_path, ref_percentiles)
    
    return ref_landmarks, ref_percentiles, golden_img, golden_mask

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

    # --- THE FIX: Upgrade memory capacity to prevent static ---
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
    ref_landmarks = np.load(landmarks_path)
    ref_percentiles = np.load(percentiles_path)
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
                            arr = arr.astype(np.float32)
                            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 4095.0
                            # --------------------------------------
                            # --- YOUR EXACT LOGIC END ---
                            mask = seg.makeMask(arr)
                            mask = np.squeeze(mask)
                            
                            standardized_img = apply_nyul_udupa(arr, mask, ref_landmarks, ref_percentiles)
                                
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