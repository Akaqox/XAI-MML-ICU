import pandas as pd
import os
from utils.segmentation import Segment
from utils.utils import create_destroy_dic

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from dataset.dataloader import DataGenerator  # Assuming your class is in utils or paste it above
from tensorflow.keras import backend as K

import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage import exposure
from scipy.ndimage import median_filter, gaussian_filter

# Assuming these are available in your utils
from utils.utils import readConfig, readXray, cropImg, segmentLung, loadSegmentModel, show_segment, create_destroy_dic, standardization
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score as sk_f1_score  
from sklearn.metrics import classification_report
# --- CONFIGURATION ---
config = readConfig()
paths = config["paths"]
process_size = (224, 224) 
visualization = config["runningConfig"]["visualization_on"]
th = 0.40

# --- CONFIGURATION ---
CSV_PATH = "dataset/dataset_icu.csv"          # Input Metadata
OUTPUT_CSV = "dataset/dataset_icu_seg.csv"    # Output (Safety: Don't overwrite input yet)
SRC_DIR = "dataset/mdrc"                      # Source of unsegmented images (0/ and 1/ folders)
DEST_DIR = "dataset/mdrc_seg"                 # Destination for segmented images
VISUALIZATION = False                         # Set to True if you want to see images pop up
PROCESS_SIZE = (224, 224)


def process():
    print("--- STARTING CSV & SEGMENTATION UPDATE ---")

    # 1. Load the Master CSV
    # dtype=str is crucial to keep "1.2.826..." as a string, not a number
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH, dtype={'patient_id': str, 'icu_label': int})
    
    # Strip invisible spaces that break matching
    df['patient_id'] = df['patient_id'].str.strip()
    
    # Create a lookup dictionary: patient_id -> row data
    meta_map = df.set_index('patient_id').to_dict('index')
    print(f"Loaded metadata for {len(meta_map)} patients.")

    # 2. Initialize Segmenter (Assumed imported from your environment)
    try:
        seg = Segment()
        print("Segmenter loaded.")
    except NameError:
        print("Error: 'Segment' class not found. Make sure it is defined/imported before running.")
        return

    # 3. Prepare Output Directories
    os.makedirs(os.path.join(DEST_DIR, "0"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "1"), exist_ok=True)

    # 4. Get Source Files (These should be the unsegmented .npy files)
    # We look for files in both 0/ and 1/ source folders
    src_files_0 = glob.glob(os.path.join(SRC_DIR, "0", "*.npy"))
    src_files_1 = glob.glob(os.path.join(SRC_DIR, "1", "*.npy"))
    all_src_files = src_files_0 + src_files_1
    
    print(f"Found {len(all_src_files)} source files to process.")

    # 5. Process and Re-Index
    csv_data = []
    
    # We process in batches/lists to match your Segment class structure
    # But we need to carefully track filenames.
    
    # Let's map source path -> patient_id
    valid_files_0 = []
    valid_files_1 = []
    
    print("Validating file mappings...")
    for f_path in all_src_files:
        # CORRECT WAY to get ID: "1.2.8.npy" -> "1.2.8"
        # splitext splits at the LAST dot only.
        filename = os.path.basename(f_path)
        patient_id = os.path.splitext(filename)[0] 
        
        if patient_id in meta_map:
            # Check which class folder it belongs to based on CSV or Source? 
            # We trust the CSV label.
            label = meta_map[patient_id]['icu_label']
            if label == 0:
                valid_files_0.append(f_path)
            else:
                valid_files_1.append(f_path)
        else:
            print(f"Skipping {patient_id} (Not found in CSV)")

    # 6. Run Segmentation on VALID files only
    if valid_files_0:
        print(f"Processing {len(valid_files_0)} files for Class 0...")
        seg.cropAll(valid_files_0, 0, DEST_DIR)
        
    if valid_files_1:
        print(f"Processing {len(valid_files_1)} files for Class 1...")
        seg.cropAll(valid_files_1, 1, DEST_DIR)

    # 7. Build the New CSV Table
    # We iterate over the *DESTINATION* files to ensure we only list what actually exists
    dest_files = glob.glob(os.path.join(DEST_DIR, "**", "*.npy"), recursive=True)
    
    print(f"Building final CSV from {len(dest_files)} segmented files...")
    
    for f_path in dest_files:
        filename = os.path.basename(f_path)
        patient_id = os.path.splitext(filename)[0]
        
        # Get metadata
        if patient_id in meta_map:
            row = meta_map[patient_id]
            csv_data.append({
                'patient_id': patient_id,
                'original_label': row.get('original_label', 'Unknown'),
                'icu_label': row.get('icu_label', 0),
                'processed_path': f_path
            })
        else:
            # This should rarely happen if we filtered correctly above
            print(f"Warning: Orphan file found {filename}")

    # 8. Save
    if csv_data:
        df_final = pd.DataFrame(csv_data)
        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"DONE. Saved {len(df_final)} rows to {OUTPUT_CSV}")
    else:
        print("Failed: No data rows were generated.")
def create_mean_reference(generator, sample_size=None):
    """
    Calculates the pixel-wise average of images in the generator.
    This creates a 'canonical' X-ray with the average contrast of the training set.
    """
    # 1. Access the file list directly from your loader
    # This ensures we are looking at the exact same 'train' dataset
    paths = generator.img_list
    
    total_files = len(paths)
    if sample_size is None or sample_size > total_files:
        sample_size = total_files

    # 2. Shuffle to get a representative random sample
    # (We operate on a copy so we don't mess up the actual loader order)
    subset_paths = np.copy(paths)
    np.random.shuffle(subset_paths)
    subset_paths = subset_paths[:sample_size]
    
    print(f"Computing robust reference from {len(subset_paths)} training images...")

    accumulated_img = None
    count = 0

    for i, path in enumerate(subset_paths):
        try:
            # 3. Load RAW Image (Bypass Generator's normalization)
            img = np.load(path)
            # Ensure 2D (H, W)
            if img.ndim == 3: img = img[:,:,0]
            
            # Initialize accumulator on first valid image
            if accumulated_img is None:
                accumulated_img = np.zeros_like(img, dtype=np.float64)
            
            # Accumulate
            accumulated_img += img
            count += 1
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(subset_paths)}...", end='\r')
                
        except Exception as e:
            print(f"Skipped bad file {path}: {e}")

    # 4. Calculate Mean
    if count == 0:
        raise ValueError("No images were loaded!")
        
    mean_reference = accumulated_img / count
    print(f"\nDone. Computed mean from {count} images.")
    
    return mean_reference.astype(np.float32)

   
#process()

CSV_PATH = "dataset/dataset_icu_seg.csv"
MODEL_PATH = "model/cnn_mnet.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 1 

print("\n=== DEBUGGING DATA MISMATCH ===")


# --- DEBUGGING BLOCK ---
print("\n" + "="*40)
print("DEBUG: CHECKING LABEL CONSISTENCY")
print("="*40)

df = pd.read_csv(CSV_PATH)
# 1. Get Unique Labels from your CSV/DataFrame
unique_labels = df['original_label'].unique()
print(f"Found {len(unique_labels)} unique labels in CSV:")

for i, label in enumerate(unique_labels):
    # repr() shows hidden characters like \xa0 or double spaces
    print(f"  [{i}] Raw: '{label}'  |  Code (repr): {repr(label)}")

print("-" * 40)

# 2. Check against your manual list
severity_order = [
    "Negative for Pneumonia",
    "Mild Opacities  (1-2 lung zones)",
    "Moderate Opacities (3-4 lung zones)",
    "Severe Opacities (>4 lung zones)"
]

print("Checking for mismatches against 'severity_order'...")
found_labels = set(df['original_label'].values)

for ordered_label in severity_order:
    if ordered_label in found_labels:
        print(f"  [MATCH] Found: '{ordered_label}'")
    else:
        print(f"  [FAIL]  MISSING in CSV: '{ordered_label}'")
        print(f"          (Did you mean one of the labels printed above?)")

print("="*40 + "\n")
# -----------------------
# --- 1. DEFINE CUSTOM METRIC ---
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# --- 2. PREPARE DATAFRAME ---
print(f"Loading metadata from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# FIX: Standardized ID Extraction (Preserves 1.2.826...)
if 'processed_path' in df.columns:
    df['to_patient_id'] = df['processed_path'].apply(lambda x: os.path.splitext(os.path.basename(str(x)))[0])
else:
    print("Error: 'processed_path' column missing.")
    exit()

df = df.rename(columns={'icu_label': 'is_icu', 'label': 'is_icu'})
df['to_patient_id'] = df['to_patient_id'].astype(str).str.strip()

print(f"Loaded {len(df)} rows.")

hypp = config["parameters"]
ref_creator = DataGenerator(
    dataset="train",
    df= pd.read_csv(config["paths"]["dataset"] + "imputed.csv"),
    model_name = "mobilenet",
    IMG_SIZE = hypp["IMG_SIZE"], 
    batch_size = hypp["batch_size"]
)

robust_ref_img = create_mean_reference(ref_creator, sample_size=5000)

plt.figure(figsize=(5, 5))
plt.imshow(robust_ref_img, cmap='gray')
plt.title("Generated Robust Reference\n(Average of Training Set)")
plt.axis('off')
plt.show()

np.save("dataset/reference.npy", robust_ref_img)

# --- 3. INITIALIZE GENERATOR ---
test_gen = DataGenerator(
    dataset=DEST_DIR,  
    df=df, 
    model_name="mobilenet", 
    IMG_SIZE=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    shuffle=False,      
    inference_mode=True 
)

print(f"Generator found {len(test_gen.img_list)} images.")

# --- 4. LOAD MODEL ---
print(f"Loading model {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'f1_score': f1_score})
except Exception as e:
    print(f"Warning: Loading without custom metric ({e})")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# --- 5. INFERENCE ---
print("Running Inference...")
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = (y_pred_probs > th).astype(int).flatten()

# Get True Labels
y_true = []
for i in range(len(test_gen)):
    _, batch_y = test_gen[i]
    y_true.extend(batch_y)
y_true = np.array(y_true)

# --- 6. ALIGN LABELS MANUALLY (With Custom Remapping) ---
print("Aligning labels manually & Remapping Classes...")

detailed_labels = []
y_true = [] 

# Create Lookup Dictionary for TEXT labels only
id_to_detailed = dict(zip(df['to_patient_id'], df['original_label']))

found_count = 0
unknown_count = 0

for file_path in test_gen.img_list:
    # 1. ID Extraction
    pid = os.path.splitext(os.path.basename(file_path))[0]
    
    # 2. Get Detailed Label
    d_label = id_to_detailed.get(pid, "Unknown")
    detailed_labels.append(d_label)
    
    # 3. FORCE BINARY MAPPING (The Fix)
    # We map based on the string, ignoring the CSV 'is_icu' column
    if d_label in ["Negative for Pneumonia", "Mild Opacities  (1-2 lung zones)"]:
        b_label = 0  # Non-ICU
    elif d_label in ["Moderate Opacities (3-4 lung zones)", "Severe Opacities (>4 lung zones)"]:
        b_label = 1  # ICU
    else:
        # Fallback for Unknown or mismatched strings
        # (defaults to 0, or you can print a warning)
        b_label = 0 
        
    y_true.append(b_label)
    
    if d_label == "Unknown":
        unknown_count += 1
    else:
        found_count += 1

detailed_labels = np.array(detailed_labels)
y_true = np.array(y_true) 

print(f"DEBUG: Label Matching -> Found: {found_count} | Unknown: {unknown_count}")
print(f"DEBUG: New Class Distribution -> Class 0: {np.sum(y_true==0)} | Class 1: {np.sum(y_true==1)}")


print("\n" + "="*40)
print("CLASSIFICATION METRICS")
print("="*40)

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"F1 Score:  {sk_f1_score(y_true, y_pred):.4f}") 
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print("-" * 40)
print("Detailed Report:")

# ADDED digits=3 HERE
print(classification_report(y_true, y_pred, target_names=['Non-ICU', 'ICU'], digits=3))

print("="*40 + "\n")

# B. Plot Matrices
sns.set_context("talk", font_scale=1.1) 

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 1. Binary Matrix
cm_binary = confusion_matrix(y_true, y_pred)
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            annot_kws={"size": 18, "weight": "bold"},
            xticklabels=['Non-ICU', 'ICU'], yticklabels=['Non-ICU', 'ICU'])

axes[0].set_title('Binary Confusion Matrix', fontsize=20, fontweight='bold', pad=20)
axes[0].set_ylabel('True Label', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=16, fontweight='bold')

# 2. Detailed Matrix
cm_detailed = pd.crosstab(
    pd.Series(detailed_labels, name='Original'),
    pd.Series(y_pred, name='Predicted'),
    dropna=False
)
cm_detailed.columns = ['Pred Non-ICU', 'Pred ICU']

severity_order = [
    "Negative for Pneumonia",
    "Mild Opacities  (1-2 lung zones)",      
    "Moderate Opacities (3-4 lung zones)",
    "Severe Opacities (>4 lung zones)"
]

cm_detailed = cm_detailed.reindex(severity_order).fillna(0).astype(int)

sns.heatmap(cm_detailed, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1],
            annot_kws={"size": 16})

axes[1].set_title('Detailed Severity Analysis', fontsize=20, fontweight='bold', pad=20)
axes[1].set_ylabel('Clinical Severity', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Model Prediction', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()
sns.set_context("notebook")


global_idx = 0
num_per_class = 5
buckets = {
    "True Neg": [], "True Pos": [], 
    "False Pos": [], "False Neg": []
}

# Loop through the Generator batches
for batch_idx in range(len(test_gen)):
    # Get processed batch (X) directly from loader
    # We ignore the batch_y from loader because we want your FIXED y_true
    X_batch, _ = test_gen[batch_idx]
    
    # Predict on the whole batch (faster)
    probs_batch = model.predict(X_batch, verbose=0)
    
    # Loop through items in the batch
    for j in range(len(X_batch)):
        if global_idx >= len(y_true): break
        
        # 1. Get Data
        img_processed = X_batch[j]      # This is the (224,224,3) input
        prob = probs_batch[j][0]
        pred = 1 if prob > th else 0
        true = int(y_true[global_idx])  # Use the manual fixed label
        
        global_idx += 1 # Increment for next image
        
        # 2. Categorize
        key = None
        if true == 0 and pred == 0: key = "True Neg"
        elif true == 1 and pred == 1: key = "True Pos"
        elif true == 0 and pred == 1: key = "False Pos"
        elif true == 1 and pred == 0: key = "False Neg"
        
        # 3. Add to Bucket (if space exists)
        if key and len(buckets[key]) < num_per_class:
            # We take the first channel for visualization (since it's grayscale repeated)
            # Img is likely standardized (float), imshow handles this by scaling min-max
            vis_img = img_processed[:, :, 0] 
            buckets[key].append((vis_img, prob))
            
    # Stop early if all buckets are full
    if all(len(v) >= num_per_class for v in buckets.values()):
        break

# --- PLOTTING ---
fig, axes = plt.subplots(4, num_per_class, figsize=(12, 11))
fig.suptitle("Model Input Analysis (Processed Data)", fontsize=16)

keys = ["True Neg", "True Pos", "False Pos", "False Neg"]

for r, key in enumerate(keys):
    samples = buckets[key]
    for c in range(num_per_class):
        ax = axes[r, c]
        
        # Label the Row
        if c == 0:
            ax.set_ylabel(key, fontsize=12, fontweight='bold')
            
        if c < len(samples):
            img, conf = samples[c]
            # Plotting standardized data:
            # cmap='gray' automatically scales min-max for floats. 
            # If it looks too dark, the standardization might have outliers.
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Conf: {conf:.2f}", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No Sample", ha='center', va='center', color='gray')
            
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()