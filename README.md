# XAI-MML-ICU: Explainable Multimodal Machine Learning for ICU Admission Prediction
This repository contains the experimental codebase for the research paper:

"Explainable Multimodal Machine Learning Model for Predicting Intensive Care Unit Admission"
by S.Kizilisik, A.Terzi, M.Koc and S.Candemir

---


**📄 Abstract**
Timely prediction of Intensive Care Unit (ICU) admission is crucial for optimizing clinical decision-making and resource management, especially in high-pressure healthcare settings. This study investigates the effectiveness of a multimodal machine learning framework that integrates imaging data and clinical data—including vital signs, laboratory results, and co-morbidities—to predict the ICU requirement of COVID-19 patients at the time of hospital admission. Utilizing a publicly available dataset, we implemented a pipeline that includes lung region segmentation, data preprocessing and augmentation, and feature learning via a pre-trained convolutional neural network architecture. The multimodal model, trained with focal loss to address class imbalance, achieved an area under the receiver operating characteristic curve of 0.96. To interpret the model’s decision, we used Gradient-weighted Class Activation Mapping to visualize salient lung regions and SHapley Additive exPlanations to assess the individual importance of clinical features. The most influential predictors included C-reactive protein, creatinine, eGFR, glucose, and symptom duration, consistent with findings from correlation analysis. The results validate the clinical relevance of our approach, which offers a transparent and effective tool for early ICU risk stratification using data commonly available upon admission.

---


**✨ Key Features & Methodology**
* **Multimodal Data Fusion:** Fusion of imaging data (Chest X-rays) and diverse clinical data (vital signs, lab results, co-morbidities).
* **Automated Lung Region Segmentation:** Preprocessing to isolate irrelevant lung areas from imaging data.
* **Robust Data Augmentation:** Techniques to enhance dataset diversity and model generalization.
* **Feature Learning:** Extracts powerful features from imaging data using the pre-trained **MobileNetV3Small Convolutional Neural Network (CNN) architecture**, specifically chosen for its efficiency, mobility, and portability.
* **Class Imbalance Handling:** Employs Focal Loss during training to effectively address imbalanced datasets, common in medical prediction tasks.
  
<div align="center">
  <img src="final_results/multimodal_arch.png" width="70%" />
</div>



**💡 Explainable AI (XAI):**

* **Gradient-weighted Class Activation Mapping (Grad-CAM):** For visualizing salient regions in Chest X-rays that contribute to the model's decision.

* **SHapley Additive exPlanations (SHAP):** For quantifying the individual importance and impact of clinical features on predictions.

* **Key Predictor Identification:** Identifies clinically relevant features such as C-reactive protein, creatinine, Sodium, glucose, and symptom duration as influential predictors.



<div align="center" style="display: inline-block;">
  <div style="width: 45%;">
    <img src="final_results/grad_multiple.png" width="75%" />
    <p align="center"><em>Figure: Grad-CAM visualizations of individual chest X-ray in test dataset. Subfigures (a–d) show correctly classified discharged patients with low ICU admission
probabilities (0.06, 0.16, 0.18, and 0.28, respectively). Minimal activation is observed in the lung regions, indicating absence of pathological findings. Subfigure
(d) includes visible medical equipment, which may have slightly influenced the model’s ICU admission probability towards 0.28. Subfigures (e–h) correspond
to correctly classified ICU-admitted patients, with ICU prediction probabilities (0.74, 0.75, 0.80, and 0.93, respectively). These heatmaps show prominent
activations in the lung areas, suggesting the model bases its decisions primarily on pulmonary pathology.</em></p>
  </div>
  
  ---
  
  <img src="final_results/Shapley.png" width="75%" />
</div>

# 🚀 Performance
## CXR Model

<div align="center" style="display: flex; gap: 10px; width: 60%; align-items: center; flex-wrap: nowrap; overflow-x: auto;">
  <img src="final_results/mobilenet/0874/img_plot.png" style="width: 44%; height: auto; margin:auto;" />
  <img src="final_results/mobilenet/0874/img_calibration.png" style="width: 22%; height: auto; margin:auto;" />
  <img src="final_results/mobilenet/0874/img_cm_plot.png" style="width: 22%; height: auto; margin:auto;" />
</div>

## Clinic Model

<div align="center" style="display: flex; gap: 10px; align-items: center;">
  <img src="final_results/clinical/clinical3/clinic_plot.png"  style="width: 44%; height: auto; margin:auto;" />
  <img src="final_results/clinical/clinical3/clinic_calibration.png"  style="width: 22%; height: auto; margin:auto;" />
  <img src="final_results/clinical/clinical3/clinic_cm_plot.png"  style="width: 22%; height: auto; margin:auto;" />
</div>

## Multimodal Fused Approach

<!-- First row: 3 images side by side -->
<div align="center" style="display: flex; gap: 10px; justify-content: center; margin-bottom: 20px;">
  <img src="final_results/multi_test/multiv6/multi_plot.png" style="width: 44%; height: auto; margin:auto;" />
  <img src="final_results/multi_test/multiv6/multi_calibration.png" style="width: 22%; height: auto; margin:auto;"/>
  <img src="final_results/multi_test/multiv6/multi_cm_plot.png" style="width: 22%; height: auto; margin:auto;" />
</div>


## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | AUC | Brier Score | 
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | 
| **Imaging-only Model (CNN)** | 0.874 | 0.938 | 0.879 | 0.907 | 0.928 | 0.150 |
| **Clinical-only Model (MLP)** | 0.779 | 0.926 | 0.744 | 0.825 | 0.835 | 0.177 |
| **Fused Model (opt.Thr)** | 0.889 | 0.952 | 0.886 | 0.918 | 0.951 | 0.102 | 
| **Fused Model (High Sens.)** | 0.874 | 0.879 | 0.950 | 0.913 | 0.951 | - |
| **Imaging-only (Ext., Unadj.)** | 0.697 | 0.693 | 0.698 | 0.696 | 0.770 | - |
| **Imaging-only (Ext., Adj.)** | 0.717 | 0.712 | 0.721 | 0.716 | 0.779 | - | 
| **Imaging-only (Ext., Neg/Sev)** | 0.811 | 0.743 | 0.797 | 0.769 | 0.892 | - | 

---

## Subgroup Performance (Fused Model)

| Subgroup | Accuracy | Precision | Recall | F1 Score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | 
| **Male** | 0.881 | 0.960 | 0.885 | 0.921 | 0.945 |
| **Female** | 0.909 | 0.917 | 0.893 | 0.905 | 0.962 |
| **Age < 60** | 0.887 | 0.970 | 0.877 | 0.921 | 0.958 |
| **Age ≥ 60** | 0.891 | 0.923 | 0.902 | 0.912 | 0.947 |
| **Comorbidity (Yes)** | 0.899 | 0.953 | 0.912 | 0.932 | 0.956 |
| **Comorbidity (No)** | 0.877 | 0.950 | 0.849 | 0.897 | 0.942 |

# 🛠️ Getting Started
Prerequisites can be found in env_backup.yaml file

Git

Installation
Clone the repository:
```
git clone git@github.com:Akaqox/ICU-MML-XAI.git
cd ICU-MML-XAI
```

Install dependencies:
It is highly recommended to use a virtual environment such as conda.
```

conda update -n base -c defaults conda
conda env create -f env_backup.yaml -n YOUR_ENV_NAME
conda activate YOUR_ENV_NAME


```
**Configuration:**
The project uses a config.json file for hyperparameters and paths. Ensure you review and update config["paths"]["PATH"] and other relevant settings before running.

Example config.json structure for first run
```
runningConfig:
  augmentation: 1
  segmentation: 1
  construct_dataset: 1
  train: 1
paths:
      "PATH": "",
    "dataset": "dataset/",

```
**Usage**
The main pipeline is controlled by the runningConfig flags in config.json. Set True for stages you wish to execute.

To run the full pipeline (or selected stages):
```
python main.py
```
The main.py script orchestrates the following stages based on your config.json:

Processing and Augmentation Stage: Loads data and performs data augmentation.
```
if runningConfig["augmentation"] == True:
    # ... augmentation logic
```
Segmentation Stage: Performs lung region segmentation and cropping (there some adjustable options in config file).
```
if runningConfig["segmentation"] == True:
    # ... segmentation logic
```
Dataset Construction Stage: Constructs the final dataset for training.
```
if runningConfig["construct_dataset"] == True:
    # ... dataset construction logic
```
Training Stage: Initiates the model training and evaluation process automatically. The provided code runs the training loop 50 times for experimental purposes and .
```
if runningConfig["train"] == True:
    for i in range(50):
        # train.fit()
```
**📊 Dataset**
COVID-19-NY-SBU
MIDRC-C

**📝 Citation**
This code is provided for research purposes only. If you use any part of this codebase or the methodology described, please cite the following paper:
```
(TO BE UPDATED)
@article{Kizilisik_ICU_Prediction,
  author={Kizilisik, S. and Terzi, A. and Koc, M and  Candemir, S.},
  title={Explainable Multimodal Machine Learning Model for Predicting Intensive Care Unit Admission},
  journal={Journal Name (e.g., IEEE Transactions on Medical Imaging)},
  year={Year of Publication}, # e.g., 2024
  volume={Volume Number},
  number={Issue Number},
  pages={Page Range},
  doi={DOI Link} # e.g., 10.1109/TMI.2024.XXXXXXX
}
```


# ✉️ Contact
Salih Kızılışık, Sema Candemir are with the Artificial Intelligence in
Healthcare Laboratory, Computer Engineering Department, Eskisehir Technical University, Eskişehir, TURKEY e-mail: salihk@ogr.eskisehir.edu.tr,
semacandemir@eskisehir.edu.tr

Corresponding author:SemaCandemir
(e-mail:semacandemir@eskisehir.edu.tr).


# ⚖️ License
This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.
