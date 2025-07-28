# ICU-MML-XAI: Explainable Multimodal Machine Learning for ICU Admission Prediction
This repository contains the experimental codebase for the research paper:

"Explainable Multimodal Machine Learning Model for Predicting Intensive Care Unit Admission"
by S.Kizilisik, S.Candemir, A.Terzi and M.Koc

📄 Abstract
Timely prediction of Intensive Care Unit (ICU) admission is crucial for optimizing clinical decision-making and resource management, especially in high-pressure healthcare settings. This study investigates the effectiveness of a multimodal machine learning framework that integrates imaging data and clinical data—including vital signs, laboratory results, and co-morbidities—to predict the ICU requirement of COVID-19 patients at the time of hospital admission. Utilizing a publicly available dataset, we implemented a pipeline that includes lung region segmentation, data preprocessing and augmentation, and feature learning via a pre-trained convolutional neural network architecture. The multimodal model, trained with focal loss to address class imbalance, achieved an area under the receiver operating characteristic curve of 0.93. To interpret the model’s decision, we used Gradient-weighted Class Activation Mapping to visualize salient lung regions and SHapley Additive exPlanations to assess the individual importance of clinical features. The most influential predictors included C-reactive protein, creatinine, eGFR, glucose, and symptom duration, consistent with findings from correlation analysis. The results validate the clinical relevance of our approach, which offers a transparent and effective tool for early ICU risk stratification using data commonly available upon admission.

✨ Key Features & Methodology
This project implements a comprehensive multimodal machine learning pipeline for ICU admission prediction, featuring:

Multimodal Data Fusion: Seamless integration of imaging data (Chest X-rays) and diverse clinical data (vital signs, lab results, co-morbidities).

Automated Lung Region Segmentation: Preprocessing to isolate relevant lung areas from imaging data.

Robust Data Augmentation: Techniques to enhance dataset diversity and model generalization.

Feature Learning: Utilizes a pre-trained Convolutional Neural Network (CNN) architecture for extracting powerful features from imaging data.

Class Imbalance Handling: Employs Focal Loss during training to effectively address imbalanced datasets, common in medical prediction tasks.

Explainable AI (XAI):

Gradient-weighted Class Activation Mapping (Grad-CAM): For visualizing salient regions in Chest X-rays that contribute to the model's decision.

SHapley Additive exPlanations (SHAP): For quantifying the individual importance and impact of clinical features on predictions.

Key Predictor Identification: Identifies clinically relevant features such as C-reactive protein, creatinine, eGFR, glucose, and symptom duration as influential predictors.

🚀 Performance
The multimodal model achieved an Area Under the Receiver Operating Characteristic Curve (AUC) of 0.96, demonstrating strong predictive capabilities for ICU admission.

🛠️ Getting Started
Prerequisites can be found in env_backup.yaml file


Git

Installation
Clone the repository:

git clone git@github.com:Akaqox/ICU-MML-XAI.git
cd ICU-MML-XAI


Install dependencies:
It is highly recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
pip install -r requirements.txt # (Assuming you will create this file with all necessary libraries)

Configuration:
The project uses a config.json file for hyperparameters and paths. Ensure you review and update config["paths"]["PATH"] and other relevant settings before running.

# Example config.json structure for first run
runningConfig:
  augmentation: 1
  segmentation: 1
  construct_dataset: 1
  train: 1
paths:
  PATH: "/path/to/your/data/root"
  # ... other paths
# ... other model/training parameters

Usage
The main pipeline is controlled by the runningConfig flags in config.yaml. Set True for stages you wish to execute.

To run the full pipeline (or selected stages):

python main.py

The main.py script orchestrates the following stages based on your config.yaml:

Processing and Augmentation Stage: Loads data, calculates mean resolution, and performs data augmentation.

if runningConfig["augmentation"] == True:
    # ... augmentation logic

Segmentation Stage: Performs lung region segmentation.

if runningConfig["segmentation"] == True:
    # ... segmentation logic

Dataset Construction Stage: Constructs the final dataset for training.

if runningConfig["construct_dataset"] == True:
    # ... dataset construction logic

Training Stage: Initiates the model training process. The provided code runs the training loop 50 times for experimental purposes.

if runningConfig["train"] == True:
    for i in range(50):
        # ... training logic

📊 Dataset
COVID

📝 Citation
This code is provided for research purposes only. If you use any part of this codebase or the methodology described, please cite the following paper:
(TO BE UPTADET
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


✉️ Contact
For any inquiries, please contact the authors:

S.Kizilisik: akaqox@gmail.com

S.Candemir: semacandemir@eskisehir.edu.tr

Ayşegül Terzi: (Please add Ayşegül Terzi's email if available)

⚖️ License
This project is licensed under the MIT License. See the LICENSE file for details.
