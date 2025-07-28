# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 21:41:16 2024

@author: Akaqox(Salih KIZILIŞIK), Ayşegül Terzi

This script is experimental codes of paper "Explainable Multimodal Machine Learning Model
for Predicting Intensive Care Unit Admission " by S.Kizilisik et al. You may use the codes only for research. 
Plese cite the paper if you use any part of the codes.

"""

import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from utils.utils import generate_unique_name, readConfig
from utils.metrics import f1_score
import numpy as np
from keras.models import load_model
import seaborn as sns
from tensorflow.keras.losses import BinaryFocalCrossentropy
from datetime import datetime
from utils.gradcam import XAI
from keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Concatenate
import shap


config = readConfig()
run = config["runningConfig"]
hypp = config["focal_loss"]
hypp = config["focal_loss"]
trainp =  config["parameters"]
lossp = config["focal_loss"]



class Evaluate():
    def __init__(self, history, save_dir, model, loss):
        self.num_epochs = len(history.history['loss'])
        self.num_train_images = len(glob.glob("dataset/ready_to_train/train" + "/*/*.npy")) 
        self.save_dir = save_dir
        self.model = model
        self.history = history
        self.loss = loss
        # Generate unique names
        self.model_name = generate_unique_name("weights", self.num_epochs, self.num_train_images)
        self.plot_name = generate_unique_name("plot", self.num_epochs, self.num_train_images)
        self.early_stopping_epoch = self.num_epochs - trainp["patience"]
        self.roc_plot_name = generate_unique_name("roc_prec_recall", self.num_epochs, self.num_train_images)
        self.cm_plot_name = generate_unique_name("cm_plot", self.num_epochs, self.num_train_images)
        self.prc_plot_name = generate_unique_name("prc", self.num_epochs, self.num_train_images)
        self.acc = 0
        self.img_pct = 0
        self.clinic_pct = 0
        self.clinical_dic = None
        
        
    def __call__(self, test_gen, pcts, pct_dict):
        self.img_pct = pcts[0]
        self.clinic_pct = pcts[1]
        self.clinical_dic = pct_dict
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        class_names = ['Not ICU','ICU']  # Swap classes
        #plot evaluation plots
        self.plotLC()

        y_true, y_pred, acc, loss = self.evaluate_model(test_gen,self.loss)
        
        AUC, th, tpr, fpr = self.ROCurve(y_true, y_pred)
        classification_report_str = self.CM(y_true, y_pred, th, class_names)
        PR_AUC, AP = self.plotPRC(y_true, y_pred)
    
        # Save Classification Report to a text file (append mode)
        report_filename = f'reports.txt'
        with open(os.path.join(self.save_dir[-1], report_filename), 'a') as report_file:
            report_file.write("\n\n--------------------- New Evaluation -------------------\n")
            report_file.write(f"{timestamp}\n")
            report_file.write(f"Model Path: models/{self.model_name}\n")
            report_file.write(f"Plot Path: plots/{self.cm_plot_name}\n")
            report_file.write(f"loss parameters: {hypp}\n")
            report_file.write(f"Train parameters: {trainp}\n")
            report_file.write(f"Test Accuracy :{self.acc}\n")
            report_file.write(f"Test Loss :{loss}\n")
            report_file.write(f"Optimal Threshold: {th}\n")
            report_file.write(f"AUC :{AUC}\n")
            report_file.write(f"PR_AUC :{PR_AUC}\n")
            report_file.write(f"AP_SCORE :{AP}\n")
            report_file.write(f"Plot Path: plots/{self.plot_name}\n")
            report_file.write(f"Confusion Matrix Path: ../out/plots/{self.cm_plot_name }\n")
            report_file.write(classification_report_str)
            
            report_file.write("\n-------------------SHAPLEY ANALYSIS-------------------\n")
            report_file.write(f"Average Contribution of Images: {self.img_pct:.2f}%\n")
            report_file.write(f"Average Contribution of Clinic Data: {self.clinic_pct:.2f}%\n")
            if trainp["model"]== "multi":
                report_file.write("\nTop Clinical SHAP Contributions (%):\n")
                for feature, pct in sorted(self.clinical_dic.items(), key=lambda x: x[1], reverse=True):
                    report_file.write(f"  {feature}: {pct:.2f}%\n")
        # Print Classification Report
        print(classification_report_str)

    def plotLC(self):
        
        # Extract the number of epochs
    
        # Save the model in keras format with a unique name in the "models" directory
        self.model.save(os.path.join(self.save_dir[0], f"{self.model_name}.keras"))
        # Plot the accuracy and loss
        plt.figure(figsize=(12, 6))
        sns.set(style="whitegrid")
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.num_epochs + 1), self.history.history["accuracy"], label="Training acc", color="blue")
        plt.plot(range(1, self.num_epochs + 1), self.history.history["val_accuracy"], "--", label="Validation acc", color="blue")
        plt.axvline(self.early_stopping_epoch, linestyle="--", color="black", label="Best model")  # Dashed line for early stopping
        plt.title("Accuracy Curve", fontsize=18)
        plt.xlabel("Epochs",fontsize=16)
        plt.ylabel("Accuracy", fontsize=16)
        plt.legend(loc="lower right", prop={'size': 14})
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.num_epochs + 1), self.history.history["loss"], label="Training loss",color="red")
        plt.plot(range(1, self.num_epochs + 1), self.history.history["val_loss"], "--", label="Validation loss",color="red")
        plt.axvline(self.early_stopping_epoch, linestyle="--", color="black", label="Best model")  # Dashed line for early stopping
        plt.title("Loss Curve", fontsize=18)
        plt.xlabel("Epochs",  fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.legend(loc="upper right", prop={'size': 14})
        
        # Save the plot with a unique name in the "plots" directory
        plt.savefig(os.path.join(self.save_dir[1], f"{self.plot_name}.png"), dpi=300)
        
        # Display the plot
        plt.show()
        
    def plotPRC(self, y_true, y_pred):
        """
        Plots the Precision-Recall curve and computes PR AUC.
    
        Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_pred (array-like): Predicted scores or probabilities.
    
        Returns:
        None
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
        ap_score = average_precision_score(y_true, y_pred, pos_label=1)
        # Option 1: Use sklearn's auc for PR AUC
        pr_auc = auc(recall, precision)
    
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, color="green", label=f'AUC-PR = {pr_auc:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
    
        # Save the plot
        plot_filename = self.prc_plot_name + ".png"
        plt.savefig(os.path.join(self.save_dir[1], plot_filename), dpi=300)
        plt.show()
        return pr_auc, ap_score
    

    def ROCurve(self, y_true, y_pred):
        """
        Takes the true binary labels and the predicted probabilities and generates the ROC curve.
        :param Y_test: true binary labels (0 or 1)
        :param y_pred: predicted probabilities (between 0 and 1)
        """
        print(y_true.shape)
        print(y_pred.shape)
        
        # Calculate False Positive Rate, True Positive Rate, and Thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
           
        # Find the optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
           
        print(f"Optimal Threshold: {optimal_threshold}")
        print(f"Optimal FPR: {fpr[optimal_idx]}, Optimal TPR: {tpr[optimal_idx]}")
           
        # Plot the ROC curve
        plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal Threshold = {optimal_threshold:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
         
        
        # Save the plot with a unique name in the "plots" directory
        plot_filename = self.roc_plot_name + ".png"
        plt.savefig(os.path.join(self.save_dir[1], plot_filename), dpi=300)
        plt.show()
        return roc_auc, optimal_threshold, tpr, fpr
    
    def CM(self, y_true, y_pred, th, class_names):
        
        y_pred = y_pred > th
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(5, 5))
        ax = plt.subplot()
        sns.set(font_scale=1.5) # Adjust to fit
        sns.heatmap(cm, annot=True, 
                    cmap="Blues", 
                    ax=ax, fmt="d", 
                    cbar=False,
                    linewidths=1,
                    linecolor="black", 
                    xticklabels=class_names, 
                    yticklabels=class_names);  
        
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn+fp)
        print("specificity value:   ",specificity)
    
        fp = sum(y_pred[y_true != 1])
        tn = sum(y_pred[y_true == 0] == False)
    
        sensitivity = tp / (tp + fn)
        print("sensitivity value:   ",sensitivity)
        
        ax.set_ylabel('True Label',fontsize=16);
        ax.set_xlabel('Predicted Label',fontsize=16);
        # Plot Confusion Matrix
        plt.title('Confusion Matrix',fontsize=18)
    
        # Save Confusion Matrix plot with a timestamp in the filename
    
        # Save the plot
        plot_filename = f'{self.cm_plot_name}img.png'
        plt.savefig(os.path.join(self.save_dir[1], plot_filename), dpi=300)
        plt.show()
        sns.reset_defaults()
        plt.clf()
    
        # Accuracy
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        self.acc = accuracy
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        return classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    def evaluate_model(self, test_gen):

        
        # Load the model with custom loss function
        model_path = self.save_dir[-1] + "/best_weights.keras"
        model = load_model(model_path, custom_objects={'focal_loss': self.loss,
                                                       'f1_score': f1_score})
    
        print(test_gen)
        # Evaluate the model on the test data
        test_loss, test_accuracy, _ = model.evaluate(test_gen, verbose=1)
        
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
        
        y_true_list = []
        y_pred_list = []
        expai =  XAI(model, self.save_dir[-1])
        for i, (Input, y_true) in enumerate(test_gen):
            
            y_pred = model.predict(Input, verbose=0)
            
            
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            
            expai(Input, i, y_true)
            
        y_true = np.array(y_true_list, dtype="int32")
        y_pred = np.array(y_pred_list, dtype="float32")
        
        expai.plot_overlayed_heatmap(self.save_dir[-1] + "/overlayed_heatmap_" + str(datetime.now().hour) + ":" +str(datetime.now().minute) + ".png")
        
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        return y_true, y_pred, test_accuracy, test_loss
    
    def shapley(self, model, X_sample, background, output_name, input_names=None, sample_size=50, use_gradient=True ):
        """
        Performs SHAP analysis for a specific output layer in a multi-input, multi-output Keras model.

        Parameters:
        - model: Trained Keras model.
        - test_gen: Data generator yielding ([input1, input2, ...], {output_name: label, ...}).
        - output_name: Name of the model output layer to analyze (e.g., 'death').
        - input_names: Optional list of input block names for display.
        - sample_size: Number of samples to use for SHAP (keep small to reduce compute).
        - use_gradient: If True, uses GradientExplainer instead of DeepExplainer (for better compatibility).

        Returns:
        - shap_values: SHAP values for each input.
        - sample_inputs: Input sample used for SHAP analysis.
        """
        # for i, layer in enumerate(self.model.layers):
        #     try:
        #         print(i, layer.name, layer.output.shape)
        #     except AttributeError:
        #         print(i, layer.name, "InputLayer (no output shape)")
        
        # try:
        # Step 1: Build fuser model (from original input to fusion layer output)
        
        # Get intermediate layer (Concat)
        
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        concat_layer = model.get_layer("concat")

        # --- Part 1: Encoder model ---
        fuser = Model(
            inputs=model.inputs,
            outputs=concat_layer.output,
            name="fuser_model"
        )
        concat_input = Input(shape=concat_layer.output.shape[1:], name="concat_input")

        # Start from the layer **after** "concat"
        x = concat_input
        for layer in model.layers:
            if layer.name == "normalization":
                start_adding = True
                continue
            if 'start_adding' in locals() and start_adding:
                x = layer(x)

        model = Model(inputs=concat_input, outputs=x, name="decoder_model")
        # Now you can use them
        
        background_fused = np.array(fuser(background))  # → shape: (batch, fusion_dim)
        X_fused = np.array(fuser(X_sample))  # → shape: (batch, fusion_dim)
        
        prediction = model(X_fused)
        
        
        #print(f"\n========== SHAP Analysis for '{output_name}' ==========")

        # Step 3: Generate fused inputs from your sample data
        
        explainer = shap.DeepExplainer(model, background_fused)
        shap_values = explainer.shap_values(X_fused, check_additivity=False)
        
        shap_block = np.mean([np.abs(class_shap) for class_shap in shap_values], axis=0)
        
        
        if input_names is None:
            input_names = ["img", "clinical"]
        img_dim = 576
        clinical_dim = 72
        
        img_shap = shap_block[:, :img_dim]
        clinical_shap = shap_block[:, img_dim:]

        clinical_dic = self.clinical_evaluator(clinical_shap)
        #print("IMG SHAP shape:", img_shap.shape)
        #print("Clinical SHAP shape:", clinical_shap.shape)
        #print(shap_values[0].shape)
        num_classes = shap_values[0].shape[0]
        
        #print("Average SHAP contribution percentages per modality for each output class:")
        
        # Sum SHAP per modality, summed over all samples
        img_sum = np.sum(img_shap)
        clinical_sum = np.sum(clinical_shap)
       
        total_sum = img_sum + clinical_sum
        
        if total_sum == 0:
            img_pct = clinical_pct = 0.0
        else:
            img_pct = 100 * img_sum / total_sum
            clinical_pct = 100 * clinical_sum / total_sum
        

        # Optional: SHAP summary plot (use smaller size if needed)
        #shap.summary_plot(shap_values, X_fused, feature_names=None)
        
        # except Exception as e:
        #     print(f"SHAP analysis failed for output '{output_name}': {e}")
        #     return None, None
        return shap_values, [img_pct, clinical_pct], clinical_dic


    def clinical_evaluator(self, shap_values):
        column_names = [
            "age.splits", "gender_concept_name", "Acute.Hepatic.Injury..during.hospitalization.",
            "Acute.Kidney.Injury..during.hospitalization.", "htn_v", "dm_v", "cad_v", "hf_ef_v", "ckd_v",
            "malignancies_v", "copd_v", "other_lung_disease_v", "acei_v", "arb_v", "antibiotics_use_v",
            "nsaid_use_v", "days_prior_sx", "smoking_status_v", "cough_v", "dyspnea_admission_v", "nausea_v",
            "vomiting_v", "diarrhea_v", "abdominal_pain_v", "fever_v", "BMI.over30", "BMI.over35",
            "temperature.over38", "pulseOx.under90", "Respiration.over24", "HeartRate.over100",
            "Lymphocytes.under1k", "Aspartate.over40", "Alanine.over60", "Troponin.above0.01",
            "8331-1_Oral temperature", "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry",
            "9279-1_Respiratory rate", "76282-3_Heart rate.beat-to-beat by EKG", "8480-6_Systolic blood pressure",
            "76536-2_Mean blood pressure by Noninvasive",
            "33256-9_Leukocytes [#/volume] corrected for nucleated erythrocytes in Blood by Automated count",
            "751-8_Neutrophils [#/volume] in Blood by Automated count",
            "731-0_Lymphocytes [#/volume] in Blood by Automated count",
            "2951-2_Sodium [Moles/volume] in Serum or Plasma",
            "1920-8_Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma",
            "1744-2_Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma by No addition of P-5'-P",
            "2524-7_Lactate [Moles/volume] in Serum or Plasma",
            "6598-7_Troponin T.cardiac [Mass/volume] in Serum or Plasma",
            "75241-0_Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay",
            "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay",
            "1988-5_C reactive protein [Mass/volume] in Serum or Plasma",
            "39156-5_Body mass index (BMI) [Ratio]",
            "2951-2_Sodium [Moles/volume] in Serum or Plasma.1",
            "2823-3_Potassium [Moles/volume] in Serum or Plasma",
            "2075-0_Chloride [Moles/volume] in Serum or Plasma",
            "1963-8_Bicarbonate [Moles/volume] in Serum or Plasma",
            "3094-0_Urea nitrogen [Mass/volume] in Serum or Plasma",
            "2160-0_Creatinine [Mass/volume] in Serum or Plasma",
            "62238-1_Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)",
            "2345-7_Glucose [Mass/volume] in Serum or Plasma",
            "Sodium_135-145", "Potassium_3.5-5.2", "Chloride_96-107", "Bicarbonate_21-31",
            "Blood_Urea_Nitrogen_5-20", "Creatinine_0.5-1.2", "eGFR_30-60", "D_dimer_500-3000",
            "SBP_120-139", "MAP_65-90", "procalcitonin_0.25-0.5"
        ]
        # 3. Remove is_icu and to_patient_id if present
        columns_filtered = []
        values_filtered = []
        
                
        first_row = shap_values[0, :]
        
        # 2. Compute sum of absolute SHAP values (exclude filtered columns)
        abs_values = [
            abs(val) for col, val in zip(column_names, first_row) if col not in ("is_icu", "to_patient_id")
        ]
        total_abs = sum(abs_values)
        
        # 3. Append columns and percent values
        for col, val in zip(column_names, first_row):
            if col not in ("is_icu", "to_patient_id"):
                percent = abs(val) / total_abs * 100
                columns_filtered.append(col)
                values_filtered.append(percent)
        
        # Now columns_filtered and values_filtered hold features and their % SHAP contributions
        
        # Example print top 5
        # for col, val in zip(columns_filtered[:5], values_filtered[:5]):
        #     print(f"{col}: {val:.2f}%")
            
        shap_dict = dict(zip(columns_filtered, values_filtered))
        return shap_dict
    
    def evaluate_model(self, test_gen, loss):
        
        # Load the model with custom loss function
        model_path = self.save_dir[-1] + "/best_weights.keras"
        model = load_model(model_path, custom_objects={'focal_loss': self.loss,
                                                       'f1_score': f1_score})
    
        print(test_gen)
        # Evaluate the model on the test data
        test_loss, test_accuracy, _ = model.evaluate(test_gen, verbose=1)
        
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
        
        y_true_list = []
        y_pred_list = []
        if run["gradcam"]:
            expai =  XAI(model, self.save_dir[-1], test_accuracy)
        
        for i, (Input, y_true) in enumerate(test_gen):
            
            y_pred = model.predict(Input, verbose=0)
            
            
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            if run["gradcam"]:
                expai(Input, i, y_true)
            
        y_true = np.array(y_true_list, dtype="int32")
        y_pred = np.array(y_pred_list, dtype="float32")
        
        if run["gradcam"]:
            expai.plot_overlayed_heatmap(self.save_dir[-1] + "/overlayed_heatmap_" + str(datetime.now().hour) + ":" +str(datetime.now().minute) + ".png")
        
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        return y_true, y_pred, test_accuracy, test_loss