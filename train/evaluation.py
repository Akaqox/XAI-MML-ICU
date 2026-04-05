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
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,brier_score_loss, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.utils import resample
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                                accuracy_score, precision_score, recall_score, confusion_matrix)
from utils.utils import generate_unique_name, readConfig
from sklearn.metrics import f1_score as sk_f1_score
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
from datetime import date
from utils.utils import readConfig, create_out_folder, create_destroy_dic

config = readConfig()
run = config["runningConfig"]
hypp = config["focal_loss"]
hypp = config["focal_loss"]
trainp =  config["parameters"]
lossp = config["focal_loss"]



class Evaluate():
    def __init__(self, model, save_dir = None, history = None, loss = None):
        if history is None:
            self.num_epochs = 1
            self.history = None
        else:
            self.num_epochs = len(history.history['loss'])
            self.history = history
        if loss is None:
            self.loss = 'binary_crossentropy'
        else:
            self.loss = loss
            
        if save_dir is None:     
            today = date.today().strftime("%Y-%m-%d")
            save_dir = create_out_folder(today, "unspecified") 
        self.save_dir = save_dir     
        
        self.num_train_images = len(glob.glob("dataset/ready_to_train/train" + "/*/*.npy")) 
        self.model = model
        
        
        self.test_limiter = 0
        self.limit_accuracy = 0.76
        # Generate unique names
        self.model_name = generate_unique_name("weights", self.num_epochs, self.num_train_images)
        self.plot_name = generate_unique_name("plot", self.num_epochs, self.num_train_images)
        self.early_stopping_epoch = self.num_epochs - trainp["patience"]
        self.roc_plot_name = generate_unique_name("roc_prec_recall", self.num_epochs, self.num_train_images)
        self.cm_plot_name = generate_unique_name("cm_plot", self.num_epochs, self.num_train_images)
        self.prc_plot_name = generate_unique_name("prc", self.num_epochs, self.num_train_images)
        self.calibration_name = generate_unique_name("calibration", self.num_epochs, self.num_train_images)
        self.decision_name = generate_unique_name("decision", self.num_epochs, self.num_train_images)

        self.acc = 0
        self.img_pct = 0
        self.clinic_pct = 0
        self.clinical_dic = None
        
        
    def __call__(self, test_gen, pcts, pct_dict):
        self.img_pct = pcts[0]
        self.clinic_pct = pcts[1]
        self.clinical_dic = pct_dict
        mode = None
        if self.clinic_pct > 0:
            mode = 'multi'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        class_names = ['Not ICU','ICU']  # Swap classes
        #plot evaluation plots
        if not (self.history is None):
            self.plotLC()

        y_true, y_pred, X_input, acc, loss = self.evaluate_model(test_gen)
                
        AUC, th, tpr, fpr, thresholds = self.save_roc(y_true, y_pred)
                
        # Ensure we are using the arrays from the sigmoid-activated model
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        # Find indices where sensitivity is between 0.95 and 0.96 (to avoid the 1.0 trap)
        safe_indices = np.where((tpr >= 0.95) & (tpr < 1.0))[0]

        if len(safe_indices) > 0:
            # Pick the one with the best specificity
            best_idx = safe_indices[np.argmin(fpr[safe_indices])]
            th = thresholds[best_idx]
        else:
            # If no points are in that range, take the very first point that hits 0.95
            th = thresholds[np.where(tpr >= 0.95)[0][0]]

        ci_results = self.bootstrap_evaluation(y_true, y_pred, class_names, threshold=th)
        if (ci_results['Accuracy'] < self.limit_accuracy) and self.test_limiter:
            return
        self.calibration_curve(y_true, y_pred)
        self.save_decision(y_true, y_pred)
    
        classification_report_str = self.CM(y_true, y_pred, th, class_names)
        PR_AUC, AP = self.plotPRC(y_true, y_pred)
        
        # Calculate Confidence Intervals using the dynamic threshold 'th'
        
    
        # Save Classification Report to a text file (append mode)
        report_filename = f'reports.txt'
        with open(os.path.join(self.save_dir[-1], report_filename), 'a') as report_file:
            report_file.write("\n\n--------------------- New Evaluation -------------------\n")
            report_file.write(f"{timestamp}\n")
            report_file.write(f"Model Path: models/{self.model_name}\n")
            report_file.write(f"Plot Path: plots/{self.cm_plot_name}\n")
            report_file.write(f"loss parameters: {hypp}\n")
            report_file.write(f"Train parameters: {trainp}\n")
            report_file.write(f"Test Loss :{loss}\n")
            report_file.write(f"Optimal Threshold: {th}\n")
            
            report_file.write("\n--- Point Estimates & Confidence Intervals ---\n")
            # Point estimates without CI
            report_file.write(f"Accuracy (at threshold) : {ci_results['Accuracy']:.4f}\n")
            report_file.write(f"Precision : {ci_results['Precision']:.4f}\n")
            report_file.write(f"Recall : {ci_results['Recall']:.4f}\n\n")
            
            # Point estimates with CI
            report_file.write(f"Sensitivity : {ci_results['Sensitivity'][0]:.4f}\n")
            report_file.write(f"Sensitivity 95% CI : [{ci_results['Sensitivity'][1][0]:.3f}, {ci_results['Sensitivity'][1][1]:.3f}]\n")
            
            report_file.write(f"Specificity : {ci_results['Specificity'][0]:.4f}\n")
            report_file.write(f"Specificity 95% CI : [{ci_results['Specificity'][1][0]:.3f}, {ci_results['Specificity'][1][1]:.3f}]\n")
            
            report_file.write(f"AUC : {ci_results['AUC'][0]:.4f}\n")
            report_file.write(f"AUC 95% CI : [{ci_results['AUC'][1][0]:.3f}, {ci_results['AUC'][1][1]:.3f}]\n")
            
            report_file.write(f"PR_AUC : {ci_results['PR_AUC'][0]:.4f}\n")
            report_file.write(f"PR_AUC 95% CI : [{ci_results['PR_AUC'][1][0]:.3f}, {ci_results['PR_AUC'][1][1]:.3f}]\n")
            
            report_file.write(f"F1-Score: {ci_results['F1'][0]:.4f}\n")
            report_file.write(f"F1-Score 95% CI : [{ci_results['F1'][1][0]:.3f}, {ci_results['F1'][1][1]:.3f}]\n")

            report_file.write(f"Brier-Score: {ci_results['Brier'][0]:.4f}\n")
            report_file.write(f"Brier-Score 95% CI : [{ci_results['Brier'][1][0]:.3f}, {ci_results['Brier'][1][1]:.3f}]\n")

            report_file.write(f"\nAP_SCORE :{AP}\n")
            report_file.write(f"Plot Path: plots/{self.plot_name}\n")
            report_file.write(f"Confusion Matrix Path: ../out/plots/{self.cm_plot_name }\n")
            if mode == 'multi':
                if trainp["model"] == "multi":
                    X_tabular_test = X_input
                    print(len(X_tabular_test))
                    # Assuming you extracted the clinical test array into a variable named X_tabular_test
                    subgroup_text = self.generate_subgroup_report(y_true, y_pred, X_tabular_test, th)
                    
                    report_file.write("\n\n------------------- SUBGROUP ANALYSIS -------------------\n")
                    report_file.write(subgroup_text)
                    report_file.write("\n* Comorbidity defined as presence of HTN, DM, CAD, HF, CKD, or Lung Disease.\n")
                
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
        #plt.show()
        plt.clf()
    
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
        #plt.show()
        plt.clf()
        return pr_auc, ap_score
    
    
    def save_roc(self, y_true, y_pred):
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Calculate optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        
        # Create the combined DataFrame
        df = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            
            'roc_auc': np.nan,
            'opt_fpr': np.nan,
            'opt_tpr': np.nan,
            'opt_threshold': np.nan
        })
        
        # Assign metadata to the first row
        df.loc[0, 'roc_auc'] = roc_auc
        df.loc[0, 'opt_fpr'] = fpr[optimal_idx]
        df.loc[0, 'opt_tpr'] = tpr[optimal_idx]
        df.loc[0, 'opt_threshold'] = thresholds[optimal_idx]
        
        csv_filename = f"{self.roc_plot_name}.csv"
        df.to_csv(os.path.join(self.save_dir[1], csv_filename), index=False)
        
        return roc_auc, thresholds[optimal_idx], fpr, tpr, thresholds
    
    def calibration_curve(self, y_true, y_pred):
        y_prob = y_pred

        # 1. Calculate Calibration Curve & Brier Score
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform")
        brier = brier_score_loss(y_true, y_prob)

        # 2. Plotting
        fig, ax1 = plt.subplots(figsize=(7, 7))

        # The Calibration Curve
        ax1.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'Model (Brier: {brier:.4f})', color='blue')
        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

        # 3. Add a Histogram (Standard for Medical Papers)
        # This shows the density of your predictions along the x-axis
        ax2 = ax1.twinx()  # Create a twin y-axis for the histogram
        ax2.hist(y_prob, bins=10, range=(0, 1), alpha=0.1, color='blue', density=True)
        ax2.set_ylabel("Density of Predictions", color='gray', fontsize=16)
        ax2.tick_params(axis='y', labelcolor='gray')

        # Formatting
        ax1.set_xlabel("Predicted Probability (Risk)", fontsize=16)
        ax1.set_ylabel("Observed Frequency (Actual)", fontsize=16)
        ax1.set_title("Model Calibration Curve Analysis", fontsize=18)
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)

        plt.tight_layout()
        plot_filename = f'{self.calibration_name}img.png'
        plt.savefig(os.path.join(self.save_dir[1], plot_filename), dpi=300)
        plt.clf()

    def save_decision(self, y_true, y_pred):
        thresholds = np.linspace(0.01, 0.99, 100)
        net_benefit_model = []
        net_benefit_all = []
        
        N = len(y_true)
        prevalence = np.mean(y_true)

        for t in thresholds:
            # 1. Model Net Benefit
            preds = (y_pred >= t)
            TP = ((preds == 1) & (y_true == 1)).sum()
            FP = ((preds == 1) & (y_true == 0)).sum()
            nb_model = (TP / N) - (FP / N) * (t / (1 - t))
            net_benefit_model.append(nb_model)
            
            # 2. Treat All Net Benefit
            nb_all = prevalence - (1 - prevalence) * (t / (1 - t))
            net_benefit_all.append(nb_all)

        # Save to CSV
        df = pd.DataFrame({
            'threshold': thresholds,
            'nb_model': net_benefit_model,
            'nb_all': net_benefit_all
        })
        
        csv_filename = f'{self.decision_name}.csv'
        df.to_csv(os.path.join(self.save_dir[1], csv_filename), index=False)
        
        
    def CM(self, y_true, y_pred, th, class_names):
        
        y_pred = (y_pred >= th).astype(int)
        y_true = np.asarray(y_true).astype(int)
        
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
        #plt.show()
        sns.reset_defaults()
        plt.clf()
    
        # Accuracy
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        self.acc = accuracy
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        return classification_report(y_true, y_pred, target_names=class_names, digits=4)
    def generate_subgroup_report(self, y_true, y_pred, X_tabular_test, threshold):
        """
        Filters patients into subgroups and calculates metrics for each.
        X_tabular_test MUST be the raw 69-feature clinical array corresponding to the test labels.
        """
        
        print("\nRunning Subgroup Analysis...")
        
        if isinstance(X_tabular_test, list) and len(X_tabular_test) == 2:
            clinical_data = np.asarray(X_tabular_test[1]) 
        else:
            clinical_data = np.asarray(X_tabular_test)
            
        # Ensure it is a 2D array (N_patients, 69_features)
        if len(clinical_data.shape) == 1:
            print("ERROR: Clinical data is 1D. Make sure you are passing the full test set array.")
            return "Subgroup error: Invalid data shape."
            
        clinical_data = np.squeeze(clinical_data)
        
        # 2. Define masks using the full 2D array ( [:, ...] keeps all patients )
        masks = {
            "Male": (clinical_data[:, 1] < 0.5),   
            "Female": (clinical_data[:, 1] >= 0.5), 
            "Age < 60": (clinical_data[:, 0] < 0.5), 
            "Age >= 60": (clinical_data[:, 0] >= 0.5),
            
            # Summing horizontally across axis=1 requires a 2D slice
            "Comorbidity (Yes)": (np.sum(clinical_data[:, [2,3,4,5,6,8,9]], axis=1) > 0.5),
            "Comorbidity (No)": (np.sum(clinical_data[:, [2,3,4,5,6,8,9]], axis=1) <= 0.5)
        }

        report_lines = []
        
        # ADDED: N, Acc, Prec, Rec to the header
        header = f"{'Subgroup':<18} | {'N':<5} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'AUC':<18} | {'PR-AUC':<18} | {'F1-Score':<18} | {'Sens':<18} | {'Spec':<18} | {'Brier':<18}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        print(header)

        for group_name, mask in masks.items():
            # Filter the true labels and predictions for this specific subgroup
            y_t_sub = np.asarray(y_true)[mask]
            y_p_sub = np.asarray(y_pred)[mask]
            
            # Calculate patient count for the subgroup
            n_count = len(y_t_sub)
            
            if n_count < 5:
                # Skip if subgroup is suspiciously small/empty
                row = f"{group_name:<18} | {n_count:<5} | N/A (Sample size too small)"
            else:
                metrics = self.get_bootstrapped_metrics(y_t_sub, y_p_sub, threshold)
                # ADDED: n_count and Point Estimates to the formatted row
                row = f"{group_name:<18} | {n_count:<5} | {metrics['Accuracy']:<6} | {metrics['Precision']:<6} | {metrics['Recall']:<6} | {metrics['AUC']:<18} | {metrics['PR-AUC']:<18} | {metrics['F1']:<18} | {metrics['Sens']:<18} | {metrics['Spec']:<18} | {metrics['Brier']:<18}"
            
            print(row)
            report_lines.append(row)
            
        return "\n".join(report_lines)
    
    def get_bootstrapped_metrics(self, y_true, y_pred, threshold=0.5, n_iterations=1000):
       
        y_true = np.asarray(y_true).astype(np.int32)
        y_pred = np.asarray(y_pred)
        y_pred_binary = (y_pred >= threshold).astype(np.int32)
        
        # 1. Point Estimates
        try:
            pt_auc = roc_auc_score(y_true, y_pred)
            pt_pr = average_precision_score(y_true, y_pred)
        except ValueError:
            pt_auc, pt_pr = np.nan, np.nan
            
        pt_f1 = sk_f1_score(y_true, y_pred_binary, zero_division=0)
        pt_brier = brier_score_loss(y_true, y_pred)
        
        # New Point Estimates (No intervals needed)
        pt_acc = accuracy_score(y_true, y_pred_binary)
        pt_prec = precision_score(y_true, y_pred_binary, zero_division=0)
        pt_recall = recall_score(y_true, y_pred_binary, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
        pt_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        pt_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 2. Bootstrap Loop
        bootstrapped_auc, bootstrapped_pr_auc, bootstrapped_f1 = [], [], []
        bootstrapped_sens, bootstrapped_spec, bootstrapped_brier = [], [], []
        
        for _ in range(n_iterations):
            if len(np.unique(y_true)) > 1:
                indices = resample(np.arange(len(y_true)), replace=True, stratify=y_true)
            else:
                indices = resample(np.arange(len(y_true)), replace=True)
                
            y_t_b, y_p_b, y_pb_b = y_true[indices], y_pred[indices], y_pred_binary[indices]
            
            try:
                bootstrapped_auc.append(roc_auc_score(y_t_b, y_p_b))
                bootstrapped_pr_auc.append(average_precision_score(y_t_b, y_p_b))
                bootstrapped_f1.append(sk_f1_score(y_t_b, y_pb_b, zero_division=0))
                bootstrapped_brier.append(brier_score_loss(y_t_b, y_p_b))
                
                tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_t_b, y_pb_b, labels=[0, 1]).ravel()
                if (tp_b + fn_b) > 0: bootstrapped_sens.append(tp_b / (tp_b + fn_b))
                if (tn_b + fp_b) > 0: bootstrapped_spec.append(tn_b / (tn_b + fp_b))
            except ValueError:
                continue
                
        # 3. Formatter Helper
        def fmt(pt, arr):
            if np.isnan(pt) or len(arr) == 0: return "N/A"
            lower, upper = np.percentile(arr, 2.5), np.percentile(arr, 97.5)
            return f"{pt:.3f} ({lower:.3f}-{upper:.3f})"
            
        return {
            "Accuracy": f"{pt_acc:.3f}",
            "Precision": f"{pt_prec:.3f}",
            "Recall": f"{pt_recall:.3f}",
            "AUC": fmt(pt_auc, bootstrapped_auc),
            "PR-AUC": fmt(pt_pr, bootstrapped_pr_auc),
            "F1": fmt(pt_f1, bootstrapped_f1),
            "Sens": fmt(pt_sens, bootstrapped_sens),
            "Spec": fmt(pt_spec, bootstrapped_spec),
            "Brier": fmt(pt_brier, bootstrapped_brier)
        }
    def bootstrap_evaluation(self, y_true, y_pred, class_names, n_iterations=1000, threshold=0.5):
        """
        Calculates exact metrics and 95% Confidence Intervals using Stratified Bootstrapping.
        """
        
        print(f"Running {n_iterations} bootstrap iterations with threshold {threshold:.4f}...")
        
        # Pre-calculate binary predictions based on the dynamic threshold
        y_pred_binary = (y_pred >= threshold).astype(int)
        point_accuracy = accuracy_score(y_true, y_pred_binary)
        
        
        if (point_accuracy < self.limit_accuracy) and self.test_limiter:
            print(point_accuracy)
            results = {
            "Accuracy": point_accuracy,}
            return results
        
        
        
        # 1. CALCULATE TRUE POINT ESTIMATES ON FULL TEST SET
        point_auc = roc_auc_score(y_true, y_pred)
        point_pr_auc = average_precision_score(y_true, y_pred)
        point_f1 = sk_f1_score(y_true, y_pred_binary)
        point_brier = brier_score_loss(y_true, y_pred)
        point_precision = precision_score(y_true, y_pred_binary)
        point_recall = recall_score(y_true, y_pred_binary) 
        
        # Calculate true Sensitivity and Specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        point_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        point_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # 2. INITIALIZE BOOTSTRAP ARRAYS
        bootstrapped_auc = []
        bootstrapped_pr_auc = []
        bootstrapped_f1 = []
        bootstrapped_sens = []
        bootstrapped_spec = []
        bootstrapped_brier = []
        for i in range(n_iterations):
            # Stratified resample
            indices = resample(np.arange(len(y_true)), replace=True, stratify=y_true)
            
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_pred_bin_boot = y_pred_binary[indices]
            
            try:
                # Recalculate metrics on the RESAMPLED data
                bootstrapped_auc.append(roc_auc_score(y_true_boot, y_pred_boot))
                bootstrapped_pr_auc.append(average_precision_score(y_true_boot, y_pred_boot))
                bootstrapped_f1.append(sk_f1_score(y_true_boot, y_pred_bin_boot))
                bootstrapped_brier.append(brier_score_loss(y_true_boot, y_pred_boot))
                
                # Sensitivity and Specificity for this resample
                tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_boot, y_pred_bin_boot).ravel()
                
                if (tp_b + fn_b) > 0:
                    bootstrapped_sens.append(tp_b / (tp_b + fn_b))
                if (tn_b + fp_b) > 0:
                    bootstrapped_spec.append(tn_b / (tn_b + fp_b))
                
            except ValueError:
                # Handles rare cases where a resample might only contain one class
                continue
                
        # 3. COMPILE RESULTS
        results = {
            "Accuracy": point_accuracy,
            "Precision": point_precision,
            "Recall": point_recall,
            "AUC": [point_auc, (np.percentile(bootstrapped_auc, 2.5), np.percentile(bootstrapped_auc, 97.5))],
            "PR_AUC": [point_pr_auc, (np.percentile(bootstrapped_pr_auc, 2.5), np.percentile(bootstrapped_pr_auc, 97.5))],
            "F1": [point_f1, (np.percentile(bootstrapped_f1, 2.5), np.percentile(bootstrapped_f1, 97.5))],
            "Sensitivity": [point_sensitivity, (np.percentile(bootstrapped_sens, 2.5), np.percentile(bootstrapped_sens, 97.5))],
            "Specificity": [point_specificity, (np.percentile(bootstrapped_spec, 2.5), np.percentile(bootstrapped_spec, 97.5))],
            "Brier": [point_brier, (np.percentile(bootstrapped_brier, 2.5), np.percentile(bootstrapped_brier, 97.5))]

        }
        
        return results
    
    def shapley(self, original_model, X_sample, background, output_name, input_names=None, sample_size=50, use_gradient=True):
        """
        Performs SHAP analysis on the fused representations.
        """
        tf.get_logger().setLevel('ERROR')

        # 1. Prevent gradient saturation
        original_model.layers[-1].activation = tf.keras.activations.linear

        # 2. Extract inputs cleanly
        if isinstance(background, dict):
            bg_list = [background["feature_input"], background["tabular_input"]]
            x_list = [X_sample["feature_input"], X_sample["tabular_input"]]
        elif isinstance(background, tuple) or isinstance(background, list):
            bg_list = list(background)
            x_list = list(X_sample)
        else:
            print("ERROR: Unrecognized data format from generator.")
            return None, [0.0, 0.0], {}

        # 3. Explain the multi-input model
        explainer = shap.DeepExplainer(original_model, bg_list)
        shap_values = explainer.shap_values(x_list, check_additivity=False)

        # 4. Average across batch (and classes if multi-class)
        if isinstance(shap_values, list) and isinstance(shap_values[0], list):
             # Multi-class output
             averaged_arrays = [np.mean([np.abs(c[i]) for c in shap_values], axis=0) for i in range(len(shap_values[0]))]
        else:
             # Single-class output
             averaged_arrays = [np.mean(np.abs(arr), axis=0) for arr in shap_values]

        # 5. DYNAMIC SHAPE MATCHING (The Fix)
        # Instead of guessing order, we explicitly identify the image and clinical arrays by their length.
        img_shap_vals = None
        clin_shap_vals = None

        for arr in averaged_arrays:
            arr_flat = arr.flatten()
            if len(arr_flat) == 576:
                img_shap_vals = arr_flat
            elif len(arr_flat) == 69:
                clin_shap_vals = arr_flat

        if img_shap_vals is None or clin_shap_vals is None:
            print("CRITICAL ERROR: Could not find arrays matching shapes 576 and 69. Check model inputs.")
            return None, [0.0, 0.0], {}

        # 6. CALCULATE PERCENTAGES
        img_sum = np.sum(img_shap_vals)
        clinical_sum = np.sum(clin_shap_vals)
        total_sum = img_sum + clinical_sum

        if total_sum == 0:
            img_pct = clinical_pct = 0.0
        else:
            img_pct = 100 * img_sum / total_sum
            clinical_pct = 100 * clinical_sum / total_sum

        # Now we are 100% guaranteed that the 69-length array is sent to the evaluator
        clinical_dic = self.clinical_evaluator(clin_shap_vals)

        return shap_values, [img_pct, clinical_pct], clinical_dic

    def clinical_evaluator(self, shap_values):
        column_names = [
            "age.splits", "gender_concept_name", "htn_v", "dm_v", "cad_v", "hf_ef_v", "ckd_v",
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
       # 1. Force into a numpy array and ensure it is flat (1D)
        shap_array = np.array(shap_values)
        
        # If the array is still 2D for some reason (e.g., [batch_size, 69]), average it.
        if len(shap_array.shape) > 1:
            shap_array = np.mean(np.abs(shap_array), axis=0)
            
        shap_array = shap_array.flatten()
        
        # Safety check: ensure lengths match before zipping
        if len(shap_array) != len(column_names):
            print(f"WARNING: Feature count mismatch. Expected {len(column_names)}, got {len(shap_array)}")
            
        # 2. Compute total sum of absolute values
        total_abs = np.sum(np.abs(shap_array))
        
        shap_dict = {}
        
        # 3. Map values to column names and calculate percentages
        for col, val in zip(column_names, shap_array):
            if col not in ("is_icu", "to_patient_id"):
                if total_abs > 0:
                    percent = (abs(val) / total_abs) * 100
                else:
                    percent = 0.0
                shap_dict[col] = percent
                
        return shap_dict
    
    def evaluate_model(self, test_gen):
        if self.model is None:
            # Load the model with custom loss function
            model_path = self.save_dir[-1] + "/best_weights.keras"
            model = load_model(model_path, custom_objects={'focal_loss': self.loss,
                                                        'f1_score': f1_score})
        else:
            model = self.model
            
        model.compile(
            optimizer='Adam', loss=self.loss, metrics='acc'
            )
        print(test_gen)
        # Evaluate the model on the test data
        test_results = model.evaluate(test_gen, verbose=1)
        
        test_loss = test_results[0]
        test_accuracy = test_results[1]
        
        y_true_list = []
        y_pred_list = []
        input_list = []
        
        if run["gradcam"]:
            expai =  XAI(model, self.save_dir[-1], test_accuracy)
        
        for i, (Input, y_true) in enumerate(test_gen):
            
            y_pred = model.predict(Input, verbose=0)
 
            if isinstance(Input, (list, tuple)):
                tabular_batch = Input[1]  # Multi-input model: grab the clinical branch
            else:
                tabular_batch = Input     # Single-input model: it is already an array
                
            # 2. Now you can safely check the shape and append
            if len(tabular_batch.shape) >= 1: # (Or whatever condition you need)
                input_list.append(np.squeeze(tabular_batch))
                
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            if run["gradcam"]:
                expai(Input, i, y_true)
                
        
        y_true = np.array(y_true_list, dtype="int32")
        y_pred = np.array(y_pred_list, dtype="float32")
        X_input = np.array(input_list, dtype="float32")
        
        if run["gradcam"]:
            expai.plot_overlayed_heatmap(self.save_dir[-1] + "/overlayed_heatmap_" + str(datetime.now().hour) + ":" +str(datetime.now().minute) + ".png")
        
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        return y_true, y_pred, X_input, test_accuracy, test_loss

def plot_combined_decision_curves(csv_list, model_labels):
    """
    csv_list: list of paths to the saved CSV files
    model_labels: names for the legend (e.g., ['CNN', 'ResNet'])
    """
    plt.figure(figsize=(7, 7))
    
    # Load first CSV to get baselines (Treat All/None)
    first_df = pd.read_csv(csv_list[0])
    thresholds = first_df['threshold']
    nb_all = first_df['nb_all']
    prevalence = first_df['nb_all'].iloc[0] # Approx prevalence from t=0.01

    # 1. Plot Baselines (Same style as your code)
    plt.plot(thresholds, nb_all, lw=2, label="Treat All", color='gray', linestyle='--')
    plt.axhline(y=0, color='black', lw=1, label="Treat None")

    # 2. Plot Each Model
    # You can define a list of colors if you have many models
    colors = ['blue', 'red', 'green']
    
    for i, csv_path in enumerate(csv_list):
        df = pd.read_csv(csv_path)
        plt.plot(df['threshold'], df['nb_model'], lw=2, alpha = 0.7,
                 label=model_labels[i], color=colors[i % len(colors)])

    # 3. Professional Adjustments (Exact copy of your style)
    plt.ylim(-0.05, prevalence + 0.1)
    plt.xlabel("Threshold Probability", fontsize=16)
    plt.ylabel("Net Benefit", fontsize=16)
    plt.title("Decision Curve Analysis", fontsize=18)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    plt.savefig("results/Combined_DCA.png", dpi=800)
    plt.close()
    
def plot_combined_roc_curves( csv_list, model_labels):
    """
    csv_list: List of file paths (strings) OR pandas DataFrames
    model_labels: List of names for the legend
    """
    plt.figure(figsize=(7, 7))
    colors = ['blue', 'red', 'green']
    
    for i, item in enumerate(csv_list):
        # Handle both file paths and pre-loaded DataFrames
        if isinstance(item, str):
            df = pd.read_csv(item)
            
        # Extract metadata from the first row
        roc_auc = df['roc_auc'].iloc[0]
        opt_fpr = df['opt_fpr'].iloc[0]
        opt_tpr = df['opt_tpr'].iloc[0]
        opt_threshold = df['opt_threshold'].iloc[0]
        
        name = model_labels[i]
        color = colors[i % len(colors)]

        # Plot the main ROC line
        plt.plot(df['fpr'], df['tpr'], color=color, alpha=0.7, lw=2, 
                 label=f'{name} (AUC = {roc_auc:.2f})')
        
        # Plot the optimal point marker
        plt.scatter(opt_fpr, opt_tpr, color=color, edgecolor='black', s=60, zorder=5,
                    label=f'{name} Opt Threshold = {opt_threshold:.2f}')

    # Final visual styling
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic Comparison', fontsize=18)
    
    # Position legend slightly better if it gets too crowded
    plt.legend(loc="lower right", fontsize=10, frameon=True) 
    plt.grid(alpha=0.3)
    
    plt.savefig("results/Combined_ROC.png", dpi=500)
    plt.close()
    
