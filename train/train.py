# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 2024

@authors: S.Kizilisik (kzlsksalih@gmail.com)


This script is experimental codes of paper " Explainable Multimodal Machine Learning Model
for Predicting Intensive Care Unit Admission" by S.Candemir et al. You may use the codes only for research. 
Plese cite the paper if you use any part of the codes.

"""

import os
import gc
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryFocalCrossentropy
# from utils.losses import CategoricalFocalCrossentropy
from datetime import date
import tensorflow as tf
import keras

from utils.metrics import f1_score
from utils.utils import readConfig, create_out_folder, create_destroy_dic
from train.evaluation import Evaluate
from model.models import VGG16_Multi, clinic_model, feature_extraction, VGG16_fine_tune, mobilenetv3_transfer_learning, mnet_Multi
from dataset.dataloader import DataGenerator
import keras_tuner as kt
from sklearn.model_selection import RandomizedSearchCV

config = readConfig()
run = config["runningConfig"]
paths = config["paths"]
ready = os.path.join(paths["PATH"], paths["ready"])

def generator(file_list):
    for f in file_list:
        x = np.load(f)
        x = np.repeat(x[..., np.newaxis], 3, axis=-1)
        yield [f, x.astype(np.float32)]
        
class Train():
    def __init__(self):
        
        self.hypp = config["parameters"]
        lossp = config["focal_loss"]
        model_arch = self.hypp["model"]
        hp = self.hypp
        
        # Choose model architecture
        if model_arch == "vgg":
            model = VGG16_Multi(weights='imagenet')
            
        elif model_arch == "multi":
            model = mnet_Multi(weights='imagenet', hp=hp)
            
        elif model_arch == "fine_tune":
            model = VGG16_fine_tune(weights='imagenet', hp=hp)
            
        elif model_arch == "clinic":
            model = clinic_model()
            
        elif model_arch == "mobilenet":
            model = mobilenetv3_transfer_learning(weights='imagenet', hp=hp)
            
        else:
            model = mobilenetv3_transfer_learning(weights='imagenet', hp=hp)
        
        alpha = lossp["alpha"]
        gamma = lossp["gamma"]
        learning_rate = hp["learning_rate"]
        
        if run["search"] == True:
            # Hyperparameter search: Learning rate, alpha, gamma
            alpha = hp.Choice('alpha', values=[0.25, 0.35, 0.50, 0.75])
            gamma = hp.Choice('gamma', values=[1.5, 2.0])
            pass
        
        # Define focal loss
        self.loss = BinaryFocalCrossentropy(alpha=alpha, gamma=gamma)
         
        self.opt = Adam(learning_rate=learning_rate,
                        beta_1=0.9, 
                        beta_2=0.999,
                        amsgrad=False)
        
        # Compile model
        model.compile(optimizer=self.opt, loss=self.loss, metrics=['accuracy', f1_score])
        
        self.model = model


    def extract_feature(self):
        @tf.function
        def predict_batch(model, batch):
            return model(batch, training=False)
        
        if run["create_feature"] or (not os.path.isdir(paths["feature"])):
            create_destroy_dic(paths["feature"])
            print("Feature directory created.")
            
            file_list = glob(paths["ready"] + "/*/*/*.npy")
            model = feature_extraction(paths["feature_model"])
            
            batch_size = 72  # keep low for memory
            batch_data = []
            batch_paths = []

            for fpath in tqdm(file_list, desc="Extracting features"):
                x = np.load(fpath).astype(np.float32)
                if x.ndim == 2:
                    x = np.repeat(x[..., np.newaxis], 3, axis=-1)

                batch_data.append(x)
                batch_paths.append(fpath)

                if len(batch_data) == batch_size or fpath == file_list[-1]:
                    batch_data_np = np.stack(batch_data)

                    feats = predict_batch(model, batch_data_np)

                    for feat, f in zip(feats, batch_paths):
                        rel_path = os.path.relpath(f, paths["ready"])
                        save_path = os.path.join(paths["feature"], rel_path)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        np.save(save_path, feat.numpy())  # convert tensor to numpy

                    # Cleanup
                    del batch_data_np, feats
                    batch_data = []
                    batch_paths = []

                    gc.collect()
                    tf.keras.backend.clear_session()  # optional: clear TF session to free memory

    def shap_analysis(self, evaluate, train_gen, test_gen, output_name, sample_size=50):
        background, _ = train_gen[0]
        
        total_img_pct = 0.0
        total_clinical_pct = 0.0
        count = 0
        
        clinical_accumulator = {}
        
        for X_sample, _ in test_gen:
            # Call SHAP
            shap_values, [img_pct, clinical_pct], clinical_dic = evaluate.shapley(self.model, X_sample, background, output_name)
        
            # Accumulate modality-level percentages
            total_img_pct += img_pct
            total_clinical_pct += clinical_pct
            count += 1
        
            # Accumulate clinical SHAP percent dictionary
            for feature, pct in clinical_dic.items():
                if feature not in clinical_accumulator:
                    clinical_accumulator[feature] = 0.0
                clinical_accumulator[feature] += pct
        
            print(f"[Batch {count}] img: {img_pct:.2f}%, clinical: {clinical_pct:.2f}%")
        
            if count >= sample_size:
                break
        
        # Compute averages
        avg_img_pct = total_img_pct / count
        avg_clinical_pct = total_clinical_pct / count
        
        # Average clinical feature contributions
        avg_clinical_dict = {k: v / count for k, v in clinical_accumulator.items()}
        
        # Sort and get top 10
        top10_clinical = sorted(avg_clinical_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Print results
        print("\n✅ Average SHAP Contribution Across Batches:")
        print(f"  img:      {avg_img_pct:.2f}%")
        print(f"  clinical: {avg_clinical_pct:.2f}%")
        
        print("\n🔬 Top 10 Clinical Features by Average SHAP Contribution (%):")
        for feature, pct in top10_clinical:
            print(f"  {feature}: {pct:.2f}%")
        return [avg_img_pct, avg_clinical_pct], clinical_dic   
                
    def fit(self):
        if self.hypp["model"] == "multi":
            self.extract_feature()
            
        df = pd.read_csv(paths["dataset"] + "imputed.csv")
        
        train_gen = DataGenerator("train", df, self.hypp["model"],self.hypp["IMG_SIZE"], self.hypp["batch_size"])
        val_gen = DataGenerator("val", df, self.hypp["model"], self.hypp["IMG_SIZE"], self.hypp["batch_size"])
        test_gen = DataGenerator("test", df, self.hypp["model"], self.hypp["IMG_SIZE"], 1)
        
        es_loss = EarlyStopping(monitor='val_loss', 
                                mode='min', 
                                verbose=1, 
                                patience=self.hypp["patience"])
        
        import tensorflow as tf
        tf.keras.backend.clear_session()
        gc.collect()
        tf.keras.backend.clear_session()
        if run["search"] == True:

            tuner = kt.GridSearch(
                build_model,
                objective=kt.Objective("val_f1_score", direction="max"),
                directory="f1_tuner",  # Ensures a persistent location
                project_name=model_arch,
                executions_per_trial= 5,
                max_trials=20000)
            
            valid_trials = [
                (trial_id, trial) for trial_id, trial in tuner.oracle.trials.items()
                if trial.score is not None
            ]
            
            # Sort trials by validation F1 score in descending order
            sorted_trials = sorted(valid_trials, key=lambda x: x[1].score, reverse=True)
            
            # Print top 10 trials
            for i, (trial_id, trial) in enumerate(sorted_trials[:10]):  
                print(f"Trial {trial_id} - Validation F1 Score: {trial.score:.4f}")
            
                # Get hyperparameters from the trial
                best_hps = trial.hyperparameters.values
                print(f"Best dense2: {best_hps.get('alpha')}")
                print(f"Best dense3: {best_hps.get('gamma')}")
                # print(f"Best dropout: {best_hps.get('dropout')}")
        
        
            tuner.search(train_gen,
                        epochs=self.hypp["epochs"],
                        validation_data=val_gen,
                        callbacks=[es_loss]
                        )
            
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
            print(f"Best Alpha: {best_hps.get('alpha')}")
            print(f"Best Gamma: {best_hps.get('gamma')}")
            print(best_hps.values)
        else:
            model = self.model
            " early stop and callbacks..."

            
            #create file organization
            today = date.today().strftime("%Y-%m-%d")
            savedirs = create_out_folder(today, self.hypp["model"])  
            print(savedirs)
            checkpoint = ModelCheckpoint(savedirs[-1] + "/best_weights.keras", 
                                        monitor='val_loss', 
                                        verbose=1, 
                                        save_best_only=True, 
                                        mode='min')
            
            callbacks_list = [checkpoint, es_loss]
            
            history = model.fit(
                x=train_gen,
                batch_size= self.hypp["batch_size"],
                epochs= self.hypp["epochs"],
                verbose=1,
                validation_data=val_gen,
                shuffle=True,
                callbacks=callbacks_list
            )
        
            evaluate = Evaluate(history, savedirs, model, self.loss)
            if self.hypp["model"] == "multi":
                pcts, pct_dict  = self.shap_analysis(evaluate, train_gen, test_gen, "icu", 50)
            else:
                pcts = [0,0]
                pct_dict = []
            evaluate(test_gen, pcts, pct_dict)

        
        



