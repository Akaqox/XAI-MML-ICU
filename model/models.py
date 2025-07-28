# -*- coding: utf-8 -*-

"""
@authors: Akaqox(S.Kizilisik) (kzlsksalih@gmail.com)

This script is experimental codes of paper "Explainable Multimodal Machine Learning Model
for Predicting Intensive Care Unit Admission " by S.Kizilisik et al. You may use the codes only for research. 
Plese cite the paper if you use any part of the codes.

"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate
from keras.models import Model, model_from_json
from keras.callbacks import LearningRateScheduler, Callback, ReduceLROnPlateau
from keras.layers import GlobalAveragePooling2D, BatchNormalization
from keras.applications import VGG16, MobileNetV3Small
from keras.utils import custom_object_scope
from utils.metrics import f1_score

def VGG16_Multi(weights='imagenet', hp = None):
    with custom_object_scope({"f1_score": f1_score}):
        base_model = tf.keras.models.load_model('model/fine_tune.keras')
    
    # Extract only the convolutional base (excluding the last 3 layers)
    cnn_part = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-7].output)
     
    # Freeze the CNN layers (optional)
    for layer in cnn_part.layers:
        layer.trainable = False
    # Define the primary image input
    image_input = cnn_part.input  # This will be (224, 224, 3)

    # Define the additional input (e.g., mask input)
    tabular_input = Input(shape=(72,), dtype="float32", name="tabular_input")

    # Process image input through the base model
    x = cnn_part.output
    
    x = Flatten()(x)
    
    # x = Dense(units=128, activation="relu")(x)
    tabular_input = Flatten()(tabular_input)
    x = Concatenate(axis=-1)([x, tabular_input])
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(units=64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(units=64, activation="relu")(x)
    output = Dense(units=1, activation="sigmoid")(x)

    # Define the complete model
    model = Model(inputs=[image_input, tabular_input], outputs=output)

    # Print summary
    model.summary()
    
    return model

def clinic_model(hp = None):
    # Define the additional input (e.g., mask input)
    tabular_input = Input(shape=(72,), dtype="float32", name="mask_input")
    x = BatchNormalization()(tabular_input)
    x = Dropout(0.3)(x)
    x = Dense(units=32, activation="relu")(tabular_input)
    x = Dropout(0.3)(x)
    x = Dense(units=32, activation="relu")(x)
    output = Dense(units=1, activation="sigmoid")(x)
    # Define the complete model
    model = Model(inputs=tabular_input, outputs=output)
    return model

def VGG16_fine_tune(weights='imagenet', hp = None):
    # Define the base VGG16 model
    base_model = VGG16(weights=weights, 
                       include_top=False, 
                       input_shape=(224, 224, 3))

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers[:-3]:
        layer.trainable = False

    # Define the primary image input
    image_input = base_model.input  # This will be (224, 224, 3)

    # Process image input through the base model
    x = base_model.output
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    
    x = Flatten()(x)
    
    dense1 = 32
    dense2 = 32
    dropout = 0.1
    
    x = Dropout(dropout)(x)    
    x = Dense(units=dense1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(units=dense2, activation="relu")(x)
    
    output = Dense(units=1, activation="sigmoid")(x)

    # Define the complete model
    model = Model(inputs=image_input, outputs=output)

    # Print summary
    model.summary()
    
    return model

def mobilenetv3_transfer_learning(weights='imagenet' ,hp = None):
    
    
    base_model = MobileNetV3Small(weights=weights, 
                                  include_top=False, 
                                  input_shape=(224, 224, 3),
                                  alpha=1.0,
                                  minimalistic=False,
                                  input_tensor=None,
                                  pooling=None,
                                  dropout_rate=0.1,
                                  classifier_activation="softmax",
                                  include_preprocessing=False,)

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
        
    # Define the primary image input
    image_input = base_model.input  # This will be (224, 224, 3)
    
    # Process image input through the base model
    x = base_model.output
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    output = Dense(units=1, activation="sigmoid")(x)
    
    # Define the complete model
    model = Model(inputs=image_input, outputs=output)
    model.summary()
    return model

def feature_extraction(model_path='model/cnn_mnet.keras'):
    with custom_object_scope({"f1_score": f1_score}):
        full_model = tf.keras.models.load_model(model_path)
    
    # Choose the output of the layer two before last
    target_layer = full_model.layers[-2]  # -1 is last, -2 second last, -3 third last
    
    feature_extractor = tf.keras.Model(inputs=full_model.input, outputs=target_layer.output)
    return feature_extractor

def mnet_Multi(weights='imagenet', hp = None):
    
    feature_input = Input(shape=(576,), name="feature_input")
    tabular_input = Input(shape=(72,), name="tabular_input")

    # Flatten both
    x1 = Flatten(name="flatten_feature")(feature_input)
    x2 = Flatten(name="flatten_tabular")(tabular_input)

    # Concatenate
    x = Concatenate(name="concat")([x1, x2])
    
    x = BatchNormalization(name="normalization")(x)
    x = Dropout(0.3)(x)
    x = Dense(units=64, activation="relu", name="dense1")(x)
    x = Dropout(0.3)(x)
    x = Dense(units=64, activation="relu")(x)
    output = Dense(units=1, activation="sigmoid", name="icu")(x)

    # Define the complete model
    model = Model(inputs=[feature_input, tabular_input], outputs=output)

    # # Print summary
    model.summary()
    
    return model