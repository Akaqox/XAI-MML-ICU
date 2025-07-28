import tensorflow as tf
from tensorflow.keras import backend as K

def f1_score(y_true, y_pred):
    """
    F1 Score: A measure of a model's accuracy.
    """
    y_true = K.round(y_true)  # Round the true values
    y_pred = K.round(y_pred)  # Round the predicted values

    tp = K.sum(K.cast(y_true * y_pred, 'float32'))  # True positives
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float32'))  # True negatives
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'))  # False positives
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'))  # False negatives

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())  # F1 score
    return f1    

