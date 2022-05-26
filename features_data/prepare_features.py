import tensorflow as tf
import numpy as np
import pandas as pd
import re
import os
import cv2
from data_helper import load_features, load_labels, __model_file__, split_train_test_valid
from modeling import get_training_model, CheckpointCallback
from run_training_encoders import run_training, run_training_encoders
from matplotlib import pyplot as plt
from collections import Counter

l2_rate = 0.0008
dropout = 0.5

def find_number(text, c):
    return re.findall(r'%s(-?\d+\.\d+)' % c, text)

def substring_after(s, delim):
    return s.partition(delim)[0]

def create_feature_files(train_encoders_best_file, features): #features: array --> e.g., ["color", "vein"]
    experiments = pd.read_csv(train_encoders_best_file, index_col=[0])
    data = experiments[experiments['feature'].isin(features)]
    model_files = data['model_file'].unique()
    data = data.drop(columns={'feature', 'model_file', 'val_acc', 'test_acc', 'l2_rate' , 'dropout'})
    data = data.drop_duplicates(keep='first', inplace=False)
    data.reset_index(drop=True, inplace=True)
    data['feature'] = np.asarray([','.join(features) for i in range(len(data))])
    data['model_file'] = [substring_after(model_files[0], 'ENCODER') + 'ENCODER-{}-l2rate{}-dropout{}-fold{}.h5'.format('_'.join(features), l2_rate, dropout, i) for i in range(data.shape[0])]
    data['l2_rate'] = l2_rate* np.ones(10)
    data['dropout'] = dropout* np.ones(10)

    return data
