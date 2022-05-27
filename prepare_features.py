import numpy as np
import pandas as pd
import re
import os


import argparse
parser = argparse.ArgumentParser(description='Train encoders.')

parser.add_argument('-o', '--expfile', type=str,
					help='Experiment file.')
parser.add_argument('-f', '--features', nargs='+', default=[])


l2_rate = 0.0008
dropout = 0.5

def find_number(text, c):
    return re.findall(r'%s(-?\d+\.\d+)' % c, text)

def substring_after(s, delim):
    return s.partition(delim)[0]

def create_features(train_encoders_best_file, features): #features: array --> e.g., ["color", "vein"]
    experiments = pd.read_csv(train_encoders_best_file, index_col=[0])
    data = experiments[experiments['feature'].isin(features)]
    model_files = data['model_file'].unique()
    l2_rates_values = data['l2_rate'].values
    dropout_values = data['dropout'].values
    data = data.drop(columns={'feature', 'model_file', 'val_acc', 'test_acc', 'l2_rate' , 'dropout'})
    data = data.drop_duplicates(keep='first', inplace=False)
    data.reset_index(drop=True, inplace=True)
    data['feature'] = np.asarray([','.join(features) for i in range(len(data))])
    data['model_file'] = [substring_after(model_files[0], 'ENCODER') + 'ENCODER-{}-l2rate{}-dropout{}-fold{}.h5'.format('_'.join(features), l2_rate, dropout, i) for i in range(data.shape[0])]
    if len(features) != 1:
        data['l2_rate'] = l2_rate* np.ones(10)
        data['dropout'] = dropout* np.ones(10)
    else:
        data['l2_rate'] = l2_rates_values
        data['dropout'] = dropout_values

    return data

if __name__ == "__main__":
    args = parser.parse_args()
    features = args.features
    data = create_features(args.expfile, features)
    #Save data
    name_file = "features_data/" + "_".join(features) + "_feature.csv"
    data.to_csv(name_file, index=False)
    print(f"File {name_file} already saved.")
