
import argparse
parser = argparse.ArgumentParser(description='Training decoders.')

parser.add_argument('kfoldfile', type=str,
					help='10fold cross validation file.')

import pandas as pd
import numpy as np
import os
from modeling import EncoderExtractor
from data_helper import load_features, load_labels, __features__, split_train_test_valid, normalize_feature_data, __decoder_file__
from sklearn.svm import SVC
import pickle
from collections import Counter

__C_values__ = [1e3, 1e4, 1e5]
__gamma_values__ = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

#Read data
image_labels = pd.read_csv('Dataset_10FoldCV_indexed.csv')
flowers = image_labels['Flower'].values
counter = Counter(flowers)
class_word_names = np.array(sorted(list(counter.keys())))
images = load_features('image')


def run_training_decoders(kfold_file):
    predicted_targets = np.array([])
    actual_targets = np.array([])
    false_data    = []
    kfold = pd.read_csv(kfold_file)

    best_encoders = pd.read_csv(kfold_file[:-4] + "_best_encoders.csv")
    best_decoders = pd.DataFrame(columns=['kfold_file', 'fold', 'model_path', 'C', 'gamma', 'val_acc', 'test_acc'])
    extractor = EncoderExtractor()
    X = load_features(__features__)
    y = load_labels()
    
    i = 0
    val_accs, test_accs = [], [] 
    for fold in range(1,11):
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_train_test_valid(__features__, kfold, fold, X, y)

        test_label_fold = image_labels[image_labels[f'Fold_{fold}'] == 'Test']
        image_arr = images[test_label_fold.index]

        model_paths = []
        for feature in __features__:
            encoder_file = best_encoders[np.logical_and(best_encoders['fold'] == fold, best_encoders['feature'] == feature)].iloc[0]['model_file']
            model_paths.append(encoder_file)
        extractor.load_encoders(model_paths)

        X_train = extractor.extract(X_train)
        X_valid = extractor.extract(X_valid)
        X_test = extractor.extract(X_test)
        X_train, X_valid, X_test = normalize_feature_data("combine",  X_train, X_valid, X_test)

        best_val_acc, best_test_acc = 0.0, 0.0
        output_path = os.path.join(kfold_file[:-4] + "_models/", __decoder_file__.format(fold))
        bad_C_gamma_values = [] 
        for C in __C_values__:
            for gamma in __gamma_values__:
                if [C, gamma] in bad_C_gamma_values:
                    continue
                decoder = SVC(gamma=gamma, C=C)
                decoder.fit(X_train, y_train)
                val_acc = np.mean(decoder.predict(X_valid) == y_valid)
                test_acc = np.mean(decoder.predict(X_test) == y_test)
                y_pred = decoder.predict(X_test)
                if val_acc < 0.98:
                    bad_C_gamma_values.append([C, gamma])
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_y_pred = y_pred
                    best_C, best_gamma = C, gamma
                    with open(output_path, "wb") as outfile:
                        pickle.dump(decoder, outfile)

        indices = np.where(best_y_pred != y_test)[0]
        for id in indices:
          image_data = (image_arr[id], class_word_names[y_test[id]], class_word_names[best_y_pred[id]]) #image arr, true, false
          false_data.append(image_data)

        predicted_targets = np.append(predicted_targets, best_y_pred)
        actual_targets = np.append(actual_targets, y_test)

        print("Fold {} - val_acc {} - test_acc {}".format(fold, best_val_acc, best_test_acc))
        val_accs.append(best_val_acc)
        test_accs.append(best_test_acc)
        best_decoders.loc[i] = [kfold_file, fold, output_path, best_C, best_gamma, best_val_acc, best_test_acc]
        i += 1

    print("End of 10-fold CV")
    print("Valid accuracy: {:.4f} +- {:.4f}".format(np.mean(val_accs), np.std(val_accs)))
    print("Test accuracy: {:.4f} +- {:.4f}".format(np.mean(test_accs), np.std(test_accs)))
    best_decoders.to_csv(kfold_file[:-4] + "_decoders.csv", index=False)
    #Save data
    np.save('results/predicted_data.npy', predicted_targets)
    np.save('results/actual_data.npy', actual_targets)
    np.save('results/false_data.npy', false_data)

if __name__ == "__main__":
    args = parser.parse_args()
    run_training_decoders(args.kfoldfile)