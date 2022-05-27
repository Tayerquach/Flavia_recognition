import argparse
parser = argparse.ArgumentParser(description='Train encoders.')

parser.add_argument('expfile', type=str,
					help='Experiment file.')

from data_helper import load_features, load_labels, __model_file__, split_train_test_valid
from modeling import get_training_model, CheckpointCallback
import pandas as pd
import numpy as np
import os

def run_training(features, kfold, fold, l2_rate, dropout):
	features = features.split(',')
	"""run a single training
	"""
	print("==========================")
	if len(features) > 1:
		print("Training feature {} - l2_rate {} - dropout {} - fold {}".format('_'.join(features), l2_rate, dropout, fold))
		model_path = __model_file__.format('_'.join(features), l2_rate, dropout, fold)
	else:
		print("Training feature {} - l2_rate {} - dropout {} - fold {}".format(features[0], l2_rate, dropout, fold))
		model_path = __model_file__.format(features[0], l2_rate, dropout, fold)
	
	X = load_features(features)
	y = load_labels()

	outdir = kfold[:-4] + "_models/"
	if not os.path.exists(outdir):
		os.mkdir(outdir)

	kfold = pd.read_csv(kfold)

	for feature in features:
		if feature in ['vein', 'image']:
			max_epochs = 200
		else:
			max_epochs = 1000000

	(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_train_test_valid(features, kfold, fold, X, y)
	model_path = os.path.join(outdir, model_path)

	checkpoint = CheckpointCallback(verbose=False)
	# with strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
	model = get_training_model(features, l2_rate=l2_rate, dropout=dropout)

	model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
				verbose=0,
				epochs=max_epochs,
				batch_size=128,
				callbacks=[checkpoint])
	model.save_weights(model_path)

	_, train_acc = model.evaluate(X_train, y_train, verbose=0)
	_, val_acc = model.evaluate(X_valid, y_valid, verbose=0)
	_, test_acc = model.evaluate(X_test, y_test, verbose=0)

	print("Train_time {:.4f}, train_acc {:.4f}, val_acc {:.4f}, test_acc {:.4f}".format(checkpoint.training_time, train_acc, val_acc, test_acc))

	return val_acc, test_acc

def run_training_features(experiment_file):
	"""run training from a csv file,
    save trained models to 'kfold[:-4]+"_models/"' directory
	"""
	experiments = pd.read_csv(experiment_file)
	for exp_i in range(len(experiments)):
		kfold_file, folds, features, l2_rate, dropout = experiments.iloc[exp_i][['kfold_file','fold','feature','l2_rate','dropout']]
		val_acc, test_acc = run_training(features, kfold_file, folds, l2_rate, dropout)

		experiments.at[exp_i, 'val_acc'] = val_acc
		experiments.at[exp_i, 'test_acc'] = test_acc

		experiments.to_csv(experiment_file, index=False)
	print("Complete training ", experiment_file)

if __name__ == "__main__":
	args = parser.parse_args()
	run_training_features(args.expfile)