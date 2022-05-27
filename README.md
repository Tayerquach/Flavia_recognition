# Flavia_recognition 
=========================== 

This repository contains codes in the paper **An Effective Leaf Recognition Using  Convolutional Neural Networks Based  Features**, which is submitted to Multimedia Tools and Applications. 

# Preparation 
## Libraries 

Install required python packages by pip3 
```console 
pip3 install -r requirements.txt 
``` 

## Dataset 

Dataset is downloaded from https://sourceforge.net/projects/flavia/files/Leaf%20Image%20Dataset/1.0/Leaves.tar.bz2/download, all images are then extracted to folder "Leaves/". 

We prepare labels sorted by filenames in features_data/labels.npy. You can confirm the correction of the labels at http://flavia.sourceforge.net/. 

Due to the size limits on Github, we uploaded features' files (after preprocessing) at https://drive.google.com/drive/folders/12ZB9GLvzVX6mhR8hhG8OaM_Q9MsJ1T1_?usp=sharing 

## Kfold_file 
Kfold_file is a .csv file that follows the format of our Dataset_10FoldCV_indexed.csv, which must contain columns "Filename", "Fold_1", "Fold_2", ..., "Fold_10" . The order of leaves must be sorted by their filenames (numercial order). Values of "Fold" columns must be "Train", "Valid" or "Test". 

## Train encoders 

Training-encoder experiment file must be prepared beforehand. This file is to run training encoders with predefined l2 regularization and dropout rates. The file follows the format of train_encoders_example.csv and contains columns "kfold_file", "fold", "feature", "l2_rate", "dropout", "val_acc", "test_acc". 'feature' must be one of values 'image', 'vein', 'xyprojection', 'shape', 'color', 'texture', 'fourier'. 'fold' must in range 1 to 10. 

To train encoders, run 
```console 
python3 run_training_encoders.py train_encoders_example.csv 
``` 

This command creates a "{kfold_file}_models/ folder, trains and saves encoders to the folder in .h5 format. Filenames of the model follows the format "ENCODER-{feature}-l2rate{}-dropout{}-fold{}.h5" (defined in data_helper.py). 

## Select best encoders 

Copy all trained encoders into the "{kfold_file}_models/" folder and run 
```console 
python3 run_selecting_best_encoders.py Kfoldfile.csv 
``` 

This command looks for all encoders in the directory "{kfold_file}_models/", lists out their performances into "{Kfoldfile}_encoders_performances.csv" and select among them the best one each feature and each fold. The select encoder filenames are saved in "{Kfoldfile}_best_encoders.csv". 

# Feature analyses 

## Create feature files 
```console 
python3 prepare_features.py -o train_encoders_best.csv -f features 
``` 

File "*train_encoders_best.csv*" can be found in the directory 

*features*: image vein color shape texture fourier xyprojection

*Note*: All files are then saved in folder "features_data" 

E.g., 
```console 
python3 prepare_features.py -o train_encoders_best.csv -f color texture 
``` 

## Train features 
```console 
python3 run_training_encoders.py feature_file_csv 
``` 

**1) Training individual features** 

E.g., The vein feature group is found with the file path "features_data/vein_feature.csv". Then, we run script 
```console 
python3 run_training_features.py features_data/vein_feature.csv 
``` 

**2) Training combination** 

E.g., The color and texture feature groups are found with the file path "features_data/color_texture_feature.csv". Then, we run script 
```console 
python3 run_training_features.py features_data/color_texture_feature.csv 
``` 

# Final results 
Command line 
```console 
python3 run_training_decoders.py Kfoldfile.csv 
``` 

This command reads best encoders' filenames in "{Kfoldfile}\_best\_encoders.csv" and loads the corresponding saved .h5 files, trains decoders each fold to "{kfold_file}\_models/DECODER-fold{}.pickle". 

In the directory, Kfoldfile is *Dataset_10FoldCV_indexed* 

## Our results 

Our experiments' results on Dataset_10FoldCV_indexed.csv are saved on "LEAF_v20". In summary, we reached the result 

**End of 10-fold CV** 

*Valid accuracy*: 0.9979 +- 0.0035 

*Test accuracy*: 0.9969 +- 0.0035 

# Visualisation 
Command line 
```console 
python3 visualize_results.py -p predicted_data_file -a actual_data_file  -f false_images_file 
``` 

These files used can be found in **results** folder. 

E.g., 

*results/predicted_data.npy* 

*results/actual_data.npy* 

*results/false_data.npy* 

```console 
python3 visualize_results.py -p results/predicted_data.npy -a results/actual_data.npy -f results/false_data.npy 
``` 

Outputs are then saved in "results" folder. 

## Contact 
- Nguyen Thanh Binh  (University of Science Ho Chi Minh city, ngtbinh@hcmus.edu.vn) 
- Quach Mai Boi (Dublin City University, mai.quach3@mail.dcu.ie) 

## References 
<pre><code> @article{quach2021effective,
  title={An Effective Leaf Recognition Using Convolutional Neural Networks Based Features},
  author={Quach, Boi M and Cuong, Dinh V and Pham, Nhung and Huynh, Dang and Nguyen, Binh T},
  journal={arXiv e-prints},
  pages={arXiv--2108},
  year={2021} 
}</code></pre>
