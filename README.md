# LearningToSeparate

This repository contains python implementation of our paper "Learning to Separate: Soundscape Classification using Foreground and Background".

## Getting Started

These instructions will help you to run python programs in sequence.

### Steps to decompose the audio files into the foreground and the background using rpca

1. Download or clone this repository into local system.
2. Download DCASE 2017 ASC(task 1) dataset and extract all the zip files into single folder.
3. Copy all the extracted wav files to folder "<path_to_repo_download>/LearningToSeparate/dataset/stereo/"
4. Navigate to "<path_to_repo_download>/LearningToSeparate/dataset/" and run python program "sterio2mono.py".
```
cd <path_to_repo_download>/LearningToSeparate/dataset/
python sterio2mono.py
```
4. Run the following command to copy the audio files into respective class folders.
```
python copyfiles.py
```
5. Run the following matlab program to decompose all the audio files into foreground and background.
```
cd ../rpca
matlab -nodisplay -r readfilenames
```
6. Run the following commands to copy the foreground and the background into seperate folders.
```
cp -r example/output/*_E.wav ../dataset/rpca_out_foreground/
cp -r example/output/*_A.wav ../dataset/rpca_out_background/
```
7. Run the following commands to remove substring "_A" and "_E" from all the file names.
```
cd ../dataset/rpca_out_foreground/
rename 's/_E//g' *.wav

cd ../rpca_out_background/
rename 's/_A//g' *.wav
```
8. Run the following command to copy the audio files into respective class folders.
```
cd ..
python copyfiles_fg.py
python copyfiles_bg.py
```
### Steps to extract log-mel vgg features and perform other experiments on log-mel features
1. Run the following program to generate log-mel spectrogram.
```
python mel_spec/code/read_mel.py

```
2. Run the following program to extract VGG feature for audio, foreground and background.
```
python mel_spec/code/conv_feat.py
python mel_spec/code/conv_feat_fg.py
python mel_spec/code/conv_feat_bg.py
```
3. Run the following program to generated 4 fold numpy arrays for audio, foreground and background.
```
python mel_spec/code/test_copy_audio.py
python mel_spec/code/test_copy_audio_fg.py
python mel_spec/code/test_copy_audio_bg.py

python mel_spec/code/train_copy_audio.py
python mel_spec/code/train_copy_audio_fg.py
python mel_spec/code/train_copy_audio_bg.py
```
4. Run the following program to generate basis for foreground and background.
```
python mel_spec/code/pca_single_basis_fg.py
python mel_spec/code/pca_single_basis_bg.py
```
5. Run the following program to get baseline system accuracy.
```
python mel_spec/code/svm_audio.py
python mel_spec/code/svm_bg.py
python mel_spec/code/svm_fg.py
```
6. Run the following program to get the results for VGG16 after suppressing foreground and after suppressing the background. Amount of foregroud or background to be suppressed can be changed by varying p and q values in the program. 
```
python mel_spec/code/classify_single_basis_fg.py
python mel_spec/code/classify_single_basis_bg.py
```
7. Run the following program to get the results for early and late fussion. Amount of foregroud and background to be suppressed can be changed by varying p, q (for foreground) and e, f (for background) values in the program. 
```
python mel_spec/code/classify_single_basis_fussion.py
python mel_spec/code/classify_single_basis_late_fussion.py
```

### Steps to extract L3-Net features and perform other experiments on L3-Net features

1. Run the following program to generate L3-Net features.
```
python l3net/code/l3_feat.py

```
2. Run the following program to generated 4 fold numpy arrays for audio, foreground and background.
```
python l3net/code/test_copy_audio.py
python l3net/code/test_copy_fg.py
python l3net/code/test_copy_bg.py

python l3net/code/train_copy_audio.py
python l3net/code/train_copy_fg.py
python l3net/code/train_copy_bg.py
```
4. Run the following program to generate basis for foreground and background.
```
python l3net/code/pca_single_basis_fg_sklearn.py
python l3net/code/pca_single_basis_bg_sklearn.py
```
5. Run the following program to get baseline system accuracy.
```
python l3net/code/svm_audio.py
python l3net/code/svm_bg.py
python l3net/code/svm_fg.py
```
6. Run the following program to get the results for L3-Net after suppressing foreground and after suppressing the background. Amount of foregroud or background to be suppressed can be changed by varying p and q values in the program. 
```
python l3net/code/classify_single_basis_fg.py
python l3net/code/classify_single_basis_bg.py
```
7. Run the following program to get the results for early and late fussion. Amount of foregroud and background to be suppressed can be changed by varying p, q (for foreground) and e, f (for background) values in the program. 
```
python l3net/code/classify_single_basis_fussion.py
python l3net/code/classify_single_basis_late_fussion.py
```
