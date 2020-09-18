import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

input_path = '../../dataset/foreground/' 
output='../feat/mel_out/foreground/'

classes=['beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
         'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
         'office', 'park', 'residential_area', 'train', 'tram']

for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path +clas+'/'+file
        y, sr = librosa.load(filePath)
        S=librosa.feature.melspectrogram(y=y, sr=sr)
        #plt.figure(figsize=(4, 2))
        librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
        #librosa.display.specshow(librosa.power_to_db(S,ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        #plt.colorbar(format='%+2.0f dB')
        outFileName = output+clas+'/'+file.split('.')[0]+'.png'
        plt.savefig(outFileName)
        plt.close()

print("completed foreground")

input_path = '../../dataset/background/' 
output='../feat/mel_out/background/'

classes=['beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
         'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
         'office', 'park', 'residential_area', 'train', 'tram']

for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path +clas+'/'+file
        y, sr = librosa.load(filePath)
        S=librosa.feature.melspectrogram(y=y, sr=sr)
        #plt.figure(figsize=(4, 2))
        librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
        #librosa.display.specshow(librosa.power_to_db(S,ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        #plt.colorbar(format='%+2.0f dB')
        outFileName = output+clas+'/'+file.split('.')[0]+'.png'
        plt.savefig(outFileName)
        plt.close()

print("completed background")

input_path = '../../dataset/audio/' 
output='../feat/mel_out/audio/'

classes=['beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
         'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
         'office', 'park', 'residential_area', 'train', 'tram']

for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path +clas+'/'+file
        y, sr = librosa.load(filePath, mono=False, sr=44100)
        y_mono = librosa.to_mono(y)
        S=librosa.feature.melspectrogram(y=y_mono, sr=sr)
        #plt.figure(figsize=(4, 2))
        librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
        #librosa.display.specshow(librosa.power_to_db(S,ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        #plt.colorbar(format='%+2.0f dB')
        outFileName = output+clas+'/'+file.split('.')[0]+'.png'
        plt.savefig(outFileName)
        plt.close()

print("completed audio")