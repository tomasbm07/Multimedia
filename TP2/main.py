import librosa #https://librosa.org/
import librosa.display
import librosa.beat
import sounddevice as sd  #https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats as scs


def extract_features(signal : np.array):
    mean = signal.mean()
    stdDev = signal.std()
    skewness = scs.skew(signal)
    kurtosis = scs.kurtosis(signal)
    median = np.median(signal)
    max_value = signal.max()
    min_value = signal.min()
    return (mean, stdDev, skewness, kurtosis, median, max_value, min_value)


if __name__=='__main__':
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")

    path = "dataset/MER_audio_taffc_dataset/Q"
    music_df = pd.DataFrame(columns=[""])
    #910x190
    feature_matrix = np.zeros((900, 190))
    index=0
    for i in range(1,5):
        for audio in os.listdir(f"{path}{i}"):
            print(index)
            y, fs = librosa.load(f"{path}{i}/{audio}", sr=sr, mono = mono)
            #Spectral features extraction
            mfcc = np.apply_along_axis(extract_features, 1, librosa.feature.mfcc(y=y, n_mfcc=13)).flatten()
            spc_centroid= np.apply_along_axis(extract_features, 1, librosa.feature.spectral_centroid(y=y)).flatten()
            spc_bdwth = np.apply_along_axis(extract_features, 1, librosa.feature.spectral_bandwidth(y=y)).flatten()
            spc_contrast = np.apply_along_axis(extract_features, 1, librosa.feature.spectral_contrast(y=y)).flatten()
            spc_flatness = np.apply_along_axis(extract_features, 1, librosa.feature.spectral_flatness(y=y)).flatten()
            spc_rollof = np.apply_along_axis(extract_features, 1, librosa.feature.spectral_rolloff(y=y)).flatten()
            f0 = np.apply_along_axis(extract_features, 0, librosa.yin(y, 20, 11025))
            f0[f0==11025] = 0;
            rms = np.apply_along_axis(extract_features, 1, librosa.feature.rms(y=y)).flatten()
            zcr = np.apply_along_axis(extract_features, 1, librosa.feature.zero_crossing_rate(y=y)).flatten()
            tempo = librosa.beat.tempo(y=y)
            
            feature_vector = np.concatenate((mfcc,spc_centroid, spc_bdwth, spc_contrast, spc_flatness, spc_rollof,
                                            f0, rms, zcr, tempo))
            
            feature_matrix[index]=feature_vector;
            index+=1;

    np.apply_along_axis(librosa.util.normalize, 1, feature_matrix)
    feature_matrix.tofile("exercise2_features.csv", sep = ";")
    del feature_matrix