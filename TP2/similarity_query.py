from unittest import result
import librosa  # https://librosa.org/
import librosa.display
import librosa.beat
import sounddevice as sd  # https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats as scs
from scipy.spatial.distance import (
    euclidean,
    cosine,
    cityblock,
)  # cityblock -> Manhattan

# Receives uniform signal and extracts features
def extract_features(signal: np.array):
    mean = signal.mean()
    stdDev = signal.std()
    skewness = scs.skew(signal)
    kurtosis = scs.kurtosis(signal)
    median = np.median(signal)
    max_value = signal.max()
    min_value = signal.min()
    return (mean, stdDev, skewness, kurtosis, median, max_value, min_value)


def similarity_metric(x, y, metric):
    if metric == "euclidean":
        return euclidean(x, y)
    elif metric == "cosine":
        return cosine(x, y)
    elif metric == "manhattan":
        return cityblock(x, y)


features = np.genfromtxt('exercise2_features_normalized.csv', delimiter=';')
list_songs = []

for i in range(1, 5):
    list_songs += os.listdir(f"dataset/MER_audio_taffc_dataset/Q{i}")
    
list_songs = np.array(sorted(list_songs))

query = "MT0000202045.mp3"

print(np.where(list_songs == query)[0][0]) 
query = features[10]

print(query.shape)

results = np.ones((900, 900), dtype=np.float32)

for m in ["euclidean", "cosine", "manhattan"]:
    for i in range(900):
        for j in range(900):
            results[i, j] = similarity_metric(query, features[j], m)
    np.savetxt(f"similarity_matrix/queries/MT0000202045/extracted_{m}.csv", results, delimiter=";")
    print(f"Done {m}")
