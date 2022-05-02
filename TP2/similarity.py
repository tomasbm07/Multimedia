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


features_extracted = np.loadtxt('exercise2_features_normalized.csv', delimiter=';')
features_top_100 = np.loadtxt("top_100_features_normalized.csv", delimiter=";")

results_extracted = np.ones((900, 900), dtype=np.float32)
results_top_100 = np.ones((900, 900), dtype=np.float32)

for m in ["euclidean", "cosine", "manhattan"]:
    for i in range(900):
        for j in range(900):
            results_extracted[i, j] = similarity_metric(features_extracted[i], features_extracted[j], m)            
            results_top_100[i, j] = similarity_metric(features_top_100[i], features_top_100[j], m)
    
    np.savetxt(f"similarity_matrix/extracted_{m}.csv", results_extracted, fmt = "%1.6f", delimiter=";")
    print(f"Done extracted {m}")
    
    np.savetxt(f"similarity_matrix/top100_{m}.csv", results_top_100, fmt = "%1.6f", delimiter=";")
    print(f"Done top_100 {m}")
