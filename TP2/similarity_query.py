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


def get_songs():
    list_songs = []
    for i in range(1, 5):
        list_songs += os.listdir(f"dataset/MER_audio_taffc_dataset/Q{i}")

    return list_songs


features = np.loadtxt("similarity_matrix/extracted_cosine.csv", delimiter=";")
#top_100 = np.loadtxt("top_100_features_normalized.csv", delimiter=";")

list_songs = np.array(sorted(get_songs()))

query = "MT0000202045.mp3"
i = np.where(query == list_songs)[0][0]


results_extracted = np.ones((900,), dtype=np.float32)
results_top_100 = np.ones((900,), dtype=np.float32)

