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


def min_max_scale(y: np.array) -> np.float32:
    min_v = y.min()
    max_v = y.max()
    return (y - min_v) / (max_v - min_v)

if __name__=='__main__':
    feature_matrix = np.loadtxt("exercise2_features.csv", delimiter = ";")

    feature_matrix = np.apply_along_axis(min_max_scale, 1, feature_matrix)
    np.savetxt("exercise2_features_normalized.csv", feature_matrix, delimiter=";")

