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



if __name__=='__main__':
	
    feature_matrix = np.loadtxt("exercise2_features.csv", delimiter = ";")

    feature_matrix = np.apply_along_axis(librosa.util.normalize, 1, feature_matrix)
    np.savetxt("teste.csv", feature_matrix, delimiter=";")
    del feature_matrix
