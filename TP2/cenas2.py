import pandas as pd
import os

top_100 = pd.read_csv("top_100_features_normalized.csv", header=None)
names = []

for i in range(1, 5):
    path = f"dataset/MER_audio_taffc_dataset/Q{i}"
    for m in os.listdir(path):
        names.append(m)

top_100.insert(0, None, value = names, allow_duplicates = False)
top_100.to_csv("teste.csv",  sep = ";", index = False)
