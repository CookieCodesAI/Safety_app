import numpy as np
import os
import pathlib
import glob
import librosa

def filename(file):
    name = os.path.basename(file).split('_')
    if name[2] == "ANG": 
        group = 0
    elif name[2] == "DIS":
        group = 1
    elif name[2] == "FEA":
        group = 2
    elif name[2] == "HAP":
        group = 3
    elif name[2] == "NEU":
        group = 4
    elif name[2] == "SAD":
        group = 5
    return group

#load audio
def structure_data():
    files = glob.glob("data/CREAMA-D/AudioWAV/*.wav")
    PATH = "data/CREAMA-D/AudioWAV"
    data = pathlib.Path(PATH)
    if not data.exists():
        print("The Dataset does not exist")
    X_data = []
    y_data = []
    for file in files:
        y, sr = librosa.load(file, sr = 16000)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        dbs = librosa.power_to_db(S)
        dbs = dbs.astype(np.float32)
        dbs = (dbs - np.min(dbs)) / (np.max(dbs) - np.min(dbs)+ 1e-6)
        max_len = 180
        width = dbs.shape[1]
        if width < max_len:
            db_fixed = np.pad(dbs, ((0,0), (0, (max_len-width))), mode = "constant" )
        else:
            db_fixed = dbs[:, :max_len]
        db_fixed = np.expand_dims(db_fixed, axis = -1)
        X_data.append(db_fixed)
        y_data.append(filename(file))
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    return X_data, y_data
