import wavio
import os
import scipy.signal as signal
import pandas as pd
import re
import matplotlib.pyplot as plt
import librosa
import numpy as np

data = []

for actor in os.listdir(os.path.join(os.getcwd(), "soundAnalysis","voiceSamples")):
    for file_ in os.listdir(os.path.join(os.getcwd(), "soundAnalysis", "voiceSamples", actor)):

        if file_.split("-")[2] in ["03", "04"]:

            filePath = os.path.join(os.getcwd(), "soundAnalysis", "voiceSamples", actor, file_)

            print(file_)
            wf = wavio.read(filePath)
            X, sample_rate = librosa.load(filePath, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X,
                                                sr=sample_rate,
                                                n_mfcc=13),
                            axis=0)
            data.append((int(file_.split("-")[2]), mfccs))


df = pd.DataFrame(data)
df.to_csv("happySad_MFCC.csv")
