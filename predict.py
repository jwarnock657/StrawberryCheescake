import pandas as pd
import librosa
import numpy as np
import os
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib


from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

model = load_model("speechAnalysisMFCCModel.hdf5")

def predict(mfccs, happySad):
    #mfccs = pd.read_csv('soundAnalysis/happySad_MFCC.csv')
    #mfccs.head()



    max_features = 2000
    tokeniser = Tokenizer(num_words=max_features, split=' ')
    tokeniser.fit_on_texts(mfccs)
    x = tokeniser.texts_to_sequences(mfccs)
    x = pad_sequences(x, maxlen=221)
    prediction = model.predict(x)

    print(happySad, prediction, mfccs[:50], np.average(prediction))




if __name__ =="__main__":
    dir = 'soundAnalysis/voiceSamples/Actor_01/'

    files = [f for f in os.listdir(dir) if f.split("-")[2] in ["03", "04"]]
    print(files)

    for f in files:

        X, sample_rate = librosa.load(dir+f, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X,
                                             sr=sample_rate,
                                             n_mfcc=13),
                        axis=0)

        mfccs = " ".join([str(f) for f in mfccs.data])

        predict(mfccs, int(f.split("-")[2])%2)