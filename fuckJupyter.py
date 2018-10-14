import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D, SpatialDropout1D, Flatten
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import re

import seaborn as sns

mfccs = pd.read_csv('soundAnalysis/happySad_MFCC.csv')
mfccs.head()
print(type(mfccs['MFCC'].values))

max_features = 2000
tokeniser = Tokenizer(num_words=max_features, split=' ')
tokeniser.fit_on_texts(mfccs['MFCC'].values)
x = tokeniser.texts_to_sequences(mfccs['MFCC'].values)
x = pad_sequences(x, maxlen=221)

embedding_dimensions = 128
lstm_out = 200
batch_size = 32
model = Sequential()
model.add(Embedding(2500, embedding_dimensions,input_length = x.shape[1], dropout = 0.2))
model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

print(mfccs.size)
y = pd.get_dummies(mfccs["Sentiment"]).values
batch_size = 64
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 1234)

earlyStopping=keras.callbacks.EarlyStopping(monitor='val_acc', min_delta = 0.1, patience=0, verbose=2, mode='auto')

model.fit(X_train, Y_train,
  batch_size=32,
  epochs=5,
  verbose=1,
  validation_data = (X_test, Y_test))

model.save("speechAnalysisMFCCModel.hdf5")