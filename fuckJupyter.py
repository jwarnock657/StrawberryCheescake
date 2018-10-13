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

tweets = pd.read_csv('train_kaggle.csv', encoding = "ISO-8859-1")
tweets.head()

tweets['SentimentText'] = tweets['SentimentText'].apply(lambda x : re.sub('[^a-zA-Z0-9\s]', '', x.lower()))
print(tweets['SentimentText'][0])

max_features = 2000
tokeniser = Tokenizer(num_words=max_features, split=' ')
tokeniser.fit_on_texts(tweets['SentimentText'].values)
x = tokeniser.texts_to_sequences(tweets['SentimentText'].values)
x = pad_sequences(x)
print (len(x[0]))

embedding_dimensions = 128
lstm_out = 200
batch_size = 32
#custom_adam = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model = Sequential()
model.add(Embedding(2500, embedding_dimensions,input_length = x.shape[1], dropout = 0.2))
model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
#print(model.summary())

print(tweets.size)
y = pd.get_dummies(tweets['Sentiment']).values
batch_size = 64
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 1234)

earlyStopping=keras.callbacks.EarlyStopping(monitor='val_acc', min_delta = 0.1, patience=0, verbose=2, mode='auto')

model.fit(X_train, Y_train,
  batch_size=32,
  epochs=5,
  verbose=1,
  validation_data = (X_test, Y_test))