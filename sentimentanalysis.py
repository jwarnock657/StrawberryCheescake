import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D, SpatialDropout1D, Flatten
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import seaborn as sns
print("OK")


## Read in the training data
text = pd.read_csv("train_kaggle.csv", encoding = "ISO-8859-1")
text.head()
## Set all to lower case 
text['SentimentText'] = text['SentimentText'].apply(lambda x : re.sub('[^a-zA-Z0-9\s]', '', x.lower()))

## Tokenizse 
max_features = 2000
tokeniser = Tokenizer(num_words=max_features, split=' ')
tokeniser.fit_on_texts(text['SentimentText'].values)
x = tokeniser.texts_to_sequences(text['SentimentText'].values)
x = pad_sequences(x, maxlen=200)
print (len(x[0]))

#embedding_dimensions = 128
#lstm_out = 200 
#batch_size = 32

##build the model
#model = Sequential()
#model.add(Embedding(2500, embedding_dimensions,input_length = x.shape[1]))
#model.add(Dropout(rate=0.2))
#model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
#model.add(Dense(2,activation='softmax'))
#model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
#print(model.summary())
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
##print(tweets.size)
y = pd.get_dummies(text['Sentiment']).values
batch_size = 406
##split data into training and test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 1234)

## stop if loss drops by 0.1 or more 
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_acc', min_delta = 0.1, patience=0, verbose=2, mode='auto')

## Train the model Yeepaaaa
print(X_train.size, X_train)
model.fit(X_train, Y_train, epochs=5, batch_size = batch_size, validation_data=(X_test, Y_test), shuffle=True, callbacks=[earlyStopping])

## Evaluate and calculate accuracy 
validation_size = 1500
X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


model_json = model.to_json()
with open("WordModel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("WordsModel.h5")
print("Saved model to disk")
#twt = ["I am so so sad please help me im sad."]

#tokeniser = Tokenizer(num_words=max_features, split=' ')
#print("create tokenizer")
#tokeniser.fit_on_texts(twt)
#twt = tokeniser.texts_to_sequences(twt)
#twt = pad_sequences(twt, maxlen=200)
#sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
#if(np.argmax(sentiment) == 0):
#    print("negative")
#elif (np.argmax(sentiment) == 1):
#    print("positive")