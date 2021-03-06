import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import pandas as pd
from keras.models import model_from_json

from keras import regularizers
import os
import csv
import threading
import time

import os
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import librosa
import glob

import pyaudio
import wave


def train():
    mylist2 = []
    mylist = []
    for i in os.listdir("soundAnalysis/voiceSamples"):
        [mylist.append(f) for f in os.listdir(os.path.join("soundAnalysis/voiceSamples", i))]
        [mylist2.append(os.path.join("soundAnalysis/voiceSamples", i, f)) for f in
         os.listdir(os.path.join("soundAnalysis/voiceSamples", i))]

    feeling_list = []
    for item in mylist:
        if item[6:-16] == '02' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_calm')
        elif item[6:-16] == '02' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_calm')
        elif item[6:-16] == '03' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_happy')
        elif item[6:-16] == '03' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_happy')
        elif item[6:-16] == '04' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_sad')
        elif item[6:-16] == '04' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_sad')
        elif item[6:-16] == '05' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_angry')
        elif item[6:-16] == '05' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_angry')
        elif item[6:-16] == '06' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_fearful')
        elif item[6:-16] == '06' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_fearful')
        elif item[:1] == 'a':
            feeling_list.append('male_angry')
        elif item[:1] == 'f':
            feeling_list.append('male_fearful')
        elif item[:1] == 'h':
            feeling_list.append('male_happy')
        # elif item[:1]=='n':
        # feeling_list.append('neutral')
        elif item[:2] == 'sa':
            feeling_list.append('male_sad')

    labels = pd.DataFrame(feeling_list)

    df = pd.DataFrame(columns=['feature'])

    bookmark = 0
    for index, y in enumerate(mylist):
        if mylist[index][6:-16] != '01' and mylist[index][6:-16] != '07' and mylist[index][6:-16] != '08' and mylist[
                                                                                                                  index][
                                                                                                              :2] != 'su' and \
                mylist[index][:1] != 'n' and mylist[index][:1] != 'd':
            file_ = mylist2[index]
            print(bookmark)

            X, sample_rate = librosa.load(file_, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X,
                                                 sr=sample_rate,
                                                 n_mfcc=13),
                            axis=0)
            feature = mfccs
            # [float(i) for i in feature]
            # feature1=feature[:135]
            df.loc[bookmark] = [feature]
            bookmark = bookmark + 1

    df3 = pd.DataFrame(df['feature'].values.tolist())
    newdf = pd.concat([df3, labels], axis=1)
    rnewdf = newdf.rename(index=str, columns={"0": "label"})

    from sklearn.utils import shuffle
    rnewdf = shuffle(newdf)
    rnewdf[:10]

    rnewdf = rnewdf.fillna(0)

    newdf1 = np.random.rand(len(rnewdf)) < 0.8
    train = rnewdf[newdf1]
    test = rnewdf[~newdf1]

    trainfeatures = train.iloc[:, :-1]
    trainlabel = train.iloc[:, -1:]

    testfeatures = test.iloc[:, :-1]
    testlabel = test.iloc[:, -1:]

    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder

    X_train = np.array(trainfeatures)
    y_train = np.array(trainlabel)
    X_test = np.array(testfeatures)
    y_test = np.array(testlabel)

    lb = LabelEncoder()

    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)

    model = Sequential()

    model.add(Conv1D(256, 5, padding='same',
                     input_shape=(216, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    # model.add(Conv1D(128, 5,padding='same',))
    # model.add(Activation('relu'))
    # model.add(Conv1D(128, 5,padding='same',))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # TODO: FIX EPOCH TO HIGHER NUMBRE
    cnnhistory = model.fit(x_traincnn, y_train, batch_size=16, epochs=140, validation_data=(x_testcnn, y_test),
                           verbose=2)

    model_name = 'Emotion_Voice_Detection_Model.h5'
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    import json
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)


def test():
    mylist2 = []
    mylist = []
    for i in os.listdir("soundAnalysis/voiceSamples"):
        [mylist.append(f) for f in os.listdir(os.path.join("soundAnalysis/voiceSamples", i))]
        [mylist2.append(os.path.join("soundAnalysis/voiceSamples", i, f)) for f in
         os.listdir(os.path.join("soundAnalysis/voiceSamples", i))]

    feeling_list = []
    for item in mylist:
        if item[6:-16] == '02' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_calm')
        elif item[6:-16] == '02' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_calm')
        elif item[6:-16] == '03' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_happy')
        elif item[6:-16] == '03' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_happy')
        elif item[6:-16] == '04' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_sad')
        elif item[6:-16] == '04' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_sad')
        elif item[6:-16] == '05' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_angry')
        elif item[6:-16] == '05' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_angry')
        elif item[6:-16] == '06' and int(item[18:-4]) % 2 == 0:
            feeling_list.append('female_fearful')
        elif item[6:-16] == '06' and int(item[18:-4]) % 2 == 1:
            feeling_list.append('male_fearful')
        elif item[:1] == 'a':
            feeling_list.append('male_angry')
        elif item[:1] == 'f':
            feeling_list.append('male_fearful')
        elif item[:1] == 'h':
            feeling_list.append('male_happy')
        # elif item[:1]=='n':
        # feeling_list.append('neutral')
        elif item[:2] == 'sa':
            feeling_list.append('male_sad')

    labels = pd.DataFrame(feeling_list)

    df = pd.DataFrame(columns=['feature'])

    bookmark = 0
    for index, y in enumerate(mylist):
        if mylist[index][6:-16] != '01' and mylist[index][6:-16] != '07' and mylist[index][6:-16] != '08' and mylist[
                                                                                                                  index][
                                                                                                              :2] != 'su' and \
                mylist[index][:1] != 'n' and mylist[index][:1] != 'd':
            file_ = mylist2[index]
            print(bookmark)

            X, sample_rate = librosa.load(file_, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X,
                                                 sr=sample_rate,
                                                 n_mfcc=13),
                            axis=0)
            feature = mfccs
            # [float(i) for i in feature]
            # feature1=feature[:135]
            df.loc[bookmark] = [feature]
            bookmark = bookmark + 1

    df3 = pd.DataFrame(df['feature'].values.tolist())
    newdf = pd.concat([df3, labels], axis=1)
    rnewdf = newdf.rename(index=str, columns={"0": "label"})

    from sklearn.utils import shuffle
    rnewdf = shuffle(newdf)
    rnewdf[:10]

    # Empty array of zeros
    rnewdf = rnewdf.fillna(0)

    # Split array by about 80/20
    newdf1 = np.random.rand(len(rnewdf)) < 0.8
    train = rnewdf[newdf1]
    test = rnewdf[~newdf1]

    trainfeatures = train.iloc[:, :-1]
    trainlabel = train.iloc[:, -1:]

    testfeatures = test.iloc[:, :-1]
    testlabel = test.iloc[:, -1:]

    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder

    X_train = np.array(trainfeatures)
    y_train = np.array(trainlabel)
    X_test = np.array(testfeatures)
    y_test = np.array(testlabel)

    lb = LabelEncoder()

    temp = y_test

    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    print("y_test", temp, y_test)

    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)

    model = Sequential()

    model.add(Conv1D(256, 5, padding='same',
                     input_shape=(216, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    # model.add(Conv1D(128, 5,padding='same',))
    # model.add(Activation('relu'))
    # model.add(Conv1D(128, 5,padding='same',))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    model_name = 'Emotion_Voice_Detection_Model.h5'
    save_dir = os.path.join(os.getcwd(), 'saved_models')

    loaded_model.load_weights(os.path.join(save_dir, model_name))
    print('Loaded model from disk...')

    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)

    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

    preds = loaded_model.predict(x_testcnn, batch_size=32, verbose=1)
    preds1 = preds.argmax(axis=1)
    abc = preds1.astype(int).flatten()
    predictions = (lb.inverse_transform((abc)))
    preddf = pd.DataFrame({'predictedvalue': predictions})

    actual = y_test.argmax(axis=1)
    abc123 = actual.astype(int).flatten()
    actualvalues = (lb.inverse_transform((abc123)))
    actualdf = pd.DataFrame({'actualvalues': actualvalues})

    finaldf = actualdf.join(preddf)

    print(finaldf[0:30], abc123[0:30])


class CNN:
    def __init__(self, model_h5, model_json):
        opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
        json_file = open(model_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        model_name = model_h5
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        loaded_model.load_weights(os.path.join(save_dir, model_name))
        loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.loaded_model = loaded_model
        self.loaded_model.predict(np.zeros((1, 216, 1)))

    def analyzer(self, fileName, num):

        # mylist2 = []
        # mylist = []
        # for i in os.listdir("soundAnalysis/voiceSamples"):
        #     [mylist.append(f) for f in os.listdir(os.path.join("soundAnalysis/voiceSamples", i))]
        #     [mylist2.append(os.path.join("soundAnalysis/voiceSamples", i,f)) for f in os.listdir(os.path.join("soundAnalysis/voiceSamples", i))]
        #
        #
        # feeling_list =[]
        # for item in mylist:
        #     if item[6:-16]=='02' and int(item[18:-4])%2==0:
        #         feeling_list.append('female_calm')
        #     elif item[6:-16]=='02' and int(item[18:-4])%2==1:
        #         feeling_list.append('male_calm')
        #     elif item[6:-16]=='03' and int(item[18:-4])%2==0:
        #         feeling_list.append('female_happy')
        #     elif item[6:-16]=='03' and int(item[18:-4])%2==1:
        #         feeling_list.append('male_happy')
        #     elif item[6:-16]=='04' and int(item[18:-4])%2==0:
        #         feeling_list.append('female_sad')
        #     elif item[6:-16]=='04' and int(item[18:-4])%2==1:
        #         feeling_list.append('male_sad')
        #     elif item[6:-16]=='05' and int(item[18:-4])%2==0:
        #         feeling_list.append('female_angry')
        #     elif item[6:-16]=='05' and int(item[18:-4])%2==1:
        #         feeling_list.append('male_angry')
        #     elif item[6:-16]=='06' and int(item[18:-4])%2==0:
        #         feeling_list.append('female_fearful')
        #     elif item[6:-16]=='06' and int(item[18:-4])%2==1:
        #         feeling_list.append('male_fearful')
        #     elif item[:1]=='a':
        #         feeling_list.append('male_angry')
        #     elif item[:1]=='f':
        #         feeling_list.append('male_fearful')
        #     elif item[:1]=='h':
        #         feeling_list.append('male_happy')
        #     #elif item[:1]=='n':
        #         #feeling_list.append('neutral')
        #     elif item[:2]=='sa':
        #         feeling_list.append('male_sad')
        # #print(feeling_list)
        #
        # labels = pd.DataFrame(feeling_list)

        X, sample_rate = librosa.load('soundAnalysis/' + fileName, res_type='kaiser_fast', duration=2.5, sr=22050 * 2,
                                      offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        featurelive = mfccs
        livedf2 = featurelive

        livedf2 = pd.DataFrame(data=livedf2)

        livedf2 = livedf2.stack().to_frame().T

        twodim = np.expand_dims(livedf2, axis=2)

        livepreds = self.loaded_model.predict(twodim,
                                              batch_size=32,
                                              verbose=1)

        livepreds1 = livepreds.argmax(axis=1)

        liveabc = livepreds1.astype(int).flatten()

        with open("soundAnalysis/voiceAnalysis.csv", "w") as csvfile:
            state = ["female_angry", "female_calm", "female_fearful", "female_happy", "female_sad", "male_angry",
                     "male_calm", "male_fearful", "male_happy", "male_sad"]
            writer = csv.DictWriter(csvfile, fieldnames=state)

            dict = {}

            for i in range(10):
                dict[state[i]] = livepreds[0][i]

            writer.writeheader()
            writer.writerow(dict)

        print(livepreds)

    def worker(self, num):

        CHUNK = 1024
        FORMAT = pyaudio.paInt16  # paInt8
        CHANNELS = 2
        RATE = 44100  # sample rate
        RECORD_SECONDS = 4
        WAVE_OUTPUT_FILENAME = "output{}.wav".format(num)
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)  # buffer

        for c in range(1, 100):

            print("{}: recording".format(num))

            frames = []

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)  # 2 bytes(16 bits) per channel

            print("{}: done recording".format(num))

            wf = wave.open("soundAnalysis/" + WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            self.analyzer(WAVE_OUTPUT_FILENAME, num)

        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    cnn = CNN("Emotion_Voice_Detection_Model.h5", "model.json")
    threads = []
    for i in range(4):
        t = threading.Thread(target=cnn.worker, args=(i,))
        threads.append(t)
        t.start()
        time.sleep(1)