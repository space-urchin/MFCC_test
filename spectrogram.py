import os
from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import model_from_json

imwidth                             = 50
imheight                            = 34
audio_dir = "./recordingsbangla/"
file_names = [f for f in os.listdir(audio_dir) if '.wav' in f]

total_samples = len(file_names)

_x = []
_y = []

for i in range(len(file_names)):   
    rate, data = wavfile.read(audio_dir + file_names[i])
    os.system('cls')
    print(f'{i}/{total_samples}')
    print(file_names[i])
    if(len(data.shape)==2):
        data = data[:,0]

    # frequencies, times, spectrogram = signal.spectrogram(data, rate)
    mel_data = mfcc(data, samplerate=rate, winlen=1, numcep=13, nfilt=26, nfft=48000,preemph=0.97)
    imarray = np.resize(mel_data,(34,50))
    _x.append(imarray)
    _y.append(int(file_names[i][1:4])-1)

x = np.array(_x)
y = np.array(_y)
print(x.shape)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y)

print("Size of Training Data:", np.shape(x_train))
print("Size of Training Labels:", np.shape(y_train))
print("Size of Test Data:", np.shape(x_test))
print("Size of Test Labels:", np.shape(y_test))

num_classes = 11

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.reshape(x_train.shape[0], imheight, imwidth, 1)
x_test = x_test.reshape(x_test.shape[0], imheight, imwidth, 1)
input_shape = (imheight, imwidth, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, batch_size=10, epochs=10, verbose=1, validation_data=(x_test, y_test))


