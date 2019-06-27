# ASJ Autumn 2019: dimensional speech emotion recognition from acoustic feature and word embedding

import numpy as np
import matplotlib.pyplot as plt
import pickle

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, CuDNNLSTM, TimeDistributed, Bidirectional, Embedding, Dropout, Flatten
from keras.utils import to_categorical
from sklearn.preprocessing import label_binarize

from features import *
from helper import *
from attention_helper import *

np.random.seed(99)

# loat output/label data
path = '/media/bagus/data01/dataset/IEMOCAP_full_release/data_collected_full.pickle'

with open(path, 'rb') as handle:
    data = pickle.load(handle)
len(data)

v = [v['v'] for v in data]
a = [a['a'] for a in data]
d = [d['d'] for d in data]

vad = np.array([v, a, d])
vad = vad.T
print(vad.shape)

# load input (speech feature) data
voiced_feat = np.load('voiced_feat_full.npy')
print(voiced_feat.shape)

# categotical model
def attention_model(optimizer='rmsprop'):
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True), input_shape=(100, 31)))
    model.add(Bidirectional(AttentionDecoder(64,64)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='linear'))

    model.compile(loss='mse', optimizer=optimizer, metrics=['mape'])

    return model

model1 = attention_model()
model1.summary()

# train data
hist = model1.fit(voiced_feat, vad, batch_size=32, epochs=30, verbose=1, 
                  shuffle=True, validation_split=0.2)

acc1 = hist.history['val_mean_absolute_percentage_error']
print('max: {:.4f}, min:{:.4f}, avg:{:.4f}'.format(max(acc1), min(acc1), np.mean(acc1)))

# it works until line above, the output is:
# max: 26.3977, min:21.8468, avg:23.2664

# The following lines below is not validated yet
# model 2
def api_model():
    inputs = Input(shape=(100, 31))
    net = Masking()(inputs)
    net = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inputs)
    net = Bidirectional(CuDNNLSTM(128, return_sequences=True))(net)
    #net = Dropout(0.2)(net)
    
    outputs = []
    out1 = TimeDistributed(Dense(1, activation='linear'))(net)  
    #outputs.append(out1)
    out2 = Dense(1, activation='linear')(net)  
    #outputs.append(out2)
    out3 = Dense(1, activation='linear')(net)  
    #outputs.append(out3)
    
    model   = Model(inputs=inputs, outputs=[out1, out2, out3])
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mape'])
    return model

model2 = api_model()
model2.summary()
hist = model2.fit(voiced_feat, [v, a, d], batch_size=32, epochs=30, verbose=1, shuffle=True, validation_split=0.2)
