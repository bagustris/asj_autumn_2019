# cocosda 2019: dimensional speech emotion recognition from acoustic feature and word embedding
# to be run on jupyter-lab, block certain lines, Shift-Enter to execute

import numpy as np
import matplotlib.pyplot as plt
import pickle

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, CuDNNLSTM, TimeDistributed, Bidirectional, Embedding, Dropout, Flatten, concatenate, CuDNNGRU
from keras.utils import to_categorical
from sklearn.preprocessing import label_binarize

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

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
split = 8500

# model1
def speech_model(optimizer='rmsprop'):
    # speech model
    input_speech = Input(shape=(100, 31))
    model_speech = Flatten()(input_speech)
    model_speech = Dense(32, activation='relu')(model_speech)
    model_speech = Dense(32, activation='relu')(model_speech)
    model_speech = Dropout(0.3)(model_speech)
    #model_speech = Flatten()(model_speech)
    model_speech = Dense(3, activation='linear')(model_speech)
    
    model = Model(inputs=input_speech, outputs=model_speech)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mape', 'mae'])

    return model

model = speech_model()
hist = model.fit(voiced_feat[:split], vad[:split], epochs=30, batch_size=32, verbose=1, validation_split=0.2)
acc = hist.history['val_mean_absolute_percentage_error']
print('max: {:.4f}, min:{:.4f}, avg:{:.4f}'.format(max(acc), min(acc), np.mean(acc)))

# evaluation 
eval_metrik = model.evaluate(voiced_feat[split:], vad[split:])
print(eval_metrik)
# [0.6861955348046435, 23.784291114336177, 0.6740260705923089]

# model 2: GRU+GRU
def speech_model2(optimizer='rmsprop'):
    # speech model
    input_speech = Input(shape=(100, 31))
    model_speech = CuDNNGRU(32, return_sequences=True)(input_speech)
    model_speech = CuDNNGRU(32, return_sequences=False)(model_speech)
    model_speech = Dropout(0.3)(model_speech)
    model_speech = Dense(3, activation='linear')(model_speech)
    
    model = Model(inputs=input_speech, outputs=model_speech)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mape', 'mae'])

    return model


model2 = speech_model2()
hist2 = model2.fit(voiced_feat[:split], vad[:split], epochs=30, batch_size=32, verbose=1, validation_split=0.2)
acc2 = hist2.history['val_mean_absolute_percentage_error']
print('max: {:.4f}, min:{:.4f}, avg:{:.4f}'.format(max(acc2), min(acc2), np.mean(acc2)))

# evaluation 
eval_metrik2 = model2.evaluate(voiced_feat[split:], vad[split:])
print(eval_metrik2)
# [0.6467004129064013, 23.465309678462514, 0.6526033936420612]
# LSTM: [0.659281604858557, 23.46477922981168, 0.6569926305743609]

# model3 : BGRU + BGRU
def speech_model3(optimizer='rmsprop'):
    # speech model
    input_speech = Input(shape=(100, 31))
    model_speech = Bidirectional(CuDNNGRU(64, return_sequences=True))(input_speech)
    model_speech = Bidirectional(CuDNNGRU(64))(model_speech)
    model_speech = Dropout(0.3)(model_speech)
    model_speech = Dense(3, activation='linear')(model_speech)
    
    model = Model(inputs=input_speech, outputs=model_speech)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mape', 'mae'])
    return model
    
model3 = speech_model3()
hist3 = model3.fit(voiced_feat[:split], vad[:split], epochs=30, batch_size=32, verbose=1, validation_split=0.2)
acc3 = hist3.history['val_mean_absolute_percentage_error']
print('max: {:.4f}, min:{:.4f}, avg:{:.4f}'.format(max(acc3), min(acc3), np.mean(acc3)))

# evaluation 
eval_metrik3 = model3.evaluate(voiced_feat[split:], vad[split:])
print(eval_metrik3)
# [0.6623892277537266, 23.429723827617366, 0.6623892277537266]
# LSTM: 

# model 4 : GRU with Attention model
def speech_model4(optimizer='rmsprop'):
    # speech model
    input_speech = Input(shape=(100, 31))
    model_speech = Bidirectional(CuDNNGRU(64, return_sequences=True))(input_speech)
    model_speech = Bidirectional(AttentionDecoder(64,64))(model_speech)
    model_speech = Flatten()(model_speech)
    model_speech = Dropout(0.3)(model_speech)
    model_speech = Dense(3, activation='linear')(model_speech)
    
    model = Model(inputs=input_speech, outputs=model_speech)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mape', 'mae'])
    return model

model4 = speech_model4()
hist4 = model4.fit(voiced_feat[:split], vad[:split], epochs=30, batch_size=32, verbose=1, validation_split=0.2)
acc4 = hist4.history['val_mean_absolute_percentage_error']
print('max: {:.4f}, min:{:.4f}, avg:{:.4f}'.format(max(acc4), min(acc4), np.mean(acc4)))

# evaluation 
eval_metrik4 = model4.evaluate(voiced_feat[split:], vad[split:])
print(eval_metrik4)

