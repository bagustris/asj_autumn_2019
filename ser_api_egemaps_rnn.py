# ASJ Autumn 2019: RNN-based dimensional speech emotion recognition

import numpy as np
import matplotlib.pyplot as plt
import pickle

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, CuDNNLSTM, TimeDistributed, Bidirectional, Flatten, \
                         Embedding, Dropout, Flatten, CuDNNGRU, GRU, LSTM, BatchNormalization, \
                         RNN, SimpleRNN, Activation
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error 

#from attention_helper import AttentionDecoder
from read_csv import load_features

np.random.seed(99)

features = np.load('X_egemaps.npy')
vad = np.load('y_egemaps.npy')

print('Feature shape: ', features.shape)
print('Label shape: ', vad.shape)

# standardization
scaled = 'standard'

# standardization
if scaled == 'standard':
    scaler = StandardScaler()
elif scaled == 'minmax':
    scaler = MinMaxScaler()
elif scaled == 'robust':
    scaler = RobustScaler()
else:
    print("Unrecognized scaler method")
scaler = scaler.fit(features.reshape(features.shape[0]*features.shape[1], features.shape[2]))
scaled_feat = scaler.transform(features.reshape(features.shape[0]*features.shape[1], features.shape[2]))
scaled_feat = scaled_feat.reshape(features.shape[0], features.shape[1], features.shape[2])

# Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
def ccc(gold, pred):
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1,  keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.common.epsilon())
    return ccc

def ccc_loss(gold, pred):  
    # input (num_batches, seq_len, 1)
    ccc_loss   = K.constant(1.) - ccc(gold, pred)
    return ccc_loss

# API model, if use RNN, first two rnn layer must return_sequences=True
def api_model():
    inputs = Input(shape=(features.shape[1], features.shape[2]), name='feat_input')
    
    net = BatchNormalization()(inputs)
    net = CuDNNLSTM(64, return_sequences=True)(net)
    net = CuDNNLSTM(64, return_sequences=True)(net)
    net = CuDNNLSTM(64, return_sequences=True)(net)
    net = Flatten()(net)
    net = Dropout(0.4)(net)
    target_names = ('v', 'a', 'd')
    outputs = [Dense(1, name=name)(net) for name in target_names]

    model = Model(inputs=inputs, outputs=outputs) #=[out1, out2, out3])
    model.compile(loss=ccc_loss, #{'v': ccc_loss, 'a': ccc_loss, 'd': ccc_loss}, 
                  loss_weights={'v': 0.7, 'a': 0.3, 'd':0.6}, 
                  optimizer='rmsprop', metrics=['mse', 'mae', 'mape', ccc])
    return model

model2 = api_model()
model2.summary()

earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
hist = model2.fit(features[:8000], vad[:8000].T.tolist(), batch_size=64, 
                  validation_split=0.2, epochs=100, verbose=1, shuffle=True, callbacks=[earlystop])

#, a[:8000], d[:8000]
#loss, mse_v, mse_a, mse_d, ccc_v, ccc_a, ccc_d 
metrik2 = model2.evaluate(features[8000:], vad[8000:].T.tolist())#, a[8000:], d[8000:]])
print(metrik2)

# make prediction
predict2 = model2.predict(features[8000:])
predict2 = np.array(predict2).reshape(3, 2039).T

# manual mse computation
#mse_v = mean_squared_error(vad[8000:, 0], predict2[:,0])
#mse_a = mean_squared_error(vad[8000:, 1], predict2[:,1])
#mse_d = mean_squared_error(vad[8000:, 2], predict2[:,2])

#print(mse_v, mse_a, mse_d)
