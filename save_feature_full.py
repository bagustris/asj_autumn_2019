# run this to unuse cuda
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# script to save feature
import numpy as np
import pickle 
#import audiosegment 
from helper import *
from features import *
import pandas as pd
from scipy.io import wavfile

def calculate_features(frames, freq, options):
    window_sec = 0.02
    window_n = int(freq * window_sec)

    st_f = stFeatureExtraction(frames, freq, window_n, window_n / 2)

    if st_f.shape[1] > 2:
        i0 = 1
        i1 = st_f.shape[1] - 1
        if i1 - i0 < 1:
            i1 = i0 + 1
        
        deriv_st_f = np.zeros((st_f.shape[0], i1 - i0), dtype=float)
        for i in range(i0, i1):
            i_left = i - 1
            i_right = i + 1
            deriv_st_f[:st_f.shape[0], i - i0] = st_f[:, i]
        return deriv_st_f
    elif st_f.shape[1] == 2:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f
    else:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f

# load speech data
list_data = pd.read_csv('audio_data.csv', header=None)
path = '/media/bagus/data01/dataset/IEMOCAP_full_release/'
voiced_feat = []

for i in range(len(list_data)):
    framerate, x_head = wavfile.read(path+str(list_data.iloc[i,0]))
    st_features = calculate_features(x_head, framerate, None)
    st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
    voiced_feat.append(st_features.T)
    if i%100==0:
        print(i)

voiced_feat = np.array(voiced_feat)
print(voiced_feat.shape)
np.save('voiced_feat_full.npy', voiced_feat)
