import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
import os
import pickle
from utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.chdir('..')


with open('dumps/word2numid.pkl', 'rb') as f:
    word2numid = pickle.load(f)
print('word2numid loaded!')

with open('dumps/numid2word.pkl', 'rb') as f:
    numid2word = pickle.load(f)
print('numid2word loaded!')

with open('dumps/numid2vec.pkl', 'rb') as f:
    numid2vec = pickle.load(f)
print('numid2vec loaded!')

with open('dumps/X_CV.pkl', 'rb') as f:
    X_CV = pickle.load(f)
print('X_CV loaded!')

with open('dumps/Y_CV.pkl', 'rb') as f:
    Y_CV = pickle.load(f)
print('Y_CV loaded!')

with open('dumps/sl_CV.pkl', 'rb') as f:
    sl_CV = pickle.load(f)
print('sl_CV loaded!')



x_dummy = np.array(X_CV[0:batch_size]).astype('double')
sos = np.array([numid2vec[word2numid['<go>']]]*batch_size)


encoder = EncoderRNN(num_units=num_hidden_units)
decoder = DecoderRNN(word2idx=word2numid, idx2word=numid2word, idx2emb=numid2vec, num_units=num_hidden_units, max_tokens=max_seq_length)

opt, state = encoder.load(x_dummy, sl_CV)
decoder.load(x_dummy, sos, state, opt)

x = np.array(X_CV[0 : batch_size]).astype('double')
sl = sl_CV[0 : batch_size]



output, cell_state = encoder.forward(x, sl)
wp, wl = decoder.forward(x, sos, (cell_state, output), training=True)

wp = wp.numpy()

text_arr = []

for i in range(wp.shape[0]):
    temp = []
    for j in range(wp.shape[1]):
        word = numid2word[wp[i][j]]
        temp.append(word) 
    text_arr.append(temp)
