import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
import os
import pickle
from utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#####################################################################################################################################

# Threshold maximum and minimum number of words to be used in a dialogue.
# Dialogs having number of words outside this threshold will be discarded.
max_seq_length = 80
min_seq_length = 1

# Dimension of word vector
dimension = 50

# Total number of conversations which we consider for training.
total_convs = 800   # len(convs)

# Learning parameters
num_epochs = 20
batch_size = 32
learning_rate = 1e-3
num_hidden_units = 256



with open('word2numid.pkl', 'rb') as f:
    word2numid = pickle.load(f)
print('word2numid loaded!')

with open('numid2word.pkl', 'rb') as f:
    numid2word = pickle.load(f)
print('numid2word loaded!')

with open('numid2vec.pkl', 'rb') as f:
    numid2vec = pickle.load(f)
print('numid2vec loaded!')

with open('X_CV.pkl', 'rb') as f:
    X_CV = pickle.load(f)
print('X_CV loaded!')

with open('Y_CV.pkl', 'rb') as f:
    Y_CV = pickle.load(f)
print('Y_CV loaded!')

with open('sl_CV.pkl', 'rb') as f:
    sl_CV = pickle.load(f)
print('sl_CV loaded!')

x_dummy = np.array(X_CV[0:batch_size]).astype('double')
sos = np.array([numid2vec[word2numid['<go>']]]*batch_size)
state_dummy = np.zeros((batch_size, num_hidden_units))
state_dummy = tf.convert_to_tensor(state_dummy, dtype=tf.float32)

encoder = EncoderRNN(num_units=num_hidden_units)
decoder = DecoderRNN(word2idx=word2numid, idx2word=numid2word, idx2emb=numid2vec, num_units=num_hidden_units, max_tokens=max_seq_length)

opt, state = encoder.load(x_dummy, sl_CV)
decoder.load(x_dummy, sos, state)
