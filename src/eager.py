
from sklearn.model_selection import train_test_split
import os
from utils import *
import pickle
import random

tfe.enable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.chdir('..')

#####################################################################################################################################

# Threshold maximum and minimum number of words to be used in a dialogue.
# Dialogs having number of words outside this threshold will be discarded.
max_seq_length = 60
min_seq_length = 1

# Dimension of word vector
dimension = 50

# Total number of conversations which we consider for training.
total_convs = 1000   # len(convs)

# Learning parameters
num_epochs = 10
batch_size = 64
learning_rate = 1e-3
num_hidden_units = 128

#####################################################################################################################################

lines_file = 'data/movie_lines.txt'
convs_file = 'data/movie_conversations.txt'
glove_file = 'glove.6B.50d.txt'

lines = open(lines_file, encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open(convs_file, encoding='utf-8', errors='ignore').read().split('\n')


# Creating a dictionary to map line Ids to Lines
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if(len(_line) == 5):

        text = _line[4]
        # Text Pre-processing
        text = preprocess(text)
        id2line[ _line[0] ] = text

# Creating a list of all conversations as line ids
convs = []
for line in conv_lines:
  _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
  convs.append(_line.split(','))
print('len(convs):', len(convs))


'''
# List storing maximum no. of words used in all dialogs in all conversations. Use this to set 'Maximum Sequence Length'
len_arr = []
for conv_index in range(len(convs)):
    # Check if the conversation is valid. i.e. no. of dialogues in conversation is greater than 1
    if(len(convs[conv_index]) > 1):
       for i in range(len(convs[conv_index])):
         len_arr.append( len(id2line[convs[conv_index][i]].split()) )
len_arr.sort(reverse=True)
'''



# Contains line numbers for which sentences are longer than threshold length
lines_to_ignore = []
for line_ids in list(id2line.keys()):
    if( len( id2line[line_ids].split() ) > max_seq_length or len( id2line[line_ids].split() ) < min_seq_length ):
        lines_to_ignore.append(line_ids)

print('Total number of lines to ignore : ', len(lines_to_ignore))

print('Generating vocab....')
# List to store all valid dialogues. (Text Corpus)
text_arr = []
# If any conversation has a dialogue which appears in lines_to_ignore, we discard that conversation all together
for conv_index in range(len(convs)):
    # lines from conversation which do not contain a single ignore word.
    if(  set(convs[conv_index]) & set(lines_to_ignore)  ==  set() ):
        for i in range(len(convs[conv_index])-1):
            text_arr.append( id2line[ convs[conv_index][i]   ] )



print('max_seq_length : ', max_seq_length)

print('Total number of sentences : ', len(text_arr))

# Union of words in text corpus
words_list2 = [element.split() for element in text_arr]
word_list = []
for sublist in words_list2:
    for item in sublist:
        word_list.append(item)


# Vocab list pertaining to our data (Cornell Movie Data)
vocab = list(set(word_list))
vocab.append('<unk>')
vocab.append('<pad>')
vocab.append('<go>')
vocab.append('<eos>')

print('Vocab Size = ', len(vocab))


glove_model = loadGloveModel(glove_file)
glove_model['<eos>'] = glove_model['.']
glove_model['<pad>'] = glove_model['-----']
glove_model['<go>'] = glove_model['-------']

glove_list = list(glove_model.keys())

# Words which are in vocab and have a vector representation.
intersection =  set(vocab) & set(glove_list)

# List containing words which do not have any vector representation.
no_mapping = list(set(vocab) - intersection)
print('Number of words from vocab with no mapping : ', len(no_mapping))

# Removing those words from our original vocabulary, which do not have a vector representation
vocab = list( set(vocab) - set(no_mapping) )

# Converting list of vocabulary to a dictionary of vocabulary. word2vec is same as glove_model, but with only those
# keys which are present in our corpus. This helps reduce size of our dictionary and making vector2word computation more efficient.
word2vec = {}
for key in vocab:
    word2vec[key] = glove_model[key]


print('New Vocab Size : ', len(vocab))
print('word2vec size : ', len(word2vec.keys()))

vocab_size = len(word2vec)

# Dictionary mappings

word2numid = {}
numid2vec = {}

for id, word in enumerate(list(word2vec.keys())):
    word2numid[word] = id
    numid2vec[id] = word2vec[word]

numid2word = {v: k for k, v in word2numid.items()}

with open('dumps/word2numid.pkl', 'wb') as f:
    pickle.dump(word2numid, f)
print('word2numid saved!')

with open('dumps/numid2word.pkl', 'wb') as f:
    pickle.dump(numid2word, f)
print('numid2word saved!')

with open('dumps/numid2vec.pkl', 'wb') as f:
    pickle.dump(numid2vec, f)
print('numid2vec saved!')





X = []
Y = []


print('Generating X and Y arrays')


for conv_index in range(total_convs):
    # The intersecrion returns set of elements which are present in both the lists
    if(  set(convs[conv_index]) & set(lines_to_ignore)  ==  set() ):
        print('Conversation',conv_index, '/', total_convs, ' Sentences:',len(convs[conv_index]))
        for i in range(len(convs[conv_index])-1):
            vectorX = getVector(convs[conv_index][i], id2line, word2vec, vocab, min_seq_length, max_seq_length, verbose=0)
            sentenceY = getSentence(convs[conv_index][i], id2line, min_seq_length, max_seq_length)
            vectorY = np.zeros( (max_seq_length, vocab_size) )
            for row in range(max_seq_length):
                if(sentenceY[row] in vocab):
                    col = word2numid[sentenceY[row]]
                else:
                    col = word2numid['<unk>']
                vectorY[row][col] = 1
            
            X.append(vectorX)
            Y.append(vectorY)

go_arr = word2vec[ '<go>' ]
#print('go_arr.shape : ', go_arr.shape)

total_len = len(X)
cv_split = 0.3
len_train = int( (1-cv_split)*total_len )
len_CV = int( cv_split*total_len )

print('len(X) : ', len(X))
print('len_train : ', len_train)
print('len_CV : ', len_CV)



X_train = X[0 : len_train]
Y_train = Y[0 : len_train]

X_CV = X[len_train : ]
Y_CV = Y[len_train : ]


print('Shuffling....')
zip_list = list(zip(X, Y))
random.shuffle(zip_list)
X, Y = zip(*zip_list)



with open('dumps/X_CV.pkl', 'wb') as f:
    pickle.dump(X_CV, f)
print('X_CV saved!')

with open('dumps/Y_CV.pkl', 'wb') as f:
    pickle.dump(Y_CV, f)
print('Y_CV saved!')


sl_train = []
sl_CV = []


for i in range(len(X_train)):
    count = 0
    for j in range(max_seq_length):
        if(not np.array_equal(np.array(X_train[i][j]), go_arr)):
            count = count+1
        else:
            sl_train.append(max_seq_length-count)
            break

for i in range(len(X_CV)):
    count = 0
    for j in range(max_seq_length):
        if(not np.array_equal(np.array(X_CV[i][j]), go_arr)):
            count = count+1
        else:
            sl_CV.append(max_seq_length-count)
            break


with open('dumps/sl_CV.pkl', 'wb') as f:
    pickle.dump(sl_CV, f)
print('sl_CV saved!')



print('\n')
print('Epochs:', num_epochs)
print('Learning Rate:', learning_rate)
print('Batch Size:', batch_size)
print('sl_train:', len(sl_train))
print('sl_CV:', len(sl_CV))
print('\n')


total_batches_train = int(len_train / batch_size) #int(X_train.shape[0] / batch_size)
total_batches_CV = int(len_CV / batch_size) #int(X_CV.shape[0] / batch_size)



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

encoder = EncoderRNN(num_units=num_hidden_units)
decoder = DecoderRNN(word2idx=word2numid, idx2word=numid2word, idx2emb=numid2vec, num_units=num_hidden_units, max_tokens=max_seq_length)

if(os.path.isfile('encoder_model/checkpoint')):
    print('Loading Encoder.... ')
    x_dummy = np.array(X_train[0:batch_size]).astype('double')
    sos_dummy = np.array([numid2vec[word2numid['<go>']]]*batch_size)
    opt, state = encoder.load(x_dummy, sl_CV)

    if(os.path.isfile('decoder_model/checkpoint')):
        print('Loading Decoder.... ')
        decoder.load(x_dummy, sos_dummy, state, opt)
    else:
        print('No previously saved decoder model found')
else:
    print('No previously saved model found!')






sos = np.array([go_arr]*batch_size)
#print('sos.shape: ', sos.shape)


for epoch in range(num_epochs):

    print('Training')
    for batchTrain_id in range(total_batches_train):
        batchTrain_x = np.array(X_train[ batchTrain_id*batch_size : (batchTrain_id+1)*batch_size ]).astype('double')
        batchTrain_y = np.array(Y_train[ batchTrain_id*batch_size : (batchTrain_id+1)*batch_size ]).astype('double')

        optimizer.minimize(lambda: get_loss(encoder, decoder, batchTrain_x, batchTrain_y, sl_train, sos, batchTrain_id, batch_size))

    print('Saving')
    encoder.save('encoder_model/')
    decoder.save('decoder_model/')

    print('Training Loss....')
    train_loss = 0
    for batchTrain_id in range(total_batches_train):
        batchTrain_x = np.array(X_train[ batchTrain_id*batch_size : (batchTrain_id+1)*batch_size ]).astype('double')
        batchTrain_y = np.array(Y_train[ batchTrain_id*batch_size : (batchTrain_id+1)*batch_size ]).astype('double')

        train_loss += get_loss(encoder, decoder, batchTrain_x, batchTrain_y, sl_train, sos, batchTrain_id, batch_size).numpy()

    print('CV Loss.....')
    cv_loss = 0
    for batchCV_id in range(total_batches_CV):
        batchCV_x = np.array(X_CV[ batchCV_id*batch_size : (batchCV_id+1)*batch_size ]).astype('double')
        batchCV_y = np.array(Y_CV[ batchCV_id*batch_size : (batchCV_id+1)*batch_size ]).astype('double')

        cv_loss += get_loss(encoder, decoder, batchCV_x, batchCV_y, sl_CV, sos, batchCV_id, batch_size).numpy()


    print('Epoch', epoch+1, '   Train Loss:', train_loss, '  CV Loss:', cv_loss)

