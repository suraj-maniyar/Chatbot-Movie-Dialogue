from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import tensorflow.contrib.legacy_seq2seq as seq2seq
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.chdir('..')

#####################################################################################################################################

# Threshold maximum and minimum number of words to be used in a dialogue.
# Dialogs having number of words outside this threshold will be discarded.
max_seq_length = 30
min_seq_length = 1

# Dimension of word vector
dimension = 50

# Total number of conversations which we consider for training.
total_convs = 100   # len(convs)

# Learning parameters
num_epochs = 10
batch_size = 32
learning_rate = 1e-3

nodes = 32
embed_size = 50

#####################################################################################################################################

lines_file = 'data/movie_lines.txt'
convs_file = 'data/movie_conversations.txt'

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



#print('max_seq_length : ', max_seq_length)

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


glove_model = loadGloveModel('/home/suraj/Dataset/glove.6B/glove.6B.50d.txt')
glove_model['<eos>'] = glove_model['.']
glove_model['<pad>'] = glove_model['-----']  #np.zeros(dimension)
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

# Creating an embedding for words
embedding = np.zeros((len(word2vec), dimension))
# Dictionary mapping each word to a unique number
word2numid = {}
for id, word in enumerate(list(word2vec.keys())):
    word2numid[word] = id
    embedding[id] = word2vec[word]

numid2word = {v: k for k, v in word2numid.items()}



X = []
Y = []



print('Generating X and Y arrays')

conv_data = []
for conv_index in range(total_convs):
    # The intersecrion returns set of elements which are present in both the lists
    print('\nConversation',conv_index, '/', total_convs, ' Sentences:',len(convs[conv_index]))
    if(  set(convs[conv_index]) & set(lines_to_ignore)  ==  set() ):
        for i in range(len(convs[conv_index])):
            sentence = getSentence(convs[conv_index][i], id2line, min_seq_length, max_seq_length)
            id_sen = []
            for element in sentence:
                if(element in list(word2numid.keys())):
                    id_sen.append( word2numid[element] )
                else:
                    id_sen.append( word2numid['<unk>'] )
            conv_data.append(id_sen)


for i in range(len(conv_data)-1):
    X.append(conv_data[i])
    Y.append(conv_data[i+1])

print('Converting to numpy array...')
X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)

print('Shuffling...')
X_train, X_CV, Y_train, Y_CV = train_test_split(X, Y, test_size=0.2, random_state=7)

print('\nTrain')
print(X_train.shape)
print(Y_train.shape)

print('\nCV')
print(X_CV.shape)
print(Y_CV.shape)

print('\n')
print('Epochs:', num_epochs)
print('Learning Rate:', learning_rate)
print('Batch Size:', batch_size)
print('\n')



total_batches_train = int(X_train.shape[0] / batch_size)
total_batches_CV = int(X_CV.shape[0] / batch_size)


# http://androidkt.com/pre-trained-word-embedding-tensorflow-using-estimator-api/
embeddingMatrix = np.zeros( (len(numid2word), dimension) )
for i in range(len(numid2word)):
    embeddingMatrix[i] = word2vec[numid2word[i]]
embeddingMatrix = embeddingMatrix.astype(np.float32)



tf.reset_default_graph()
