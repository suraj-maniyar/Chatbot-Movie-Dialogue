import numpy as np
from sklearn.model_selection import train_test_split
import os


os.chdir('..')

############################################################################################################

# Threshold maximum number of words to be used in a dialogue. Dialogs having more number of words than this threshold are simply discarded.
max_seq_length = 100

# Dimension of word vector
dimension = 50

# Total number of conversations which we consider for training.
total_convs = 200   # len(convs)


############################################################################################################

# Returns dictionary which gives vectors corresponding to words
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


# Define your text pre-processing here
def preprocess(text):
    text = text.lower()
    text = text.replace("you're", "you are")
    text = text.replace("he's", "he is")
    text = text.replace("she's", "she is")
    text = text.replace("it's", "it is")
    text = text.replace("what's", "what is")
    text = text.replace("that's", "that is")
    text = text.replace("there's", "there is")
    text = text.replace("how's", "how is")
    text = text.replace("'ll", " will")
    text = text.replace("'ve", " have")
    text = text.replace("won't", "will not")
    text = text.replace("'d", " would")
    text = text.replace("'can't", "can not")
    text = text.replace("i'm", "i am")
    text = text.replace('.', ' <eos> ')
    text = text.replace(",", "")
    text = text.replace(";", "")
    text = text.replace("'", "")
    text = text.replace('"', '')
    text = text.replace(':', '')
    text = text.replace('*', '')
    text = text.replace('_', '')
    text = text.replace('/', '')
    text = text.replace('~', '')
    text = text.replace('$', ' dollar ')
    text = text.replace('?', ' ?')
    text = text.replace('!', '')
    text = text.replace('-', ' ')
    text = text.replace('<u>', ' ')
    text = text.replace('<i>', ' ')
    text = text.replace('</i>', ' ')
    text = text.replace('<b>', ' ')
    text = text.replace('</b>', ' ')
    text = text.replace('%', ' percent')
    text = text.replace('</u>', ' ')
    text = text.replace('[', ' ')
    text = text.replace(']', ' ')
    text = text.replace('/', ' ')
    return text


# Returns padded vector corresponding to given line number
def getVector(lineNo, glove_model, max_seq_length, vocab):
    sentence = id2line[lineNo].split()
    pad_len = max_seq_length - len(sentence)
    if(pad_len < 0):
        print("*********************************************")
    pad = ['<pad>'] * pad_len
    sentence = pad + sentence
    vect = []
    for i in range(len(sentence)):
        if(sentence[i] in vocab):
            vect.append( glove_model[sentence[i]] )
            #print(sentence[i])
        else:
            #print('UNK')
            vect.append( glove_model['<unk>'] )
    vect = np.array(vect)
    return vect


# Returns nearest word to the given vector
def getWord(vec, glove_model):
    key_dist = {}
    for key in glove_model.keys():
        key_dist[key] = np.sum(np.square( vec - glove_model[key] ))
    min_key = min(key_dist, key=key_dist.get)
    return min_key


############################################################################################################


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
    if( len( id2line[line_ids].split() ) > max_seq_length ):
        lines_to_ignore.append(line_ids)


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
print('Total number of lines to ignore : ', len(lines_to_ignore))
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
print('Vocab Size = ', len(vocab))


glove_model = loadGloveModel('/home/suraj/Dataset/glove.6B/glove.6B.50d.txt')
glove_model['<eos>'] = glove_model['.']
glove_model['<pad>'] = np.zeros(dimension)
glove_list = list(glove_model.keys())

# Words which are in vocab and have a vector representation.
intersection =  set(vocab) & set(glove_list)

# List containing words which do not have any vector representation.
no_mapping = list(set(vocab) - intersection)
print('Number of words from vocab with no mapping : ', len(no_mapping))

# Removing those words from our original vocabulary, which do not have a vector representation
vocab = list( set(vocab) - set(no_mapping) )
print('New Vocab Size : ', len(vocab))

# Converting list of vocabulary to a dictionary of vocabulary. word2vec is same as glove_model, but with only those
# keys which are present in our corpus. This helps reduce size of our dictionary and making vector2word computation more efficient.
word2vec = {}
for key in vocab:
    word2vec[key] = glove_model[key]
print('word2vec size : ', len(word2vec.keys()))


X = []
Y = []


print('Generating X and Y arrays')
for conv_index in range(total_convs):
    # The intersecrion returns set of elements which are present in both the lists
    print('\nConversation',conv_index, '/', total_convs, ' Sentences:',len(convs[conv_index]))
    if(  set(convs[conv_index]) & set(lines_to_ignore)  ==  set() ):
        for  i in range(len(convs[conv_index])-1):
            vectorX = getVector(convs[conv_index][i], word2vec, max_seq_length, vocab)
            vectorY = getVector(convs[conv_index][i+1], word2vec, max_seq_length, vocab)
            X.append(vectorX)
            Y.append(vectorY)

print('Converting to numpy array...')
X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)

print('Shuffling...')
X_train, X_CV, Y_train, Y_CV = train_test_split(X, Y, test_size=0.2, random_state=7)

print('Train')
print(X_train.shape)
print(Y_train.shape)

print('\nCV')
print(X_CV.shape)
print(Y_CV.shape)
