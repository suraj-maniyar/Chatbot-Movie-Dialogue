import numpy as np
import os


os.chdir('..')


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
            print('UNK')
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


# List storing maximum no. of words used in a dialogue for all conversations.
len_arr = []

for conv_index in range(len(convs)):
    # Check if the conversation is valid. i.e. no. of dialogues in conversation is greater than 1
    if(len(convs[conv_index]) > 1):
       for i in range(len(convs[conv_index])):
         len_arr.append( len(id2line[convs[conv_index][i]].split()) )
len_arr.sort(reverse=True)

# Threshold maximum number of words to be used in a dialogue.
max_seq_length = 100

lines_to_ignore = []
for line_ids in list(id2line.keys()):
    if( len( id2line[line_ids].split() ) > max_seq_length ):
        lines_to_ignore.append(line_ids)


text_arr = []
# If any conversation has a dialogue which appears in dialogs_to_ignore, we discard that conversation all together
for conv_index in range(len(convs)):
    # The intersecrion returns set of elements which are present in both the lists
    if(  set(convs[conv_index]) & set(lines_to_ignore)  ==  set() ):
        for i in range(len(convs[conv_index])-1):
            text_arr.append( id2line[ convs[conv_index][i]   ] )



print('max_seq_length', max_seq_length)
print('lines_to_ignore', len(lines_to_ignore))
print('len(text_arr) : ', len(text_arr))


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
glove_model['<pad>'] = np.zeros(50)
glove_list = list(glove_model.keys())

intersection =  set(vocab) & set(glove_list)
no_mapping = list(set(vocab) - intersection)
print('No Mapping : ', len(no_mapping))


vocab = list( set(vocab) - set(no_mapping) )
print('New Vocab Size : ', len(vocab))

word2vec = {}

for key in vocab:
    word2vec[key] = glove_model[key]

print('word2vec size : ', len(word2vec.keys()))


X = []
Y = []

'''
total_convs = 60   # len(convs)

for conv_index in range(total_convs):
    # The intersecrion returns set of elements which are present in both the lists
    print(conv_index, total_convs)
    if(  set(convs[conv_index]) & set(lines_to_ignore)  ==  set() ):
        for  i in range(len(convs[conv_index])-1):
            vectorX = getVector(convs[conv_index][i], word2vec, max_seq_length, vocab)
            vectorY = getVector(convs[conv_index][i+1], word2vec, max_seq_length, vocab)
            X.append(vectorX)
            Y.append(vectorY)

X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)
'''
