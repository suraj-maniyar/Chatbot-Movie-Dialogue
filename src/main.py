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


# Threshold maximum number of words to be used in a dialogue.
max_seq_length = 100

dialogs_to_ignore = []
for line_ids in list(id2line.keys()):
    if( len( id2line[line_ids].split() ) > max_seq_length):
        dialogs_to_ignore.append(line_ids)

# Generate the training data. The input is the current dialogue and its output is the next dialogue
X = []
Y = []

# If any conversation has a dialogue which appears in dialogs_to_ignore, we discard that conversation all together
for conv_index in range(len(convs)):
    # The intersecrion returns set of elements which are present in both the lists
    if(  set(convs[conv_index]) & set(dialogs_to_ignore)  ==  set() ):
        for i in range(len(convs[conv_index])-1):
            X.append( id2line[ convs[conv_index][i]   ] )
            Y.append( id2line[ convs[conv_index][i+1] ] )

print('max_seq_length', max_seq_length)
print('dialogs_to_ignore', len(dialogs_to_ignore))
print('len(X) : ', len(X))
print('len(Y) : ', len(Y))

'''
index = 40
for i in range(index, index+5):
    print(id2line[X[i]])
    print(id2line[Y[i]])
    print()
'''

words_list2 = [element.split() for element in X]
word_list = []
for sublist in words_list2:
    for item in sublist:
        word_list.append(item)

vocab = list(set(word_list))
print('Vocab Size = ', len(vocab))


glove_model = loadGloveModel('/home/suraj/Dataset/glove.6B/glove.6B.50d.txt')
glove_model['<eos>'] = glove_model['.']
glove_list = list(glove_model.keys())

intersection =  set(vocab) & set(glove_list)
no_mapping = list(set(vocab) - intersection)
print('No Mapping : ', len(no_mapping))

vocab = list( set(vocab) - set(no_mapping) )
print('New Vocab Size : ', len(vocab))

X_train = []
Y_train = []

for i in range(len(X)):
    X_ = X[i].split()
    Y_ = Y[i].split()
    elemX = []
    elemY = []

    for j in range(len(X_)):
        if(X_[j] in vocab):
            elemX.append( glove_model[X_[j]] )
        else:
            elemX.append( glove_model['<unk>'] )

    for j in range(len(Y_)):
        if(Y_[j] in vocab):
            elemY.append( glove_model[Y_[j]] )
        else:
            elemY.append( glove_model['<unk>'] )

    elemX = np.array(elemX)
    elemY = np.array(elemY)

    X_train.append(elemX)
    Y_train.append(elemY)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print('X_train : ', X_train.shape)
print('Y_train : ', Y_train.shape)