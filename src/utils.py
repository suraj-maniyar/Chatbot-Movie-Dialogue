import numpy as np



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



# Text pre-processing
def preprocess(text):
    text = text.lower()
    text = text.replace("you're", "you are")
    text = text.replace("he's", "he is")
    text = text.replace("who's", "who is")
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
    text = text.replace("musta", "must have")
    text = text.replace("outta", "out of")
    text = text.replace("didn't", "did not")
    text = text.replace("can't", "can not")
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


# Returns padded sentence as a list
def getSentence(lineNo, id2line, min_seq_length, max_seq_length):
    sentence = id2line[lineNo].split()
    sentence = ['<go>'] + sentence
    if(len(sentence) == max_seq_length+1):
        sentence = sentence[:-1]

    assert( (len(sentence) <= max_seq_length) and (len(sentence) >= min_seq_length) ), \
    ("Length of sentence at line %s is %d" % (lineNo, len(sentence)))

    pad_len = max_seq_length - len(sentence)
    pad = ['<pad>'] * pad_len
    sentence = pad + sentence

    if(sentence[-1] != '<eos>'):
        if(len(sentence) == max_seq_length):
            sentence[-1] = '<eos>'
        else:
            sentence.append('<eos>')

    return sentence


# Returns padded vector corresponding to given line number
def getVector(lineNo, id2line, glove_model, vocab, min_seq_length, max_seq_length, verbose=0):

    sentence = getSentence(lineNo, id2line, min_seq_length, max_seq_length)

    vect = []
    for i in range(len(sentence)):
        if(sentence[i] in vocab):
            vect.append( glove_model[sentence[i]] )
            if(verbose):
                print(sentence[i])
        else:
            if(verbose):
                print('UNK')
            vect.append( glove_model['<unk>'] )
    vect = np.array(vect)
    return vect



# Returns nearest word to the given vector
def getWord(vec, word2vec):
    key_dist = {}
    for key in word2vec.keys():
        key_dist[key] = np.sum(np.square( vec - word2vec[key] ))
    min_key = min(key_dist, key=key_dist.get)
    print('Distance = ', np.sqrt(np.sum(np.square( vec - word2vec[key] ))))
    return min_key
