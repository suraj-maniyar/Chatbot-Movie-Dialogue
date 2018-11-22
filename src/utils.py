import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe


class EncoderRNN(object):
    def __init__(self, num_units=150):
        self.num_units = num_units
        self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    def forward(self, x, sl, reverse=True, final=True):
        state = self.encoder_cell.zero_state(len(x), dtype=tf.float32)
        
        x = tf.convert_to_tensor(x, dtype=tf.float32)

        timestep_x = tf.unstack(x, axis=1)

        if reverse:
            timestep_x = reversed(timestep_x)
        outputs, cell_states = [], []
        for input_step in timestep_x:
            output, state = self.encoder_cell(input_step, state)
            outputs.append(output)
            cell_states.append(state[0])

        outputs = tf.stack(outputs, axis=1)
        cell_states = tf.stack(cell_states, axis=1)

        if final:
            if reverse:
                final_output = outputs[:, -1, :]
                final_cell_state = cell_states[:, -1, :]
            else:
                idxs_last_output = tf.stack([tf.range(len(x)), sl], axis=1)
                final_output = tf.gather_nd(outputs, idxs_last_output)
                final_cell_state = tf.gather_nd(cell_states, idxs_last_output)
            return final_output, final_cell_state
        else:
            return outputs, cell_states

    def save(self, folder_to_save="encoder_model/"):
        saver = tfe.Saver(self.encoder_cell.variables)
        saver.save(folder_to_save)

    def load(self, x, sl, folder_where_saved="encoder_model/"):
        opt, state = self.forward(x, sl)
        saver = tfe.Saver(self.encoder_cell.variables)
        saver.restore(folder_where_saved)
        print('Successfully loaded Encoder Model')
        return opt, state



class DecoderRNN(object):
    def __init__(self, word2idx, idx2word, idx2emb, num_units=150, max_tokens=60):
        self.w2i = word2idx
        self.i2w = idx2word
        self.i2e = idx2emb
        self.num_units = num_units
        self.max_tokens = max_tokens
        self.decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units)
        self.word_predictor = tf.layers.Dense(len(word2idx), activation=None)

    def forward(self, x, sos, state, training=False):
        output = tf.convert_to_tensor(sos, dtype=tf.float32)
        words_predicted, words_logits = [], []

        for mt in range(self.max_tokens):
            output, state = self.decoder_cell( tf.convert_to_tensor(output, dtype=tf.float32) , state )
            logits = self.word_predictor(output)
            logits = tf.nn.softmax(logits)
            pred_word = tf.argmax(logits, 1).numpy()
            if training:
                output = x[:, mt, :]
            else:
                output = [self.i2e[i] for i in pred_word]
            words_predicted.append(pred_word)
            words_logits.append(logits)

        words_logits = tf.stack(words_logits, axis=1)
        words_predicted = tf.stack(words_predicted, axis=1)

        return words_predicted, words_logits


    def save(self, folder_to_save="decoder_model/"):
        saver = tfe.Saver(self.decoder_cell.variables)
        saver.save(folder_to_save)

    def load(self, x, sos, state, enc_output, folder_where_saved="decoder_model/"):
        self.forward(x, sos, (state, enc_output))
        saver = tfe.Saver(self.decoder_cell.variables)
        saver.restore(folder_where_saved)
        print('Successfully loaded Decoder Model')



def cost_function(wl, y, sl, batch_id, batch_size):
    loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=wl)
    mask = np.zeros((y.shape[0], y.shape[1]))
    for i in range(mask.shape[0]):
        temp = np.zeros(mask.shape[1])
        last_len = mask.shape[1] - sl[i + batch_id*batch_size]
        temp[ mask.shape[1]-last_len : ] = 1
        mask[i] = temp
    loss_ = loss_ * mask
    return tf.reduce_mean(loss_)



def get_loss(encoder, decoder, x, y, sl, sos, batch_id, batch_size):
    output, cell_state = encoder.forward(x, sl)
    _, wl = decoder.forward(x, sos, (cell_state, output), training=True)
    loss = cost_function(wl, y, sl, batch_id, batch_size)
    return loss






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
