import numpy as np
import os
import re


os.chdir('..')

lines_file = 'data/movie_lines.txt'
convs_file = 'data/movie_conversations.txt'

lines = open(lines_file, encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open(convs_file, encoding='utf-8', errors='ignore').read().split('\n')

# Creating a dictionary to map line Ids to Lines
id2line = {}
for line in lines:
  _line = line.split(' +++$+++ ')
  if(len(_line) == 5):
    id2line[ _line[0] ] = _line[4]

# Creating a list of all conversations as line ids
convs = []
for line in conv_lines:
  _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
  convs.append(_line.split(','))

# Generate the training data. The input is the current dialog and its output is the next dialog
X = []
Y = []

for conv in convs:
  for i in range(len(conv)-1):
    X.append(id2line[conv[i]])
    Y.append(id2line[conv[i+1]])

max_seq_len = len(X[0])

for element in Y:
    if len(element) > max_seq_len:
        max_seq_len = len(element)

print('Maximum Sequence Length = ', max_seq_len)
# Check the generated data
#index = 1000
#for i in range(index,index+5):
#  print(X[i])
#  print(Y[i])
#  print()
