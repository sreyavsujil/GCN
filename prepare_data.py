#!/usr/bin/python
#-*-coding:utf-8-*-

# Define dataset information
dataset_name = 'own'
sentences = ['Would you like a plain sweater or something else?​', 'Great. We have some very nice wool slacks over here. Would you like to take a look?']
labels = ['Yes' , 'No' ]
train_or_test_list = ['train', 'test']

# Create metadata for each sentence
meta_data_list = []
for i in range(len(sentences)):
    # Format: index \t train/test \t label
    meta = str(i) + '\t' + train_or_test_list[i] + '\t' + labels[i]
    meta_data_list.append(meta)
# Combine metadata into a string with line breaks
meta_data_str = '\n'.join(meta_data_list)
# Write metadata to a file
f = open('data/' + dataset_name + '.txt', 'w')
f.write(meta_data_str)
f.close()
# Combine sentences into a string with line breaks
corpus_str = '\n'.join(sentences)

# Write sentences to a file in the 'corpus' directory
f = open('data/corpus/' + dataset_name + '.txt', 'w')
f.write(corpus_str)
f.close()
