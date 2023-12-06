# Import necessary libraries
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer

# Print synsets for the word 'dogs' and 'running'
print(wn.synsets('dogs'))
print(wn.synsets('running'))

# Get and print the definition of the first synset of 'dog'
dog = wn.synset('dog.n.01')
print(dog.definition())

# Get and print the definition of the first synset of 'run'
dog = wn.synset('run.n.05')
print(dog.definition())

# Specify the dataset
dataset = 'ohsumed'

# Read the words from the vocabulary file
f = open('data/corpus/' + dataset + '_vocab.txt', 'r')
words = f.readlines()
f.close()

# Initialize a list to store word definitions
definitions = []

# Loop through each word in the vocabulary
for word in words:
    # Clean up the word
    word = word.strip()
    # Get synsets for the word
    synsets = wn.synsets(word)
    # Initialize a list to store synset definitions for the word
    word_defs = []
    # Loop through each synset and get its definition
    for synset in synsets:
        syn_def = synset.definition()
        # Join the synset definitions into a single string for the word
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    # If there are no definitions, use '<PAD>' as a placeholder
    if word_des == '':
        word_des = '<PAD>'
        # Append the word definitions to the list
    definitions.append(word_des)

# Join the word definitions into a single string
string = '\n'.join(definitions)

# Write the word definitions to a new file
f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()

# Initialize a TfidfVectorizer with a maximum of 50000 features
tfidf_vec = TfidfVectorizer(max_features=50000)
# Fit and transform the word definitions using the TfidfVectorizer
tfidf_matrix = tfidf_vec.fit_transform(definitions)
# Convert the tfidf_matrix to a dense array
tfidf_matrix_array = tfidf_matrix.toarray()
# Print the first row of the tfidf_matrix and its length
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

# Initialize a list to store word vectors
word_vectors = []

# Loop through each word and its corresponding vector
for i in range(len(words)):
    word = words[i]
    vector = tfidf_matrix_array[i]
    # Convert the vector to a string
    str_vector = [] 
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    # Join the word and vector into a single string
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    # Append the word vector to the list
    word_vectors.append(word_vector)

# Join the word vectors into a single string
string = '\n'.join(word_vectors)

# Write the word vectors to a new file
f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
f.write(string)
f.close()

