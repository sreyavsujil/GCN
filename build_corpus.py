import re
# build corpus


dataset = '20ng'

# Open the dataset file to read document information
f = open('data/' + dataset + '.txt', 'r')
lines = f.readlines()
docs = []

# Iterate through lines to read document content from separate files
for line in lines:
    temp = line.split("\t")
    doc_file = open(temp[0], 'r')
    doc_content = doc_file.read()
    doc_file.close()

    # Print document file name and content for verification
    print(temp[0], doc_content)

    # Replace newline characters with a space in the document content
    doc_content = doc_content.replace('\n', ' ')
    docs.append(doc_content)

# Print document file name and content for verification
corpus_str = '\n'.join(docs)
f.close()

# Write the corpus to a new file in the corpus directory
f = open('data/corpus/' + dataset + '.txt', 'w')
f.write(corpus_str)
f.close()
