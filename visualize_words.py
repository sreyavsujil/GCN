# Import necessary libraries
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

# Read the word vectors from the file
f = open('data/ohsumed_word_vectors_1.txt', 'r')
embedding_lines = f.readlines()
f.close()

# Initialize sets and lists to store target names, labels, and document vectors
target_names = set()
labels = []
docs = []
# Loop through each line in the embedding file
for i in range(len(embedding_lines)):
    line = embedding_lines[i].strip()
    # Split the line by tab character
    temp = line.split('\t')
    # Extract the word vector values
    emb_str = embedding_lines[i].strip().split()
    values_str_list = emb_str[1:]
    values = [float(x) for x in values_str_list]
    # Find the index of the maximum value in the vector as the label
    label = np.argmax(values)
    # Append the vector to the list of documents and the label to the list of labels
    docs.append(values)
    target_names.add(label)
    labels.append(label)

# Get unique target names (labels)
target_names = list(target_names)

# Convert labels to a NumPy array
label = np.array(labels)

# Use t-SNE to reduce the dimensionality of the document vectors to 2D
fea = TSNE(n_components=2).fit_transform(docs)
pdf = PdfPages('ohsumed_gcn_word_2nd_layer.pdf')
# Get unique classes (labels)
cls = np.unique(label)

# cls=range(10)
# Separate the features based on class
fea_num = [fea[label == i] for i in cls]
# Plot each class with a different marker
for i, f in enumerate(fea_num):
    if cls[i] in range(10):
        plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='+')
    else:
        plt.scatter(f[:, 0], f[:, 1], label=cls[i])
# plt.legend(ncol=2,  )
# plt.legend(ncol=5,loc='upper center',bbox_to_anchor=(0.48, -0.08),fontsize=11)
# plt.ylim([-20,35])
# plt.title(md_file)
plt.tight_layout()
pdf.savefig()
# Show the plot
plt.show()
pdf.close()
