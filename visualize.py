# Import necessary libraries
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

# Read the lines from the file containing information about the documents
f = open('data/ohsumed_shuffle.txt', 'r')
lines = f.readlines()
f.close()

# Read the lines from the file containing document vectors
f = open('data/ohsumed_doc_vectors.txt', 'r')
embedding_lines = f.readlines()
f.close()

# Initialize sets and lists to store target names, labels, and document vectors
target_names = set()
labels = []
docs = []

# Loop through each line in the document information file
for i in range(len(lines)):
    line = lines[i].strip()
    
    # Split the line by tab character
    temp = line.split('\t')
    
    # Check if the line contains 'test'
    if temp[1].find('test') != -1:
        # Extract the label from the line
        labels.append(temp[2])
        
        # Extract the document vector values from the corresponding line in the embedding file
        emb_str = embedding_lines[i].strip().split()
        values_str_list = emb_str[1:]
        values = [float(x) for x in values_str_list]
        
        # Append the vector to the list of documents and the label to the list of labels
        docs.append(values)
        
        # Add the label to the set of target names
        target_names.add(temp[2])

# Convert target names to a list
target_names = list(target_names)

# Convert labels to a NumPy array
label = np.array(labels)

# Use t-SNE to reduce the dimensionality of the document vectors to 2D
fea = TSNE(n_components=2).fit_transform(docs)

# Create a PDF file to save the plot
pdf = PdfPages('ohsumed_gcn_doc_test_2nd_layer.pdf')

# Get unique classes (labels)
cls = np.unique(label)

# Separate the features based on class
fea_num = [fea[label == i] for i in cls]

# Plot each class with a different marker
for i, f in enumerate(fea_num):
    if cls[i] in range(10):
        plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='+')
    else:
        plt.scatter(f[:, 0], f[:, 1], label=cls[i])

# Uncomment the following lines if you want to customize the legend and plot appearance
# plt.legend(ncol=2)
# plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.48, -0.08), fontsize=11)
# plt.ylim([-20, 35])
# plt.title(md_file)

# Ensure a tight layout and save the plot to the PDF file
plt.tight_layout()
pdf.savefig()

# Show the plot
plt.show()

# Close the PDF file
pdf.close()
