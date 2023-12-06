##  Implementing Text classification Using Graph Convolution Network
Text classification is a major problem,concern and research going  using natural language processing (NLP). A new model of deep learning has come to be known as the graph convolutional neural network. Graph convolutional network (GCN) also has advantages compared with the traditional neural network. The above CNN and RNN cannot process the feature representation of graph embedding in non-sequential order. GCN is propagated on each vertex separately, ignoring the order of input between vertices. In other words, the output of GCN is not shifted with the input order of the vertices. 
## Modules
Text Graph Construction: This module involves constructing a graph representation of the input text corpus, where each document and words are represented as a node and edges capture the relationships between documents based on their textual similarity. It is a novel approach to constructing this graph using a combination of word co-occurrence and document similarity measures.


Text Graph Convolutional Networks: This module involves applying graph convolutional neural networks to the constructed text graph for text classification. A novel approach which involves learning both node-level and graph-level representations of the text corpus using a combination of convolutional and pooling operations.
## Prerequisites
Python 3.x

TensorFlow

nltk

Numpy
## Reproducing Results

1. Run `python remove_words.py 20ng`

2. Run `python build_graph.py 20ng`

3. Run `python train.py 20ng`

4. Change `20ng` in above 3 command lines to  `ohsumed`  when producing results for Ohsumed dataset.


