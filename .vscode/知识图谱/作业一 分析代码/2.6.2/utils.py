
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from six.moves import cPickle

def plot_with_labels(low_dim_embs, labels, label_colors, filename='tsne.png'):
  plt.figure(figsize=(18, 18))
  x_list = [x[0] for x in low_dim_embs]
  y_list = [x[1] for x in low_dim_embs]
  plt.scatter(x_list, y_list, c=labels, cmap=matplotlib.colors.ListedColormap(label_colors))
  # for i, label in enumerate(labels):
  #   x, y = low_dim_embs[i,:]
  #   plt.scatter(x, y)
  #   plt.annotate(label,
  #                xy=(x, y),
  #                xytext=(5, 2),
  #                textcoords='offset points',
  #                ha='right',
  #                va='bottom')
  plt.savefig(filename)