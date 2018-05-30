from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def plot_words(word_vectors,word_list,method_type='TSNE',show_word=True,only_word=False):
    X = np.array(word_vectors)
    if method_type=='TSNE':
        model = TSNE(n_components=2, random_state=2)
        Y = model.fit_transform(X)
    elif method_type=='PCA':
        model = PCA(n_components=2)
        model.fit(X)
        Y = model.transform(X)
    else:
        print("Unknown 'method_type' please choose from TSNE/PCA.")
    np.set_printoptions(suppress=True)
    if only_word:
        Y=Y[0:len(word_list),:]
    plt.scatter(Y[:,0],Y[:,1])
    if show_word:
        for label, x, y in zip(word_list,Y[:,0],Y[:,1]):
            plt.annotate(label,xy=(x,y),xytext=(0,0),textcoords='offset points')
    else:
        plt.plot(Y[:,0],Y[:,1],'r.')
    plt.show()
    return model