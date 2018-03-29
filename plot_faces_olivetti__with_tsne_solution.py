# =============================================================================
# T-SNE sur un dataset classique d'images : Olivetti Faces
# Premiere etape:
# - recuperer le programme plot_faces_decomposition puis ne conserver que le debut
# http://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html
# =============================================================================
import numpy as np
import math
import logging
from time import time

from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# modifier le nombre de rangees et de colonnes pour voir tout le dataset
n_row, n_col = 16, 25
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)

# #############################################################################
# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape

# global centering
#faces_centered = faces - faces.mean(axis=0)
# Il est inutile de centrer les faces
faces_centered = faces

# local centering
#faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)


# bien conserver plot_gallery
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


# Premiere figure: afficher les 400 faces
plot_gallery("The 400 Olivetti faces", faces_centered[:n_components])
plt.show()

# utiliser TSNE pour reduire la dimension a 2D
tsne = manifold.TSNE(n_components=2, perplexity=30, init='random',
                     random_state=0)

# mettre les resultats de TSNE dans Y
Y = tsne.fit_transform(faces_centered)

# visualiser la deuxieme figure (avec les points)
plt.figure()
plt.scatter(Y[:, 0], Y[:, 1])
plt.show()

#%%
# Troisieme figure : Pour faciliter le tracé, on recopie les petites images des faces dans une grande

# -----------------------------------------------------------------
# # il n'y a rien à faire ici (PS: could be much more clever!)
size_picture = 3000
shift = 200
dilatation = math.ceil(size_picture / (Y.max() - Y.min()))

Y = Y - Y.min()
Y = Y * dilatation
Y[:,1] = size_picture - Y[:,1]
a = Y.astype(int)

multiple_faces = np.ones((size_picture+shift, size_picture+shift))

plt.figure()
for i, comp in enumerate(faces_centered):
       img = comp.reshape(image_shape)
       multiple_faces[a[i,1]:a[i,1]+image_shape[0],a[i,0]:a[i,0]+image_shape[1]] = img
# -----------------------------------------------------------------

# Troisieme figure: afficher l'image stockee dans multiple_faces avec l'option origin='upper'
plt.imshow(multiple_faces, origin='upper', cmap=plt.cm.gray)
plt.show()

