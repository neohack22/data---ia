# =============================================================================
# COMPRENDRE T-SNE:  Passer de 2D Ã  1D
# Le but de cet exemple est de generer quatre "blobs".
# On appelle "blob" un amas de points generes par une gaussienne.
# Puis, on realise une visualisation qui passe de 2D Ä‚ 1D
# grÃ¢ce Ã  l'utilisation de l'algorithme T-SNE
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time

from sklearn.datasets.samples_generator import make_blobs

plt.close('all')

# Voici les centres des 4 blobs (PS: ne pas tenir compte de la variable shift)
shift = 20
centers = [(5, 5, 5+shift), (15, 5, 5+shift), (15, 5, 15+shift), (15, 15, 15+shift)]
n_samples = 1000

# Par blobs, on souhaite creer 1000 observations qui contiennent 3 "features"
# On mettra les observations dans X, les targets dans y
X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=3, random_state=0)

# Premiere figure : on visualise les blobs en 3D
# Lorsque l'on affichera les blobs en couleur, quelle variable utilisera-t-on pour
# la couleur ?
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Sur la premiere figure, on rajoute une projection 2D des blobs sur le plan au sol (z=0)
# On utilise le marker "x" pour la projection au sol
# Que peut-on observer au sol ?
ax.scatter(X[:, 0], X[:, 1], np.zeros(n_samples), c=y, marker="x")
plt.show()

# Deuxieme figure : on visualise les blobs en 2D grâce a l'algorithme T-SNE
tsne = manifold.TSNE(n_components=2, perplexity=30, init='random',
                     random_state=0)
Y = tsne.fit_transform(X)
print(Y.shape)

plt.figure()
plt.scatter(Y[:, 0], Y[:, 1], c=y)
plt.show()

