

from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from random import random
       
x1 = np.array([random() for i in range(50)])
x2 = np.array([random() for i in range(50)])
print(x1)
plt.plot()
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()
# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'r', 'c', 'm', 'y']
markers = ['o', 'v', 's', 'o','v','s']
# KMeans algorithm
K=3
kmeans_model = KMeans(n_clusters=K).fit(X)
plt.plot()
for i, l in enumerate(kmeans_model.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')
plt.xlim([0, 1])
plt.ylim([0, 1])

plt.show()