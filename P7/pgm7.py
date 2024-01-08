
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score,confusion_matrix

dataset=load_iris()

X = pd.DataFrame(dataset.data)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y = pd.DataFrame(dataset.target)
y.columns = ['Targets']

plt.figure(figsize=(14,7))
colormap=np.array(['red','lime','black'])

# REAL PLOT
plt.subplot(1,3,1)
plt.title('Real')
plt.xlabel('Petal Length') 
plt.ylabel('Petal Width')
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y.Targets])

# K-PLOT
model=KMeans(n_clusters=3, random_state=0).fit(X)
plt.subplot(1,3,2)
plt.title('KMeans')
plt.xlabel('Petal Length') 
plt.ylabel('Petal Width')
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[model.labels_])

# GMM PLOT
gmm=GaussianMixture(n_components=3, random_state=0).fit(X)
y_cluster_gmm=gmm.predict(X)
plt.subplot(1,3,3)
plt.title('GMM Classification')
plt.xlabel('Petal Length') 
plt.ylabel('Petal Width')
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y_cluster_gmm])

print('The accuracy score of K-Mean: ',accuracy_score(y, model.labels_))
print('The Confusion matrixof K-Mean:\n',confusion_matrix(y, model.labels_))

print('The accuracy score of EM: ',accuracy_score(y, y_cluster_gmm))
print('The Confusion matrix of EM:\n ',confusion_matrix(y, y_cluster_gmm))
plt.show()

