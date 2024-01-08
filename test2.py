from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix,accuracy_score


iris =load_iris()
X=iris.data
y=iris.target

le=LabelEncoder()
y=le.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train,y_train)

ypred = classifier.predict(x_test)

for i in range(len(ypred)):
    print(y_test[i]," ",ypred[i])

print(confusion_matrix(y_test,ypred))

print(accuracy_score(y_test,ypred))






# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def kernel(point,xmat,k):
#     m=np.shape(xmat)[0]
#     weight = np.mat(np.eye(m))
#     for i in range(m):
#         diff=point-X[i]
#         weight[i,i]=np.exp(diff*diff.T/(-2*k**2))
#     return weight

# def localWeight(point,xmat,ymat,k):
#     wei=kernel(point,xmat,k)
#     W=(X.T*(wei*X)).I*(X.T*(wei*ymat.T))
#     return W

# def localWeightRegression(xmat,ymat,k):
#     m=np.shape(xmat)[0]
#     ypred=np.zeros(m)
#     for i in range(m):
#         ypred[i]=xmat[i]*localWeight(xmat[i],xmat,ymat,k)
#     return ypred
        

# data = pd.read_csv('lwr_data.csv')
# colA = np.array(data.colA)
# colB=np.array(data.colB)

# mColA = np.mat(colA)
# mColB = np.mat(colB)

# m=np.shape(mColA)[1]
# ones=np.ones((1,m),dtype=int)
# X = np.hstack((ones.T,mColA.T))
# print(X.shape)

# ypred = localWeightRegression(X,mColB,0.5)

# sortIndex=X[:,1].argsort(0)
# xsort=X[sortIndex][:,0]

# fig=plt.figure()
# ax=plt.subplot(1,1,1)
# ax.scatter(colA,colB,color='green')
# ax.plot(xsort[:,1],ypred[sortIndex],color='red',linewidth=3)
# plt.xlabel("ColA")
# plt.ylabel("ColB")
# plt.show()