

# In[1]:


""" 1. Implement A* search algorithm """

def aStarAlgo(start_node, stop_node):
    open_set = set(start_node)
    closed_set = set()
    g = {}
    parents = {}

    g[start_node] = 0
    parents[start_node] = start_node
    while len(open_set) > 0:
        n = None
        for v in open_set:
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v
        if n == stop_node or Graph_nodes[n] is None:
            pass
        else:
            for (m, weight) in get_neighbours(n):  
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                    if m in closed_set:
                        closed_set.remove(m)
                        open_set.add(m)
        if n is None:
            print('path does not exist!')
            return None
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('path found: {}'.format(path))
            return path
        open_set.remove(n)
        closed_set.add(n)
    print('path does not exist!')
    return None


def get_neighbours(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None


def heuristic(n):
    H_dist = {'A': 10,'B': 8,'C': 5,'D': 7,'E': 3,'F': 6,'G': 5,'H': 3,'I': 1,'J': 0}
    return H_dist[n]


Graph_nodes = {
    'A': [('B', 6), ('F', 3)],
    'B': [('C', 3), ('D', 2)],
    'C': [('D', 1), ('E', 5)],
    'D': [('C', 1), ('E', 8)],
    'E': [('I', 5), ('J', 5)],
    'F': [('G', 1), ('H', 7)],
    'G': [('I', 3)],
    'H': [('I', 2)],
    'I': [('E', 5), ('J', 3)]
}

aStarAlgo('A', 'J')


# In[2]:


#3.CANDIDATE ELIMINATION
import csv

with open("trainingexamples.csv") as f:
    file = csv.reader(f)
    data = list(file)

    s = data[1][:-1]
    g = [['?' for i in range(len(s))] for j in range(len(s))]

    for i in data:
        if i[-1] == "Yes":
            for j in range(len(s)):
                if i[j] != s[j]:
                    s[j] = '?'
                    g[j][j] = '?'

        elif i[-1] == "No":
            for j in range(len(s)):
                if i[j] != s[j]:
                    g[j][j] = s[j]
                else:
                    g[j][j] = "?"
        print("\nSteps of Candidate Elimination Algorithm", data.index(i) + 1)
        print(s)
        print(g)
    gh = []
    for i in g:
        for j in i:
            if j != '?':
                gh.append(i)
                break
    print("\nFinal specific hypothesis:\n", s)

    print("\nFinal general hypothesis:\n", gh)


# In[4]:


'''8.Write a program to implement k-Nearest Neighbour algorithm to classify 
the iris data set. Print both correct and wrong predictions. Java/Python ML library 
classes can be used for this problem.'''


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_labels, test_size=0.20)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print('Confusion matrix is as follows')
print(confusion_matrix(y_test, y_pred))
print('Accuracy Metrics')
print(classification_report(y_test, y_pred))


# In[5]:


'''2.implementation of AO* algorithm'''

def computeMinimumCostChildNodes(n, h, g):
    
    minCost = 0
    costToChildNode = {}
    costToChildNode[minCost] = []
    
    flag = True
    for nodeInfoTupleList in g[n]:
        cost = 0
        nodeList = []
        
        for c, w in nodeInfoTupleList:
            cost = cost + h[c] + w
            nodeList.append(c)
            
        if flag == True:
            minCost = cost
            costToChildNode[minCost] = nodeList
            flag=False
            
        else:
            if minCost > cost:
                minCost = cost
                costToChildNode[minCost] = nodeList
                
    return minCost, costToChildNode[minCost]

def aostar(status, h, graph, solutionGraph, n, backTracking):
    
    print("HEURISTIC VALUES :", h)
    print("SOLUTION GRAPH :", solutionGraph)
    print("PROCESSING NODE :", n)
    print("-----------------------------------------------------------------------------------------")
        
    if status[n] >= 0:
        minCost, childNodeList = computeMinimumCostChildNodes(n, h, graph)
        print(minCost, childNodeList)
        
        h[n] = minCost
        status[n] = len(childNodeList)
        solved = True
        
        for childNode in childNodeList:
            parent[childNode] = n
            if status[childNode] != -1:
                solved = solved & False
        
        if solved==True:
            status[n] = -1
            solutionGraph[n] = childNodeList
        
        if n != start:
            aostar(status, h, graph, solutionGraph, parent[n], True)
            
        if backTracking == False:
            for childNode in childNodeList:
                status[childNode] = 0
                aostar(status, h, graph, solutionGraph, childNode, False)

graph = {
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],
    'B': [[('G', 1)], [('H', 1)]],
    'C': [[('J', 1)]],
    'D': [[('E', 1), ('F', 1)]],
    'G': [[('I', 1)]],
    'I': [],
    'J': []
}

start = 'A'
parent = {}
status = { 'A': 0,'B': 0,'C': 0,'D': 0,'E': 0,'F': 0,'G': 0,'H': 0,'I': 0,'J': 0 }
solution = {}
h = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
aostar(status, h, graph, solution, start, False)

print("FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE START NODE:", start)
print("------------------------------------------------------------")
print(solution)
print("------------------------------------------------------------")


# In[8]:


#program-9 localWeightRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kernel(point,xmat,k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
    return weights

def localWeight(point,xmat,ymat,k):
    wei = kernel(point,xmat,k)
    W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))
    return W

def localWeightRegression(xmat,ymat,k):
    m,n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
    return ypred

data = pd.read_csv('LR.csv')
colA = np.array(data.colA)
colB = np.array(data.colB)
mcolA = np.mat(colA)
mcolB = np.mat(colB)
m = np.shape(mcolA)[1]
one = np.ones((1,m), dtype=int)
X = np.hstack((one.T,mcolA.T))
print(X.shape)

ypred = localWeightRegression(X,mcolB,0.5)
SortIndex = X[:,1].argsort(0)
xsort = X[SortIndex][:,0]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(colA,colB, color='green')


# In[15]:


"""7.Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same data set for clustering using
k-Means algorithm. Compare the results of these two algorithms and comment on the quality of clustering. You can add
Java/Python ML library classes/API in the program """

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture 
import sklearn.metrics as sm
import pandas as pd
import numpy as np

iris = datasets.load_iris()

X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
Y = pd.DataFrame(iris.target)
Y.columns = ['Targets']

print(X)
print(Y)
colormap = np.array(['red', 'lime', 'black'])

plt.subplot(1,2,1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[Y.Targets], s=40)
plt.title('Real Clustering')

model1 = KMeans(n_clusters=3)
model1.fit(X)

plt.subplot(1,2,2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model1.labels_], s=40)
plt.title('K Mean Clustering')
plt.show()

model2 = GaussianMixture(n_components=3) 
model2.fit(X)

plt.subplot(1,2,1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model2.predict(X)], s=40)
plt.title('EM Clustering')
plt.show()

print("Actual Target is:\n", iris.target)
print("K Means:\n",model1.labels_)
print("EM:\n",model2.predict(X))
print("Accuracy of KMeans is ",sm.accuracy_score(Y,model1.labels_))
print("Accuracy of EM is ",sm.accuracy_score(Y, model2.predict(X)))


# In[3]:


"""5.Build an Artificial Neural Network by implementing the Backpropagation algorithm and test the same using
appropriate data sets """

import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X / np.amax(X, axis=0)  # maximum of X array longitudinally
y = y / 100

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

epoch = 7000  
lr = 0.1  
inputlayer_neurons = 2  # number of features in data set
hiddenlayer_neurons = 3  # number of hidden layers neurons
output_neurons = 1  # number of neurons at output layer
# weight and bias initialization
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))
# draws a random range of numbers uniformly of dim x*y

for i in range(epoch):
    # Forward Propogation
    hinp1 = np.dot(X, wh)
    hinp = hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)
    
    # Backpropagation
    EO = y - output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad
    
    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad
    
    wout += hlayer_act.T.dot(d_output) * lr
    wh += X.T.dot(d_hiddenlayer) * lr
        
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n", output)


# In[9]:


#Program 4-ID3 Decision Tree Algorithm

import pandas as pd
import math
from pprint import pprint

def entropy(data):
    labels = data['Play Tennis']
    total_samples = len(labels)
    unique_labels = set(labels)
    entropy_value = 0

    for label in unique_labels:
        label_count = labels.tolist().count(label)
        probability = label_count / total_samples
        entropy_value -= probability * math.log2(probability)
    return entropy_value

def information_gain(data, feature):
    total_entropy = entropy(data)
    feature_values = set(data[feature])
    weighted_entropy = 0

    for value in feature_values:
        subset = data[data[feature] == value]
        probability = len(subset) / len(data)
        weighted_entropy += probability * entropy(subset)
    return total_entropy - weighted_entropy

def build_tree(data, features):
    labels = data['Play Tennis']

    if len(set(labels)) == 1:
        return labels.iloc[0]

    best_feature = max(features, key=lambda f: information_gain(data, f))
    tree = {best_feature: {}}
    
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value].drop(columns=[best_feature])
        tree[best_feature][value] = build_tree(subset, [f for f in features if f != best_feature])
    return tree

data = pd.read_csv('PlayTennis.csv',names=['Outlook','Temperature','Humidity','Wind','Play Tennis'])

features = list(data.columns[:-1])

decision_tree = build_tree(data, features)
pprint(decision_tree)

print("Training data length:", len(data))


# In[35]:


#NAIVE BAYES

import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Load Data from CSV
data = pd.read_csv('PlayTennis.csv')
print("The first 5 Values of data is :\n", data.head())

# obtain train data and train output
X = data.iloc[:, :-1]
print("\nThe First 5 values of the train data is\n", X.head())

y = data.iloc[:, -1]
print("\nThe First 5 values of train output is\n", y.head())

# convert them in numbers
le_outlook = LabelEncoder()
X.Outlook = le_outlook.fit_transform(X.Outlook)

le_Temperature = LabelEncoder()
X.Temperature = le_Temperature.fit_transform(X.Temperature)

le_Humidity = LabelEncoder()
X.Humidity = le_Humidity.fit_transform(X.Humidity)

le_Windy = LabelEncoder()
X.Wind = le_Windy.fit_transform(X.Wind)

print("\nNow the Train output is\n", X.head())

le_PlayTennis = LabelEncoder()
y = le_PlayTennis.fit_transform(y)
print("\nNow the Train output is\n",y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
print("Accuracy is:", accuracy_score(classifier.predict(X_test), y_test))


# In[ ]:





# In[ ]:




