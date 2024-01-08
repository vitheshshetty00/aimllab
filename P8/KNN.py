"""
Program 8 : Write a program to implement k-Nearest Neighbour algorithm to classify 
the iris data set. Print both correct and wrong predictions. Java/Python ML library 
classes can be used for this problem.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
""" Alternative for Importing the dataset if it iris.csv available on system
dataset = pd.read_csv('iris.csv')
Dividing data into features and labels
feature_columns = ['sepal_length', 'sepal_width', 'petal_length','petal_width']
X = dataset[feature_columns].values
y = dataset['species'].values
"""

# Dividing data into features and labels
iris = load_iris()
X = iris.data
y = iris.target
# Transforming string labels into numbers
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting K-NN to the Training set
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Printing predicted and actual values
print("y_pred y_test")
for i in range(len(y_pred)):
    print(y_pred[i], " ", y_test[i])

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculating and printing accuracy score
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')