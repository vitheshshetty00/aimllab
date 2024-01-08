"""
Program 9 : Implement the non-parametric Locally Weighted Regression algorithm in 
order to fit data points. Select appropriate data set for your experiment and draw 
graphs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Kernel function to calculate the weights
def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff = point - X[j]
        weights[j, j] = np.exp(diff * diff.T / (-2.0 * k**2))
    return weights

# Function to calculate the local weights
def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    W = (X.T * (wei * X)).I * (X.T * (wei * ymat.T))
    return W

# Function to perform local weight regression
def localWeightRegression(xmat, ymat, k):
    m, n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k)
    return ypred

# Load the data
file_path = 'd:/Vitheshshetty/temp/aimllab/lwr_data.csv'
data = pd.read_csv(file_path)
colA = np.array(data.colA)
colB = np.array(data.colB)

# Convert to matrix form
mcolA = np.mat(colA)
mcolB = np.mat(colB)

# Add a column of ones to the input data
m = np.shape(mcolA)[1]
one = np.ones((1, m), dtype=int)
X = np.hstack((one.T, mcolA.T))

print(X.shape)

# Perform local weight regression
ypred = localWeightRegression(X, mcolB, 0.5)
# Sort the data
SortIndex = X[:, 1].argsort(0)
xsort = X[SortIndex][:, 0]

# Plot the data and the regression line
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(colA, colB, color='green')
ax.plot(xsort[:, 1], ypred[SortIndex], color='red', linewidth=5)
plt.xlabel('colA')
plt.ylabel('colB')
plt.show()