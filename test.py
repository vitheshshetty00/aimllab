def recAOStar(n):
    
    print("Expanding Node : ", n)
    and_nodes = []
    or_nodes = []
    
    # Segregation of AND and OR nodes
    if n in allNodes:
        if 'AND' in allNodes[n]:
            and_nodes = allNodes[n]['AND']
        if 'OR' in allNodes[n]:
            or_nodes = allNodes[n]['OR']
    
    # If leaf node then return
    if len(and_nodes) == 0 and len(or_nodes) == 0:
        return
    
    solvable = False
    marked = {}
    
    while not solvable:
        # If all the child nodes are visited and expanded, take the least cost of all the child nodes
        if len(marked) == len(and_nodes) + len(or_nodes):
            min_cost_least, min_cost_group_least = least_cost_group(and_nodes, or_nodes, {})
            solvable = True
            change_heuristic(n, min_cost_least)
            optimal_child_group[n] = min_cost_group_least
            continue
        
        # Least cost of the unmarked child nodes
        min_cost, min_cost_group = least_cost_group(and_nodes, or_nodes, marked)
        is_expanded = False
        
        # If the child nodes have sub trees then recursively visit them to recalculate the heuristic of the child node
        if len(min_cost_group) > 1:
            if min_cost_group[0] in allNodes:
                is_expanded = True
                recAOStar(min_cost_group[0])
            if min_cost_group[1] in allNodes:
                is_expanded = True
                recAOStar(min_cost_group[1])
        else:
            if min_cost_group in allNodes:
                is_expanded = True
                recAOStar(min_cost_group)
        
        # If the child node had any subtree and expanded, verify if the new heuristic value is still the least among all nodes
        if is_expanded:
            min_cost_verify, min_cost_group_verify = least_cost_group(and_nodes, or_nodes, {})
            if min_cost_group == min_cost_group_verify:
                solvable = True
                change_heuristic(n, min_cost_verify)
                optimal_child_group[n] = min_cost_group
        # If the child node does not have any subtrees then no change in heuristic, so update the min cost of the current node
        else:
            solvable = True
            change_heuristic(n, min_cost)
            optimal_child_group[n] = min_cost_group
        
        # Mark the child node which was expanded
        marked[min_cost_group] = 1
    
    return heuristic(n)

# Function to calculate the min cost among all the child nodes
def least_cost_group(and_nodes, or_nodes, marked):
    node_wise_cost = {}
    
    for node_pair in and_nodes:
        if not node_pair[0] + node_pair[1] in marked:
            cost = 0
            cost = cost + heuristic(node_pair[0]) + heuristic(node_pair[1]) + 2
            node_wise_cost[node_pair[0] + node_pair[1]] = cost
    
    for node in or_nodes:
        if not node in marked:
            cost = 0
            cost = cost + heuristic(node) + 1
            node_wise_cost[node] = cost
    
    min_cost = 999999
    min_cost_group = None
    
    # Calculates the min heuristic
    for costKey in node_wise_cost:
        if node_wise_cost[costKey] < min_cost:
            min_cost = node_wise_cost[costKey]
            min_cost_group = costKey
    
    return [min_cost, min_cost_group]

# Returns heuristic of a node
def heuristic(n):
    return H_dist[n]

# Updates the heuristic of a node
def change_heuristic(n, cost):
    H_dist[n] = cost
    return

# Function to print the optimal cost nodes
def print_path(node):
    print(optimal_child_group[node], end="")
    node = optimal_child_group[node]
    
    if len(node) > 1:
        if node[0] in optimal_child_group:
            print("->", end="")
            print_path(node[0])
        if node[1] in optimal_child_group:
            print("->", end="")
            print_path(node[1])
    else:
        if node in optimal_child_group:
            print("->", end="")
            print_path(node)

# Describe the heuristic here
H_dist = {
    'A': -1,
    'B': 4,
    'C': 2,
    'D': 3,
    'E': 6,
    'F': 8,
    'G': 2,
    'H': 0,
    'I': 0,
    'J': 0
}

# Describe your graph here 
allNodes = {
    'A': {'AND': [('C', 'D')], 'OR': ['B']},
    'B': {'OR': ['E', 'F']},
    'C': {'OR': ['G'], 'AND': [('H', 'I')]},
    'D': {'OR': ['J']}
}

optimal_child_group = {}
optimal_cost = recAOStar('A')
print('Nodes which gives optimal cost are')
print_path('A')
print('\nOptimal Cost is :: ', optimal_cost)




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# def kernel(point,xmat,k):
#     m,n = np.shape(xmat)
#     weights = np.mat(np.eye(m))
#     for j in range(m):
#         diff = point - X[j]
#         weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
#     return weights

# def localWeight(point,xmat,ymat,k):
#     wei = kernel(point,xmat,k)
#     W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))
#     return W

# def localWeightRegression(xmat,ymat,k):
#     m,n = np.shape(xmat)
#     ypred = np.zeros(m)
#     for i in range(m):
#         ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
#     return ypred

# data = pd.read_csv('lwr_data.csv')
# colA = np.array(data.colA)
# colB = np.array(data.colB)
# mcolA = np.mat(colA)
# mcolB = np.mat(colB)
# m = np.shape(mcolA)[1]
# one = np.ones((1,m), dtype=int)
# X = np.hstack((one.T,mcolA.T))
# print(X.shape)

# ypred = localWeightRegression(X,mcolB,0.5)
# SortIndex = X[:,1].argsort(0)
# xsort = X[SortIndex][:,0]
# fig, ax = plt.subplots()
# ax.scatter(colA,colB, color='green')
# ax.plot(xsort[:, 1], ypred[SortIndex], color='red', linewidth=5)
# plt.xlabel('colA')
# plt.ylabel('colB')
# plt.show()



# import pandas as pd
# import math
# from pprint import pprint

# def entropy(data):
#     labels = data['Play Tennis']
#     total_samples = len(labels)
#     unique_labels = set(labels)
#     entropy_value = 0

#     for label in unique_labels:
#         label_count = labels.tolist().count(label)
#         probability = label_count / total_samples
#         entropy_value -= probability * math.log2(probability)
#     return entropy_value

# def information_gain(data, feature):
#     total_entropy = entropy(data)
#     feature_values = set(data[feature])
#     weighted_entropy = 0

#     for value in feature_values:
#         subset = data[data[feature] == value]
#         probability = len(subset) / len(data)
#         weighted_entropy += probability * entropy(subset)
#     return total_entropy - weighted_entropy

# def build_tree(data, features):
#     labels = data['Play Tennis']

#     if len(set(labels)) == 1:
#         return labels.iloc[0]

#     best_feature = max(features, key=lambda f: information_gain(data, f))
#     tree = {best_feature: {}}
#     print(tree)
    
#     for value in data[best_feature].unique():
#         subset = data[data[best_feature] == value].drop(columns=[best_feature])
#         tree[best_feature][value] = build_tree(subset, [f for f in features if f != best_feature])
#     return tree

# def test_sample(tree, sample):
#     for feature in tree.keys():
#         value = sample[feature]
#         if value in tree[feature].keys():
#             subtree = tree[feature][value]
#             if type(subtree) is dict:
#                 return test_sample(subtree, sample)
#             else:
#                 return subtree
#         else:
#             return None

# data = pd.read_csv('playtennis.csv',names=['Outlook','Temperature','Humidity','Wind','Play Tennis'])
# features = list(data.columns[:-1])

# decision_tree = build_tree(data, features)
# pprint(decision_tree)

# print("Training data length:", len(data))
# test_data = {
#     'Outlook': 'Overcast',
#     'Wind': 'Strong',
#     'Humidity': 'High'
# }

# result = test_sample(decision_tree, test_data)
# print("The predicted class label for the test sample is:", result)







# import pandas as pd
# data =pd.read_csv('trainingexamples.csv').values
# s=data[0][:-1]
# g=[['?' for i in range(len(s))] for j in range(len(s)) ]
# for index, i in enumerate(data):
#     if i[-1]=='Yes':
#         for j in range(len(s)):
#             if i[j]!=s[j]:
#                 s[j]='?'
#                 g[j][j]='?'
#     elif i[-1]=='No':
#         for j in range(len(s)):
#             if i[j]!=s[j]:
#                 g[j][j]=s[j]
#     print("Step:",index+1)
#     print(s)
#     print(g)   
# gh=[i for i in g for j in i if j!='?']         
# print("final specific Hypothesis:",s)
# print("final Genaral hypothesis:",gh)
        
    










# def aStarAlgo(start_node,stop_node):
#     open_set = set(start_node)
#     closed_set=set()
#     g={}
#     parents={}
    
#     parents[start_node]=start_node
#     g[start_node]=0
    
#     while(len(open_set)!=0):
#         n=None
#         for v in open_set:
#             if n== None or g[v]+heuristic(v)<g[n]+heuristic(n):
#                 n=v
#         if n==stop_node or Graph_nodes[n] is None:
#             pass
#         else:
#             for (m,weight) in get_neighbours(n):
#                 if m not in open_set and m not in closed_set:
#                     open_set.add(m)
#                     parents[m]=n
#                     g[m]=g[n]+weight
#                 else:
#                     if g[m]>g[n]+weight:
#                         g[m]=g[n]+weight
#                         parents[m]=n
#                         if m in closed_set:
#                             closed_set.remove(m)
#                             open_set.add(m)
#         if n is None:
#             return None
#         if n== stop_node:
#             path = []
#             while parents[n]!=n:
#                 path.append(n)
#                 n=parents[n]
#             path.append(n)
#             path.reverse()
#             print(path)
#             return path
#         open_set.remove(n)
#         closed_set.add(n)
#     return None
        
                            
# def get_neighbours(n):
#     if n in Graph_nodes:
#         return Graph_nodes[n]
#     else:
#         return None
    

# def heuristic(n):
#     H_dist = {'A': 10,'B': 8,'C': 5,'D': 7,'E': 3,'F': 6,'G': 5,'H': 3,'I': 1,'J': 0}
#     return H_dist[n]


# Graph_nodes = {
#     'A': [('B', 6), ('F', 3)],
#     'B': [('C', 3), ('D', 2)],
#     'C': [('D', 1), ('E', 5)],
#     'D': [('C', 1), ('E', 8)],
#     'E': [('I', 5), ('J', 5)],
#     'F': [('G', 1), ('H', 7)],
#     'G': [('I', 3)],
#     'H': [('I', 2)],
#     'I': [('E', 5), ('J', 3)]
# }

# aStarAlgo('A', 'J')