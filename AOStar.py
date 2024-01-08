def recAOStar(n):
    print("Expanding Node:",n)
    and_nodes = allNodes.get(n, {}).get('AND', [])
    or_nodes = allNodes.get(n, {}).get('OR', [])
    if not and_nodes and not or_nodes:
        return

    marked ={}
    while True:
        if len(marked)==len(and_nodes)+len(or_nodes):
            min_cost, min_cost_group = least_cost_group(and_nodes,or_nodes,{})
            change_heuristic(n,min_cost)
            optimal_child_group[n] = min_cost_group
            break

        min_cost, min_cost_group = least_cost_group(and_nodes,or_nodes,marked)
        for node in min_cost_group:
            if node in allNodes:
                recAOStar(node)

        min_cost_verify, min_cost_group_verify = least_cost_group(and_nodes, or_nodes, {})
        if min_cost_group == min_cost_group_verify:
            change_heuristic(n, min_cost_verify)
            optimal_child_group[n] = min_cost_group
            break
        else:
            change_heuristic(n, min_cost)
            optimal_child_group[n] = min_cost_group

        marked[min_cost_group]=1
    return heuristic(n)

def least_cost_group(and_nodes, or_nodes, marked):
    node_wise_cost = {}
    for node_pair in and_nodes:
        if not node_pair[0] + node_pair[1] in marked:
            node_wise_cost[node_pair[0] + node_pair[1]] = heuristic(node_pair[0]) + heuristic(node_pair[1]) + 2
    for node in or_nodes:
        if not node in marked:
            node_wise_cost[node] = heuristic(node) + 1
    min_cost_group = min(node_wise_cost, key=node_wise_cost.get)
    return node_wise_cost[min_cost_group], min_cost_group

def heuristic(n):
    return H_dist[n]

def change_heuristic(n, cost):
    H_dist[n] = cost

def print_path(node):
    print(optimal_child_group[node], end="")
    if len(optimal_child_group[node]) > 1:
        for child in optimal_child_group[node]:
            if child in optimal_child_group:
                print("->", end="")
                print_path(child)
    elif optimal_child_group[node] in optimal_child_group:
        print("->", end="")
        print_path(optimal_child_group[node])

H_dist = {'A': -1, 'B': 4, 'C': 2, 'D': 3, 'E': 6, 'F': 8, 'G': 2, 'H': 0, 'I': 0, 'J': 0}
allNodes = {'A': {'AND': [('C', 'D')], 'OR': ['B']}, 'B': {'OR': ['E', 'F']}, 'C': {'OR': ['G'], 'AND': [('H', 'I')]}, 'D': {'OR': ['J']}}
optimal_child_group = {}
optimal_cost = recAOStar('A')
print('Nodes which gives optimal cost are')
print_path('A')
print('\nOptimal Cost is :: ', optimal_cost)