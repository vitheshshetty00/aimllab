def AStarSeach(start_node,stop_node):
    open_set = set(start_node)
    closed_set = set()
    g={}
    parents = {}
    g[start_node] =0
    parents[start_node]=start_node
    
    while len(open_set)>0:
        current_node=None
        
        for v in open_set:
            if current_node==None or g[v]+ heuristic(v)<g[current_node] + heuristic(current_node):
                current_node=v
        
        if current_node!= stop_node and Graph_Nodes[current_node] != None:
            for (neighbor_node,cost) in get_neighbors(current_node):
                if neighbor_node not in open_set and neighbor_node not in closed_set:
                    open_set.add(neighbor_node)
                    parents[neighbor_node]=current_node
                    g[neighbor_node]=g[current_node]+cost
                else:
                    if g[neighbor_node]>g[current_node]+cost:
                        g[neighbor_node]=g[current_node]+cost
                        parents[neighbor_node]=current_node
                        
                        if neighbor_node in closed_set:
                            closed_set.remove(neighbor_node)
                            open_set.add(neighbor_node)
        if current_node == stop_node:
            path=[]
            while parents[current_node] != current_node:
                path.append(current_node)
                current_node=parents[current_node]
            path.append(start_node)
            path.reverse()
            print("Path:{}".format(path))
            return path

        open_set.remove(current_node)
        closed_set.add(current_node)
    
    print("Path does'nt exist")
    return None
            

def get_neighbors(node):
    if node in Graph_Nodes:
        return Graph_Nodes[node]
    else:
        return None

def heuristic(node):
    H_dist = {
 'S': 8,
 'A': 8,
 'B': 4,
 'C': 3,
 'D': 1000,
 'E': 1000,
 'G': 0,
 
 }

    return H_dist[node]

Graph_Nodes =  {'S': [['A', 1], ['B', 5], ['C', 8]],
 'A': [['D', 3], ['E', 7], ['G', 9]],
 'B': [['G', 4]],
 'C': [['G', 5]],
 'D': None,
 'E': None}

AStarSeach('S','G')