import networkx as nx

def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined in bonus.md.
    Args:
      graph....a networkx graph
      node.....a node to score potential new edges for.
    Returns:
      A list of ((node, ni), score) tuples, representing the 
                score assigned to edge (node, ni)
                (note the edge order)
    """
    neighbours1 = set(graph.neighbors(node))
    scores = []
    
    for n in graph.nodes():
        neighbours2 = set(graph.neighbors(n))
        common_neighbours = list(neighbours1 & neighbours2)     #Compute A intersect B
        num1, den1, den2 = 0
        
        if not (graph.has_edge(node,n)) and node != n:
            if (len(common_neighbours)!=0):  
                for c in common_neighbours:           #Numerator value
                    num1 += 1/(graph.degree(c))
                for n1 in neighbours1:                #Denominator value 1
                    den1 += (graph.degree(n1))        
                for n2 in neighbours2:                #Denominator value 2
                    den2 += (graph.degree(n2))
                scores.append(tuple(((node,n),(num1/((1/den1)+(1/den2))))))
            else:
                scores.append(tuple(((node,n),0.0)))
                
    scores = sorted(scores, key=lambda x: (-x[1], x[0]))
    result_list = []
  
    for i in range(len(scores)):
        result_list.append(scores[i])
    
    return result_list
