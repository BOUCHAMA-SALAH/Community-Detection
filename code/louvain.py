from itertools import combinations, chain
from collections import defaultdict
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.sparse import tril
import numpy as np

def modularity_matrix(adj_matrix : csr_matrix) -> csr_matrix:
    k_i = adj_matrix.sum(axis=1)
    k_i = csr_matrix(k_i)
    k_j = csr_matrix(k_i.T) 
    norm = 1 / k_i.sum()
    
    K = norm * (k_i.dot(k_j))

    return norm * (adj_matrix - K)


def modularity(mod_matrix : csr_matrix, communities : list) -> float:
    C = csr_matrix(mod_matrix.shape,  dtype = np.float64)
    C = C.tolil()
    for community in communities:
        for i, j in combinations(community, 2):
            C[i, j] = 1.0
            C[j, i] = 1.0
    C  = C.tocsr()
    Qmatrix = mod_matrix.multiply(C)

    return tril(Qmatrix, 0).sum()

def initialization(adj_matrix):
    # we simply affect each node to a community of its own
    # the community of node i in the returned list can be getted by using the node i as index
    # index_community[i] => community of node i  
    return list(range(adj_matrix.shape[0]))


def getCommunities(index_community):
    communities = defaultdict(set)
    for node, community in enumerate(index_community):
        communities[community].add(node)

    return list(communities.values())


def get_all_edges(nodes):
    return chain(combinations(nodes, 2), ((u, u) for u in nodes))


def first_stage(index_communities, adj_matrix: csr_matrix, max_communities: int , force_merge=False):
    
    # compute the modularity matrix of current communities affectation 
    M = modularity_matrix(adj_matrix) 
    bestSoFarCommunities = index_communities.copy()
    num_communities = len(set(index_communities))
    updated = not (max_communities and num_communities == max_communities)
    while updated:
        updated = False

        for i in range(adj_matrix.shape[0]):

            neighbors = adj_matrix.getrow(i).toarray()
            neighbors = neighbors.reshape((neighbors.shape[1]))
            num_communities = len(set(bestSoFarCommunities))
            if max_communities and num_communities == max_communities:
                break

            best_Q = modularity(M, getCommunities(bestSoFarCommunities))
            max_delta_Q = 0.0
            updated_index_communities, visited_communities = bestSoFarCommunities, set()
            modified = False

            for j, weight in enumerate(neighbors):
                
                # Skip if self-loop or not neighbor
                if i == j or  weight == 0:
                    continue

                neighbor_community = bestSoFarCommunities[j]
                if neighbor_community in visited_communities:
                    continue

                # Remove node i from its community and place it in the community
                # of its neighbor j
                candidate_index_communities = bestSoFarCommunities.copy()
                candidate_index_communities[i] = neighbor_community

                candidate_Q = modularity(M, getCommunities(candidate_index_communities))
                delta_Q = candidate_Q - best_Q
                if delta_Q > max_delta_Q or ( force_merge and (max_delta_Q == 0) ) :
                    updated_index_communities = candidate_index_communities
                    modified = True
                    max_delta_Q = delta_Q

                visited_communities.add(neighbor_community)


            if modified:
                bestSoFarCommunities = updated_index_communities
                updated = True

    return bestSoFarCommunities


def second_stage(node_to_comm, adj_matrix: csr_matrix, partition):
    community_node = defaultdict(set)
    for i, comm in enumerate(node_to_comm):
        community_node[comm].add(i)
    community_node = list(community_node.items())
    updatedAdjMatrix, updatedPartition = [], []
    rows = []
    cols = []
    data = []
    nonzeroRowColsAdj = [ (adj_matrix.nonzero()[0][i],adj_matrix.nonzero()[1][i])for i in range(len(adj_matrix.nonzero()[0]))]
    for i, (comm, nodes) in enumerate(community_node):

        allCommunityNodes = {v for u in nodes for v in partition[u]}
        
        for j, (_, neighbors) in enumerate(community_node):
            if i == j:  # Sum all intra-community weights and add as self-loop
                nonzeroWeightsEdges= [(u,v) for u, v in get_all_edges(nodes) if ( (u,v) in nonzeroRowColsAdj)]
                edge_weights = (adj_matrix[u,v]
                                for u,v in nonzeroWeightsEdges)
                edge_weight =  2* sum(edge_weights)
            else:
                nonzeroWeightsEdges= [(u,v) for u in nodes for v in neighbors if ((u,v) in nonzeroRowColsAdj)]
                edge_weights = (adj_matrix[u,v]
                                for u,v in nonzeroWeightsEdges)
                edge_weight = sum(edge_weights)
            if(edge_weight > 0):
                rows.append(i)
                cols.append(j)
                data.append(edge_weight)

        updatedPartition.append(allCommunityNodes)
    
    updatedAdjMatrix = csr_matrix((data,(rows,cols)), shape=(len(community_node), len(community_node)))

    # TODO: Use numpy more efficiently
    return updatedAdjMatrix, updatedPartition



def louvain(adj_matrix : csr_matrix, max_communities : int = None) -> list:

    index_communities = initialization(adj_matrix)
    partition = [{i} for i in range(adj_matrix.shape[0])]

    M = modularity_matrix(adj_matrix)
    optimal = False
    while not optimal:
        calculated_index_communities = first_stage(
            index_communities,
            adj_matrix,
            max_communities
        )

        if calculated_index_communities == index_communities:
            if not max_communities:
                break

            calculated_index_communities = first_stage(index_communities,adj_matrix,max_communities, force_merge=True)

        adj_matrix, partition = second_stage(calculated_index_communities,adj_matrix,partition)
        
        if max_communities and len(partition) == max_communities:
            break

        index_communities = initialization(adj_matrix)
    
    return partition
