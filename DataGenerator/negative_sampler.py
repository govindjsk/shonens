from utils import utils

import numpy as np
from tqdm import tqdm

def sample_initial_edge(nodes_to_neighbors):
    edgeidx = np.random.choice(
        sum(len(nodes_to_neighbors[n]) for n in nodes_to_neighbors))
    carry = 0
    for n in nodes_to_neighbors:
        if edgeidx < carry + len(nodes_to_neighbors[n]):
            edge = [n, list(nodes_to_neighbors[n])[edgeidx - carry]]
            break
        carry += len(nodes_to_neighbors[n])
    return edge

def mfinder_sampling(nodes_to_neighbors, k):
    neighbor_edges = []
    induced_edges = set()
    sampled_nodes = set()
    
    while len(sampled_nodes) < k:
        while len(neighbor_edges) == 0:
            edge = sample_initial_edge(nodes_to_neighbors)
            sampled_nodes = set(edge)
            neighbor_edges = set([
                frozenset([node, nnode])
                for node in sampled_nodes for nnode in nodes_to_neighbors[node]
                if nnode not in sampled_nodes])
            neighbor_edges = list(neighbor_edges)

            induced_edges = set()
            induced_edges.add(frozenset(edge))
        
        selected_edge = neighbor_edges[np.random.choice(len(neighbor_edges))]
        induced_edges.add(selected_edge)
        new_node = [n for n in selected_edge.difference(sampled_nodes)][0]
        sampled_nodes.add(new_node)
        
        neighbor_edges = [
            edge for edge in neighbor_edges if new_node not in edge]
        
        new_edges = set()
        
        for node in nodes_to_neighbors[new_node]:
            if node not in sampled_nodes:
                new_edges.add(frozenset([new_node, node]))
            else:
                induced_edges.add(frozenset([new_node, node]))
        
        neighbor_edges.extend(list(new_edges))    
        #assert len(neighbor_edges) == len(set(neighbor_edges))

    return sampled_nodes, induced_edges

def sized_mf_sampling(size_dist, nodes, nodes_to_neighbors, hyperedges):
    vals = [v for v, p in size_dist]
    p = [p for v, p in size_dist]
    sampled_size = np.random.choice(vals, p=p)
    sampled_nodes = {'a', 'b'}
    hyperedges.add(frozenset(sampled_nodes))
    while frozenset(sampled_nodes) in hyperedges:
        sampled_nodes, sampled_edge = mfinder_sampling(
            nodes_to_neighbors, sampled_size)
    hyperedges.remove(frozenset({'a', 'b'}))
    return sampled_nodes, sampled_edge

def negative_sample(
        nodes_to_neighbors, size_dist, num_negative, hyperedges):
    nodes = list(nodes_to_neighbors.keys())
    neg_samples = []
    size_dist = size_dist.items()
    for _ in tqdm(range(num_negative), leave=False):
        sampled_edge = sized_mf_sampling(
            size_dist, nodes, nodes_to_neighbors, hyperedges)
        neg_samples.append(sampled_edge)
    return neg_samples
