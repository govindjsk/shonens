import numpy as np
import scipy
import scipy.sparse as sp
import torch

def get_device():
    return ('cuda' if torch.cuda.is_available() else 'cpu')

def parse_hyperedges(hyperedges_nodes, hyperedges_lengths):
    nodes = 0
    hyperedges = []
    for i in hyperedges_lengths:
        hyperedges.append(hyperedges_nodes[nodes: nodes+i])
        nodes += i
    return hyperedges

def clean_hyperedges(hyperedges, hypergraph_nodes):
    node_occurrence = set()
    for hedge in hyperedges:
        for node in hedge:
            node_occurrence.add(node)
    if len(node_occurrence) == len(hypergraph_nodes):
        return hyperedges, hypergraph_nodes
    new_node_map = {node: idx for idx, node in enumerate(node_occurrence)}
    new_hyperedges = []
    for hedge in hyperedges:
        new_hyperedges.append([new_node_map[node] for node in hedge])
    return new_hyperedges, list(range(len(new_node_map)))

def get_hypermatrix(hyperedges, n_nodes=None):
    H_rows = []
    H_cols = []
    H_values = []
    cnodes = 0
    for idx, edge in enumerate(hyperedges):
        for node in edge:
            H_rows.append(node)
            H_cols.append(idx)
            H_values.append(1)
            cnodes = max(cnodes, node)
    cnodes += 1
    H = sp.csr_matrix(
        (H_values, (H_rows, H_cols)),
        shape=(n_nodes or cnodes, len(hyperedges)))
    return H

def get_graph_from_hypergraph_matrix(hypergraph):
    H = sp.csr_matrix(hypergraph)
    G = np.dot(H, H.T)
    return G

def read_metabolites_dataset(name):
    mat = scipy.io.loadmat('./datasets/metabolites/%s.mat' % name)
    H = mat['Model']['S'][0][0]
    H[H != 0] = 1
    H = H.toarray()

    hypergraph_nodes = range(H.shape[0])
    hyperedges = []
    hyperedges_timestamps = []
    for hedge in H.T:
        nodes = np.where(hedge > 0)[0].tolist()
        hyperedges.append(nodes)
        hyperedges_timestamps.append(0)
    
    hyperedges, hypergraph_nodes = clean_hyperedges(
        hyperedges, hypergraph_nodes)
        
    return hyperedges, hyperedges_timestamps, hypergraph_nodes


def read_benson_dataset(name):
    name_to_path = {
        'email-enron': 'email-Enron',
        'contact-primary-school': 'contact-primary-school',
        'math-sx': 'tags-math-sx',
        'NDC': 'NDC-substances',
        'DBLP': 'coauth-DBLP',
        'contact-high-school': 'contact-high-school',
        'MAG-Geo': 'coauth-MAG-Geology',
        'DAWN': 'DAWN'
    }
    p1 = name_to_path[name]
    if name in ['DBLP', 'NDC', 'DAWN'] :
        p2 = 'short-%s' % p1
    elif name in ['math-sx', 'MAG-Geo']:
        p2 = 'short-2-%s' % p1
    else:
        p2 = p1
    with open('./datasets/%s/%s-nverts.txt' % (p1, p2)) as f:
        hyperedges_lengths = list(map(int, f.readlines()))
    
    with open('./datasets/%s/%s-simplices.txt' % (p1, p2)) as f:
        hyperedges_nodes = list(map(lambda x: int(x)-1, f.readlines()))

    with open('./datasets/%s/%s-times.txt' % (p1, p2)) as f:
        hyperedges_timestamps = list(map(int, f.readlines()))

    hypergraph_nodes = range(max(hyperedges_nodes) + 1)
    hyperedges = parse_hyperedges(hyperedges_nodes, hyperedges_lengths)
    hyperedges, hypergraph_nodes = clean_hyperedges(
        hyperedges, hypergraph_nodes)
    return hyperedges, hyperedges_timestamps, hypergraph_nodes

def read_dataset(name):
    benson_datasets = [
        'email-enron', 'contact-primary-school', 'math-sx', 'NDC', 'DBLP',
        'contact-high-school', 'MAG-Geo', 'DAWN']
    if name in benson_datasets:
        return read_benson_dataset(name)
    else:
        return read_metabolites_dataset(name)

def associate_min_timestamp_with_hyperedges(
        hyperedges, hyperedges_timestamps):
    hyperedges_timestamps = [(idx, ts) for idx, ts in enumerate(
        hyperedges_timestamps)]

    hyperedges_to_timestamp = {}
    for edgeidx, ts in  hyperedges_timestamps:
        edge = frozenset(hyperedges[edgeidx])
        if edge not in hyperedges_to_timestamp:
            hyperedges_to_timestamp[edge] = ts
        hyperedges_to_timestamp[edge] = min(hyperedges_to_timestamp[edge], ts)

    return hyperedges_to_timestamp
