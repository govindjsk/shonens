import math
from collections import defaultdict

from DataGenerator import negative_sampler as sampler
from utils import utils

def read_dataset(dataset):
    return utils.read_benson_dataset(dataset)

def get_ground_train_test_split(
        dataset, hyperedges_to_timestamp, split):
    if dataset == 'DBLP':
        ground_edges = {
            frozenset(edge) for edge, ts in hyperedges_to_timestamp.items()
            if ts in range(2013, 2015)}
        train_edges = {
            frozenset(edge) for edge, ts in hyperedges_to_timestamp.items()
            if ts in range(2015, 2016)}
        test_edges = {
            frozenset(edge) for edge, ts in hyperedges_to_timestamp.items()
            if ts in range(2016, 2018)}
        test_edges = test_edges.difference(train_edges)
        ground_edges = list(ground_edges)
        test_edges = [edge for edge in test_edges if len(edge) > 2]
        train_edges = [edge for edge in train_edges if len(edge) > 2]
    elif dataset == 'MAG-Geo':
        ground_edges = {
            frozenset(edge) for edge, ts in hyperedges_to_timestamp.items()
            if ts <= 2014}
        train_edges = {
            frozenset(edge) for edge, ts in hyperedges_to_timestamp.items()
            if 2014 < ts <= 2016}
        test_edges = {
            frozenset(edge) for edge, ts in hyperedges_to_timestamp.items()
            if ts > 2016}
        ground_edges = list(ground_edges)
        test_edges = [edge for edge in test_edges if len(edge) > 2]
        train_edges = [edge for edge in train_edges if len(edge) > 2]
    else:
        sethyperedges = list(map(lambda x: x[0], sorted(
            hyperedges_to_timestamp.items(), key=lambda x: x[1])))
        sep1 = math.ceil(split[0] * len(sethyperedges))
        sep2 = math.ceil(split[1] * len(sethyperedges))
        ground_edges = sethyperedges[:sep1]
        train_edges = sethyperedges[sep1: sep2]
        test_edges = [edge for edge in sethyperedges[sep2:] if len(edge) > 2]
        train_edges = [edge for edge in train_edges if len(edge) > 2]

    return ground_edges, train_edges, test_edges

def generate_hyperedge_size_dist(hyperedges):
    size_dist = defaultdict(int)
    for edge in hyperedges:
        size_dist[len(edge)] += 1
    if 1 in size_dist:
        del size_dist[1]
    if 2 in size_dist:
        del size_dist[2]
    total = sum(v for k, v in size_dist.items())
    for i in size_dist:
        size_dist[i] = float(size_dist[i]) / total
    return size_dist

def generate_negative_samples_for_hyperedges(
        train_edges, hyperedges, neg_samples_size):
    edges = {
        frozenset({u, v}) for hedge in train_edges
        for u in hedge for v in hedge if u > v}
    nodes_to_neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        nodes_to_neighbors[u].add(v)
        nodes_to_neighbors[v].add(u)

    size_dist = generate_hyperedge_size_dist(train_edges)
    
    print('Generating Negative Samples')
    total = math.ceil(neg_samples_size)
    neg_samples = sampler.negative_sample(
        nodes_to_neighbors, size_dist, total, hyperedges)
    negative_hyperedges = [frozenset(x) for x, y in neg_samples]
    
    return negative_hyperedges
