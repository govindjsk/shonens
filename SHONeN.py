import argparse
import json

from BatchGenerator import batch_generator
from Classifiers import shonen_net
from Classifiers import hyper_groups
from DataGenerator import generator
from utils import utils

import networkx as nx
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, default='email-enron',
    help='Name of the dataset. Possible choices: email-enron, '+
         'contact-primary-school, NDC, DBLP, math-sx, contact-high-school, ' +
         'MAG-Geo')
parser.add_argument(
    '--ratio', type=int, default=5,
    help='Number of negative samples for each positive sample in test set')
parser.add_argument(
    '--max-epoch', type=int, default=50,
    help='Total number of epochs to train for.')
parser.add_argument(
    '--max-subset-size', type=int, default=None,
    help='Maximum size of subsets to be considered for subgroups')

args = parser.parse_args()

hyperedges, hyperedges_timestamps, hypergraph_nodes = utils.read_benson_dataset(
    args.dataset)

hyperedges_to_timestamp = utils.associate_min_timestamp_with_hyperedges(
    hyperedges, hyperedges_timestamps)

ground_edges, train_edges, test_edges = generator.get_ground_train_test_split(
    args.dataset, hyperedges_to_timestamp, (0.6, 0.8))

hyperedges = {frozenset(hedge) for hedge in hyperedges}

neg_samples_size = args.ratio * len(train_edges)

train_negatives = generator.generate_negative_samples_for_hyperedges(
    ground_edges, hyperedges, neg_samples_size)
train_data = train_edges + train_negatives
train_labels = [1 for _ in range(len(train_edges))] + [
    0 for _ in range(len(train_negatives))]

train_subset_scores = hyper_groups.get_hyperedge_subset_importance(
    train_data, ground_edges, max_subset_size=args.max_subset_size)
train_subgroups = hyper_groups.get_central_subgroups_from_importance_scores(
    train_data, train_subset_scores)

neg_samples_size = args.ratio * len(test_edges)
test_negatives = generator.generate_negative_samples_for_hyperedges(
    ground_edges + train_edges, hyperedges, neg_samples_size)
test_data = test_edges + test_negatives
test_labels = [1 for _ in range(len(test_edges))] + [
    0 for _ in range(len(test_negatives))]

test_subset_scores = hyper_groups.get_hyperedge_subset_importance(
    test_data, ground_edges + train_edges,
    max_subset_size=args.max_subset_size)
test_subgroups = hyper_groups.get_central_subgroups_from_importance_scores(
    test_data, test_subset_scores)

mask_node = len(hypergraph_nodes)
batch_gen = batch_generator.HyperedgeGroupBatchGenerator(
    train_data, train_subgroups, train_labels, 64, mask_node)

test_gen = batch_generator.HyperedgeGroupBatchGenerator(
    test_data, test_subgroups, test_labels, 64,
    mask_node, test_generator=True)

Htrain = utils.get_hypermatrix(ground_edges, len(hypergraph_nodes))
Gtrain = (Htrain * Htrain.T).toarray()
train_graph = nx.from_numpy_array(Gtrain)
    
node2vec_params = {
    'graph': train_graph,
    'walk_length': 40,
    'num_walks': 10,
}

model = shonen_net.SHONeN(
    len(hypergraph_nodes) + 1, 64, initial_embedding_param=node2vec_params)
model.learn_node_embeddings(10, 40)

model = model.to(utils.get_device())
model.trainer(batch_gen, test_gen, args.max_epoch)

model.save_report('%s' % (args.dataset))
model.save_best_model(args.dataset)
