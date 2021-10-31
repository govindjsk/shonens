import argparse

from BatchGenerator import batch_generator
from Classifiers import hgnn
from Classifiers import node2vec
from Classifiers import hyper_sagnn
from DataGenerator import generator
from utils import utils

import networkx as nx
import numpy as np
import scipy.io
import torch
from tqdm import tqdm

np.random.seed(1057)
torch.manual_seed(1057)

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
    '--model', type=str, default='HGNN',
    help='Model to train: HGNN. HyperSAGNN, Node2Vec')
parser.add_argument(
    '--max-epoch', type=int, default=50, help='Number of training epochs')

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

neg_samples_size = args.ratio * len(test_edges)
test_negatives = generator.generate_negative_samples_for_hyperedges(
    ground_edges + train_edges, hyperedges, neg_samples_size)

train_data = train_edges + train_negatives
train_labels = [1 for _ in range(len(train_edges))] + [
    0 for _ in range(len(train_negatives))]

test_data = test_edges + test_negatives
test_labels = [1 for _ in range(len(test_edges))] + [
    0 for _ in range(len(test_negatives))]

batch_gen = batch_generator.BatchGenerator(
    train_data, train_labels, batch_size=64)
test_gen = batch_generator.BatchGenerator(
    test_data, test_labels, batch_size=64, test_generator=True)

# Train Data: train_edges, train_negatives
# Test Data: test_edges, test_negatives
# Common: nodes, ground_edges
Htrain = utils.get_hypermatrix(ground_edges, len(hypergraph_nodes))
Htest = utils.get_hypermatrix(ground_edges + train_edges, len(hypergraph_nodes))
max_epoch = args.max_epoch

# Training on top of HGNN model.
if args.model == 'HGNN':
    model = hgnn.HGNNHyperlinkPrediction(
        len(hypergraph_nodes), 64, aggregate_method='sag-pool',
        link_pred_method='addition')
    Gtrain = model.generate_laplacian_matrix_from_hypermatrix(Htrain).to(
        utils.get_device())
    Gtest = model.generate_laplacian_matrix_from_hypermatrix(Htest).to(
        utils.get_device())
    initial_embeddings = torch.Tensor(np.diag(np.ones(len(hypergraph_nodes))))
    initial_embeddings = initial_embeddings.to(utils.get_device())
    model = model.to(utils.get_device())
    model.trainer(
        initial_embeddings, batch_gen, test_gen, Gtrain, Gtest, max_epoch)

elif args.model == 'Node2Vec':
    # Training on top of Node2vec model.
    Gtrain = (Htrain * Htrain.T).toarray()
    Gtest = (Htest * Htest.T).toarray()
    train_graph = nx.from_numpy_array(Gtrain)
    test_graph = nx.from_numpy_array(Gtest)
    model = node2vec.Node2VecHyperlinkPrediction(
        train_graph, 64, 40, 10, aggregate_method='sag-pool',
        link_pred_method='cosine')
    model.learn_node_embeddings(10, 40)
    model = model.to(utils.get_device())
    model.trainer(batch_gen, test_gen, max_epoch)

elif args.model == 'HyperSAGNN':
    # Training Hyper-SAGNN.
    Gtrain = (Htrain * Htrain.T).toarray()
    Gtest = (Htest * Htest.T).toarray()
    train_graph = nx.from_numpy_array(Gtrain)
    test_graph = nx.from_numpy_array(Gtest)
    model = hyper_sagnn.HyperSAGNN(train_graph, 64, 40, 10, num_heads=4)
    model.learn_node_embeddings(10, 40)
    model = model.to(utils.get_device())
    model.trainer(batch_gen, test_gen, max_epoch)

model.save_report('%s' % (args.dataset))
model.save_best_model(args.dataset)
