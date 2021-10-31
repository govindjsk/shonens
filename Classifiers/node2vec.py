from Classifiers.custom_layers import HyperedgePoolingLayer
from Classifiers.base_model import BaseModel
from utils import utils, exceptions

import numpy as np
from node2vec import Node2Vec
import torch
from tqdm import tqdm

# pylint: disable=no-member

class Node2VecHyperlinkPrediction(BaseModel):
    """A node2vec based model that learns to predict existence
    of a hyperlink using node embeddings.
    """
    def __init__(
            self, graph, dimensions, walk_length, num_walks,
            aggregate_method='min-pairwise',
            link_pred_method='hadamard-product', dropout=0.5):
        """Initialize node2vec model.

        Args:
            graph: np.array. Input graph on which node embeddings will be
                learned.
            dimensions: int. The dimensions of output node embeddings.
            walk_length: int. Length of the biased random walk to be executed.
            num_walks: int. No. of random walks per node.
            aggregate_method: str. The method used for aggregating node
                embeddings. It can be either 'max-pool', 'mean-pool',
                'sag-pool', 'min-pairwise' or 'mean-pairwise'.
            link_pred_method: str. The method used for link prediciton from 
                node embeddings. It can be either 'cosine', 'addition',
                'hadamard-product', 'l1-weighted', 'l2-weighted'. Required only
                if 'mean-pairwise' or 'min-pairwise' is used as
                aggregate method.
            dropout: float. Dropout probability.
        """
        super(Node2VecHyperlinkPrediction, self).__init__()
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.node2vec_model = Node2Vec(
            graph, dimensions=dimensions, walk_length=walk_length,
            num_walks=num_walks, workers=4)
        self.has_learned_node_embeddings = False
        self.aggregate_method = aggregate_method
        self.link_pred_method = link_pred_method
        self.hyperedge_embedder = HyperedgePoolingLayer(
            self.dimensions, aggregate_method=self.aggregate_method,
            link_pred_method=self.link_pred_method)
        self.fc = torch.nn.Linear(self.dimensions, self.dimensions)
        self.classif = torch.nn.Linear(self.dimensions, 1)
        self.loss = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(self.parameters())
        self.node_embeddings = None
        self.dropout = dropout

    def learn_node_embeddings(self, window_size, batch_words):
        """Learns node embeddings by treating each random walk as a string
        and learning word embeddings of the words (nodes) in random walk.

        Args:
            window_size: int. Window size of the word2vec model. Essentially
                how big context should be taken into account to learn
                node embeddings.
            batch_words: int. Batch size to learn embeddings.
        """
        model = self.node2vec_model.fit(
            window=window_size, batch_words=batch_words)
        self.node_embeddings = np.zeros(
            shape=(len(self.graph.nodes), self.dimensions))
        for node in self.graph.nodes:
            self.node_embeddings[node] = model.wv[str(node)]
        self.node_embeddings = torch.Tensor(self.node_embeddings).to(
            utils.get_device())
        self.has_learned_node_embeddings = True
    
    def forward(self, hyperedges):
        """Forward pass for hyperedge prediction.
        
        Args:
            hyperedges: list(torch.Tensor). Hyperedges for which prediction has
                to happen.
            
        Returns:
            torch.Tensor. Predicted output.
        """
        if 'pairwise' in self.aggregate_method:
            x = self.fc(self.node_embeddings)
            preds = self.hyperedge_embedder(x, hyperedges)
        else:
            x = self.hyperedge_embedder(self.node_embeddings, hyperedges)
            x = torch.dropout(x, self.dropout, train=self.training)
            x = torch.relu(self.fc(x))
            preds = torch.sigmoid(self.classif(x))
        return preds

    def trainer(self, batch_gen, test_gen, max_epoch=10):
        """Trains Hyperlink prediction model on top of node2vec node embeddings.

        Args:
            batch_gen: BatchGenerator. Batch generator object that generates
                a batch of hyperedges and corresponding labels for
                training data.
            batch_gen: BatchGenerator. Batch generator object that generates
                a batch of hyperedges and corresponding labels for test data.
            max_epoch: int. Maximum number of epochs to train for.
        
        Raises:
            NodeEmbeddingsNotFoundError: An exception is raised if node2vec
                model is not yet trained.
        """
        if not self.has_learned_node_embeddings:
            raise exceptions.NodeEmbeddingsNotFoundError
        epoch_bar = tqdm(total=max_epoch)
        batch_bar = tqdm(
            total=batch_gen.total_size // batch_gen.batch_size, leave=False,
            desc='Iterator over batches.')
        epoch_count = 0
        epoch_loss = []
        epoch_preds = []
        epoch_labels = []
        while epoch_count < max_epoch:
            hyperedges, labels, epoch_bool = batch_gen.next()
            self.optim.zero_grad()
            preds = self.forward(hyperedges)
            loss = self.loss(preds.squeeze(), labels)
            loss.backward()
            self.optim.step()
            batch_bar.update()
            epoch_loss.append(loss.detach().item())
            epoch_preds.append(preds.detach())
            epoch_labels.append(labels)
            if epoch_bool:
                epoch_count += 1
                # Train summary.
                y_preds = torch.cat(epoch_preds)
                y_true = torch.cat(epoch_labels)
                report = self.get_report(y_true, y_preds)
                self.print_report(report, epoch_count, np.mean(epoch_loss))

                # Test summary.
                if test_gen:
                    y_preds = self.predict(test_gen)
                    y_true = test_gen.get_labels()
                    test_report = self.get_report(y_true, y_preds)
                    test_loss = self.loss(y_preds, y_true).detach().item()
                    self.print_report(
                        test_report, epoch_count, test_loss, train=False)
                    self.add_summary(
                        epoch_count, report, np.mean(epoch_loss),
                        test_report, test_loss)
                    self.track_best_model(test_report['ROC'])
                else:
                    self.add_summary(epoch_count, report, np.mean(epoch_loss))
                    self.track_best_model(report['ROC'])

                epoch_bar.update()
                batch_bar.close()
                batch_bar = tqdm(
                    total=batch_gen.total_size // batch_gen.batch_size,
                    leave=False, desc='Iterator over batches.')
                epoch_loss = []
                epoch_preds = []
                epoch_labels = []
        
        batch_bar.close()
        epoch_bar.close()

    def predict(self, test_gen):
        """Predict function that predicts output for test hyperedges.

        Args:
            test_gen: BatchGenerator. A batch generator that generates batches
                of test data (only inputs).
        
        Returns:
            torch.Tensor. Output prediction for test hyperedges.
        """ 
        self.eval()
        is_last_batch = False
        test_iterator = tqdm(
            total=test_gen.total_size // test_gen.batch_size, leave=False,
            desc='Iterator test over batches.')
        predictions = []
        while not is_last_batch:
            hyperedges, is_last_batch = test_gen.next()
            preds = self.forward(hyperedges)
            predictions.append(preds.squeeze().detach())
            test_iterator.update()
        predictions = torch.cat(predictions).squeeze()
        self.train()
        return predictions
