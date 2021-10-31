import math

from Classifiers.custom_layers import FBetaLoss, HyperedgePoolingLayer
from Classifiers.base_model import BaseModel

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# pylint: disable=no-member

class HGNNConvolution(torch.nn.Module):
    """Hypergraph convolution layer."""
    def __init__(self, in_ft, out_ft, bias=True):
        """Creates an instance of hypergraph convolution layer.

        Args:
            in_ft: int. Input features size.
            out_ft: int. Output features size.
        """
        super(HGNNConvolution, self).__init__()

        self.weight = torch.nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Resets parameters."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        """Forward pass of Hypergraph convolution layer.

        Args:
            x: torch.Tensor. Features of nodes.
            G: torch.Tensor. Forward propogation adjancency matrix as defined
                for hypergraph convolution layer.
        
        Returns:
            torch.Tensor. Feature representations of next layer.
        """
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HGNNHyperlinkPrediction(BaseModel):
    """HGNNHyperlinkPrediction model adapts HGNN architecture for the purpose
    of link prediction.

    The architecture is similar to HGNN, however, instead of predicting class
    labels, the model predicts whether a particular set of nodes will form
    a hyperedge.
    """
    def __init__(
            self, in_ch, n_hid, aggregate_method='max-pool',
            link_pred_method='hadamard-product', dropout=0.5):
        """Creates an instance of HGNNHyperlinkPrediction classs.

        Args:
            in_ch: int. Input channels.
            n_hid: int. No of hidden activation units.
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
        super(HGNNHyperlinkPrediction, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNNConvolution(in_ch, n_hid)
        self.hgc2 = HGNNConvolution(n_hid, n_hid)
        self.hedge_embedder = HyperedgePoolingLayer(
            n_hid, aggregate_method, link_pred_method)
        self.fc = torch.nn.Linear(n_hid, 1)
        self.loss = torch.nn.BCELoss()
        # self.loss = FBetaLoss(0.1)
        self.optim = torch.optim.Adam(self.parameters())
        self.aggregation_mathod = aggregate_method
        self.link_pred_method = link_pred_method

    def forward(self, x, hyperedges, G):
        """Forward pass of HGNN based Hyperedge predictor.

        Args:
            x: torch.Tensor. Embeddings of nodes.
            hyperedges: list(torch.Tensor). Hyperedges for which prediction has
                to happen.
            G: torch.Tensor. Adjacency matrix of hypergraph.
        
        Returns:
            torch.Tensor. Output prediction for hyperedges.
        """
        x = torch.relu(self.hgc1(x, G))
        x = torch.dropout(x, self.dropout, train=self.training)
        x = torch.relu(self.hgc2(x, G))
        x = torch.dropout(x, self.dropout, train=self.training)
        if 'pairwise' in self.aggregation_mathod:
            preds = self.hedge_embedder(x, hyperedges)
        else:
            x = self.hedge_embedder(x, hyperedges)
            preds = torch.sigmoid(self.fc(x))
        return preds

    def trainer(self, initial_embeddings, batch_gen, test_gen, G, Gtest, max_epoch=10):
        """Trains the model for given number of epochs.

        Args:
            initial_embeddings: torch.Tensor. The initial node embeddings. Note
                that the size of initial embeddings must match input channels
                as intialized in init function.
            batch_gen: BatchGenerator. Batch generator object that generates
                a batch of hyperedges and corresponding labels for train data.
            test_gen: BatchGenerator. Batch generator object that generates
                a batch of hyperedges and corresponding labels for test data.
            G: torch.Tensor. Adjacency matrix for HGNN convolution as defined
                by HGNN convolution operation.
            Gtest: torch.Tensor. Adjacency matrix for HGNN convolution as defined
                by HGNN convolution operation for test data.
            max_epoch: int. Number of epochs to train for.
        """
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
            preds = self.forward(initial_embeddings, hyperedges, G)
            loss = self.loss(preds.squeeze(), labels)
            loss.backward()
            self.optim.step()
            batch_bar.update()
            epoch_loss.append(loss.detach().item())
            epoch_preds.append(preds.detach())
            epoch_labels.append(labels)
            if epoch_bool:
                epoch_count += 1
                y_preds = torch.cat(epoch_preds)
                y_true = torch.cat(epoch_labels)
                report = self.get_report(y_true, y_preds)
                self.print_report(report, epoch_count, np.mean(epoch_loss))

                if test_gen:
                    y_preds = self.predict(initial_embeddings, test_gen, Gtest)
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

    def predict(self, initial_embeddings, test_gen, G):
        """Predict function that predicts output for test hyperedges.

        Args:
            initial_embeddings: torch.Tensor. Initial embeddings of nodes.
            test_gen: BatchGenerator. A batch generator that generates batches
                of test data (only inputs).
            G: torch.Tensor. Adjacency matrix for HGNN convolution as defined
                by HGNN convolution operation.
        
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
            preds = self.forward(initial_embeddings, hyperedges, G)
            predictions.append(preds.squeeze().detach())
            test_iterator.update()
        predictions = torch.cat(predictions)
        self.train()
        return predictions
    
    def generate_laplacian_matrix_from_hypermatrix(self, H):
        """Generates adjacency matrix for HGNN convolution as defined
        by HGNN convolution operation.

        Args:
            H: np.array. Hypergraph incidence matrix.

        Returns:
            np.array. The generated adjacency matrix.
        """
        H = H.toarray()
        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = np.ones(n_edge)
        # the degree of the node
        DV = np.sum(H * W, axis=1)
        
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)
        invDE = np.mat(np.diag(np.power(DE, -1.0)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        
        # There could be some nodes which are not part of any of the train edges
        # but only appear at test edges.
        DV2[np.isinf(DV2)] = 0
        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        G = DV2 * H * W * invDE * HT * DV2
        return torch.Tensor(G)
