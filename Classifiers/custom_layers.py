from utils import utils, exceptions

import numpy as np
import torch
from tqdm import tqdm

# pylint: disable=no-member

class MultiheadSelfAttention(torch.nn.Module):
    """Multi Head Self Attention LAyer."""
    def __init__(self, dimensions, n_heads, self_loop=False):
        """Initialize MultiHeadAttention Layer.
        
        Args:
            dimensions: int. Dimensions of node embedding.
            n_heads: int. Number of attention heads.
            self_loop: bool. Whether to mask self_loops or not.
        """
        super(MultiheadSelfAttention, self).__init__()
        self.dimensions = dimensions
        assert n_heads > 1, "No. of attention heads must be more than 1"
        assert dimensions % n_heads == 0, (
            "dimensions of embedding must be multiple of n_heads")
        self.n_heads = n_heads
        self.Wq = torch.nn.Parameter(
            torch.Tensor(self.dimensions, self.dimensions).to(
                utils.get_device()))
        self.Wk = torch.nn.Parameter(
            torch.Tensor(self.dimensions, self.dimensions).to(
                utils.get_device()))
        self.Wv = torch.nn.Parameter(
            torch.Tensor(self.dimensions, self.dimensions).to(
                utils.get_device()))
        self._reset_parameters()
        self.self_loop = self_loop
    
    def _reset_parameters(self):
        """Re-initialize parameters."""
        torch.nn.init.xavier_normal_(self.Wq)
        torch.nn.init.xavier_normal_(self.Wk)
        torch.nn.init.xavier_normal_(self.Wv)
    
    def forward(self, node_embeddings):
        """Forward pass of dynamic embedding layer.

        This is same as the self-attention as described by Vaswani et al (2017).
        
        Args:
            node_embeddings: torch.Tensor. Node embeddings of nodes of a
                hyperedge.
        
        Returns:
            torch.Tensor. Dynamic embedded node embeddings.
        """
        if node_embeddings.dim() < 2:
            node_embeddings = node_embeddings.unsqueeze(dim=0)

        q_emb = torch.mm(node_embeddings, self.Wq)
        k_emb = torch.mm(node_embeddings, self.Wk)
        v_emb = torch.mm(node_embeddings, self.Wv)
        q_emb = q_emb.view(
            -1, self.dimensions // self.n_heads, self.n_heads).permute(2, 0, 1)
        k_emb = k_emb.view(
            -1, self.dimensions // self.n_heads, self.n_heads).permute(2, 1, 0)
        
        sim = torch.matmul(q_emb, k_emb) / np.sqrt(
            self.dimensions // self.n_heads)
        
        # Mask self loops if necessary.
        if self.self_loop:
            mask = torch.diag(torch.ones(sim.size()[1])).bool()
            mask = mask.unsqueeze(dim=0).repeat((self.n_heads, 1, 1))
            sim[mask] = float('-inf')

        v_emb = v_emb.view(
            -1, self.dimensions // self.n_heads, self.n_heads).permute(2, 0, 1)
        out_emb = torch.tanh(
            torch.matmul(torch.softmax(sim, dim=2), v_emb).permute(
                1, 2, 0).reshape(-1, self.dimensions))
        return out_emb


class PairwiseLinkPredictionLayer(torch.nn.Module):
    """Predicts pairwise links between nodes of a hyperedge."""
    def __init__(self, dimensions, link_pred_method):
        """Initialize pairwise link prediction layer.

        Args:
            dimensions: int. Dimensions of node embedding.
            link_pred_method: str. The method used for link prediciton from 
                node embeddings. It can be either 'cosine', 'addition',
                'hadamard-product', 'l1-weighted', 'l2-weighted'.
        """
        super(PairwiseLinkPredictionLayer, self).__init__()
        self.dimensions = dimensions
        self.link_pred_method = link_pred_method
        if self.link_pred_method in [
            'addition', 'hadamard-product', 'l1-weighted', 'l2-weighted']:
            self.classif = torch.nn.Linear(self.dimensions, 1)

    def cosine_link_prediction(self, u_embeddings, v_embeddings):
        """Cosine similarity based link prediction method.

        Args:
            u_embeddings: torch.Tensor. Embeddings of nodes on one side of the
                link.
            v_embeddings: torch.Tensor. Embeddings of nodes on the other side of
                the link.
        
        Returns:
            torch.Tensor. Predicted probabilities of pairwise link predictions.
        """
        u_norm = torch.norm(u_embeddings, dim=1).detach()
        u_embeddings = u_embeddings / u_norm.unsqueeze(dim=1)

        v_norm = torch.norm(v_embeddings, dim=1).detach()
        v_embeddings = v_embeddings / v_norm.unsqueeze(dim=1)

        return (torch.sum(u_embeddings * v_embeddings, dim=1) + 1) / 2.0

    def forward(self, u_embeddings, v_embeddings):
        """Forward pass of pairwise link prediction layer.

        Args:
            u_embeddings: torch.Tensor. Embeddings of nodes on one side of the
                link.
            v_embeddings: torch.Tensor. Embeddings of nodes on the other side of
                the link.
        
        Returns:
            torch.Tensor. Prediction probabilities of links of the hyperedge.
        """
        if self.link_pred_method == 'cosine':
            return self.cosine_link_prediction(u_embeddings, v_embeddings)
        elif self.link_pred_method == 'hadamard-product':
            return torch.sigmoid(
                self.classif(u_embeddings * v_embeddings)).squeeze()
        elif self.link_pred_method == 'addition':
            return torch.sigmoid(
                self.classif(u_embeddings + v_embeddings)).squeeze()
        elif self.link_pred_method == 'l1-weighted':
            return torch.sigmoid(
                self.classif(torch.abs(u_embeddings - v_embeddings))).squeeze()
        elif self.link_pred_method == 'l2-weighted':
            return torch.sigmoid(self.classif(
                torch.pow(u_embeddings - v_embeddings, 2))).squeeze()
        else:
            raise exceptions.MethodNotImplementedError(
                self.link_pred_method, self.__class__.__name__)


class HyperedgePoolingLayer(torch.nn.Module):
    """A layer that pools node embeddings to form hyperedge embedding."""
    def __init__(
            self, dimensions, aggregate_method,
            link_pred_method='hadamard-product'):
        """Initialize hyperedge pooling layer.
        
        Args:
            dimensions: int. Dimension of node embeddings.
            aggregate_method: str. The method to use for pooling node embeddings
                to form hyperedge embeddings. It can be one of the following:
                    - max-pool
                    - mean-pool
                    - sag-pool
                    - min-pairwise
                    - mean-pairwise
            link_pred_method: str. The method used for link prediciton from 
                node embeddings. It can be either 'cosine', 'addition',
                'hadamard-product', 'l1-weighted', 'l2-weighted'. Required only
                if 'mean-pairwise' or 'min-pairwise' is used as
                aggregate method.
        """
        super(HyperedgePoolingLayer, self).__init__()
        self.dimensions = dimensions
        self.aggregate_method = aggregate_method
        self.link_pred_method = link_pred_method
        if self.aggregate_method == 'sag-pool':
            self.attn = MultiheadSelfAttention(self.dimensions, n_heads=4)
            self.v = torch.nn.Parameter(
                torch.Tensor(1, self.dimensions).to(utils.get_device()),
                requires_grad=True)
            self.reset_parameters()
        elif self.aggregate_method == 'evolve-pool':
            self.compress_layer = torch.nn.Linear(
                2 * self.dimensions, self.dimensions)
        elif 'pairwise' in self.aggregate_method:
            self.link_pred_layer = PairwiseLinkPredictionLayer(
                self.dimensions, self.link_pred_method)
        
    def reset_parameters(self):
        """Resets parameters."""
        torch.nn.init.xavier_uniform_(self.v)

    def max_pool(self, embeddings):
        """Return max-pool of given embeddings."""
        return torch.max(embeddings, dim=0)[0]
    
    def avg_pool(self, embeddings):
        """Return mean-pool of given embeddings."""
        return torch.mean(embeddings, dim=0)
    
    def sag_pool(self, embeddings):
        """Return sag-pool (self-attention-graph pooling) of given
        embeddings."""
        embeddings = self.attn(embeddings)
        return torch.mm(
            torch.softmax(torch.mm(self.v, embeddings.T), dim=1), embeddings)  

    def pairwise_link_pred_probs(self, embeddings):
        """Returns probability of hyperlink prediction as the mean / min of the
        probabilities of underlynig links between nodes.

        Args:
            embeddings: torch.Tensor. Embeddings of the nodes.
        """
        nodes = embeddings.size()[0]
        edges = [
            (node1, node2) for node1 in range(nodes) for node2 in range(nodes)
            if node1 < node2]
        u_nodes = torch.LongTensor([u for u, v in edges])
        v_nodes = torch.LongTensor([v for u, v in edges])
        u_embeddings = embeddings[u_nodes]
        v_embeddings = embeddings[v_nodes]
        preds = self.link_pred_layer(u_embeddings, v_embeddings)

        if self.aggregate_method == 'mean-pairwise':
            return preds.mean().squeeze()
        elif self.aggregate_method == 'min-pairwise':
            return preds.min().squeeze()
        else:
            raise exceptions.MethodNotImplementedError(
                self.aggregate_method, self.__class__.__name__)

    def agg_embeddings(self, embeddings):
        """Applies appropriate aggregation method."""
        if self.aggregate_method == 'max-pool':
            return self.max_pool(embeddings)
        elif self.aggregate_method == 'mean-pool':
            return self.avg_pool(embeddings)
        elif self.aggregate_method == 'sag-pool':
            return self.sag_pool(embeddings)
        else:
            raise exceptions.MethodNotImplementedError(
                self.aggregate_method, self.__class__.__name__)
    
    def forward(self, x, hyperedges):
        """Forward pass of classification layer.

        Args:
            x: torch.Tensor: Input features to classification layer.
            hyperedges: list(torch.LongTensor). Each member of the list is a
                hyperedge whose node indexes are store in torch.LongTensor.

        Returns:
            torch.Tensor. Embeddings of hyperedges.
        """
        if 'pairwise' in self.aggregate_method:
            hyperedge_probs = torch.zeros(len(hyperedges)).to(utils.get_device())
            for idx, hedge in enumerate(hyperedges):
                prob = self.pairwise_link_pred_probs(x[hedge])
                hyperedge_probs[idx] = prob
            return hyperedge_probs
        else:
            hyperedge_embeddings = torch.zeros(
                (len(hyperedges), self.dimensions)).to(utils.get_device())
            for idx, hedge in enumerate(hyperedges):
                embeddings = self.agg_embeddings(x[hedge]).squeeze()
                hyperedge_embeddings[idx, :] = embeddings
            return hyperedge_embeddings
