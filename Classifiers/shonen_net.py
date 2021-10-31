from Classifiers.custom_layers import MultiheadSelfAttention
from Classifiers.base_model import BaseModel
from utils import utils, exceptions

from node2vec import Node2Vec
import numpy as np
import torch
from tqdm import tqdm

# pylint: disable=no-member

class HyperedgeAggregator(torch.nn.Module):
    """Aggregates node embeddings into hyperedge embeddings through
    hyperedge subsets."""
    def __init__(self, dimensions, attention_aggregator, dropout=0.5):
        """Creates an instance of HyperedgeAggregator layer.

        Args:
            dimensions: int. Dimension of node embedding.
            attention_aggregator: bool. Whether to use attention based
                aggregator for hyperedge embedding or averaging method.
        """
        super(HyperedgeAggregator, self).__init__()
        self.dimensions = dimensions
        self.linear = torch.nn.Linear(self.dimensions, self.dimensions)
        self.attention_aggregator = attention_aggregator
        if self.attention_aggregator:
            self.hyperedge_attention = MultiheadSelfAttention(dimensions, 4)
        self.dropout = dropout
    
    def unify_nodes(self, embeddings):
        return embeddings.mean(dim=1).squeeze()

    def unify_groups(self, group_embeddings):
        if self.attention_aggregator:
            hedmbedding = torch.dropout(
                self.hyperedge_attention(group_embeddings), self.dropout,
                train=self.training)
            return hedmbedding.mean(dim=0).squeeze()
        else:
            if group_embeddings.dim() < 2:
                group_embeddings = group_embeddings.unsqueeze(dim=0)
            return group_embeddings.mean(dim=0).squeeze()

    def forward(self, node_embeddings, hyperedges, hyperedge_subsets):
        """Forward pass of hyperedge embedding layer.
        
        Args:
            node_embeddings. torch.Tensor. Node embeddings of nodes of
                hypergraph.            
            hyperedges: List(torch.Tensor). Hyperedges.
            hyperedge_subsets: List(List(torch.Tensor)). Hyperedge subgroup
                partitions of each hyperedge.
        """
        node_embeddings = torch.relu(
            self.linear(node_embeddings)).unsqueeze(dim=0)
        hyperedge_embeddings = []
        for subgroups in hyperedge_subsets:
            group_embeddings = self.unify_nodes(
                node_embeddings[0, subgroups, :]).squeeze()
            hyperedge_embedding = self.unify_groups(group_embeddings)
            hyperedge_embeddings.append(hyperedge_embedding)
        hyperedge_embeddings = torch.stack(hyperedge_embeddings)
        return hyperedge_embeddings


class NodeDistributor(torch.nn.Module):
    """Generates node embeddings by distributing hyperedge embeddings on
    nodes."""
    def __init__(self, dimensions, dropout=0.5):
        """Creates an instance of NodeDistributor layer.

        Args:
            dimensions: int. Dimension of node / hyperedge embeddings.
        """
        super(NodeDistributor, self).__init__()
        self.dimensions = dimensions
        self.dropout = dropout
        self.linear = torch.nn.Linear(self.dimensions, self.dimensions)
    
    def forward(self, hyperedge_embeddings, H):
        """Forward pass of NodeDistributor layer.

        Args:
            hyperedge_embeddings: torch.Tensor. Embeddings of hyperedges.
            H: torch.Tensor. Hypergraph incidence matrix. Should be row
                normalized.
        
        Returns:
            torch.Tensor. Node embeddings.
        """
        hyperedge_embeddings = torch.dropout(
            torch.relu(self.linear(hyperedge_embeddings)), self.dropout,
            train=self.training)
        return torch.mm(H, hyperedge_embeddings)


class SHONeN(BaseModel):
    """SHONeN architecture."""
    def __init__(
            self, n_input, dimensions, initial_embedding_param):
        """Creates NeuralHyperNetwork instance.
        
        Args:
            n_input: int. Input dimension.
            dimensions: int. Dimension of latent node embeddings.
            initial_embedding_param: dict|None. Parameters specific to initial
                embeddings if initial_embeddings is not None.
        """
        super(SHONeN, self).__init__()
        self.dimensions = dimensions
        self.hedge1 = HyperedgeAggregator(self.dimensions)
        self.dnode1 = NodeDistributor(self.dimensions)
        self.hedge2 = HyperedgeAggregator(
            self.dimensions, attention_aggregator=False)
        self.classif = torch.nn.Linear(2 * self.dimensions, self.dimensions)
        self.classif2 = torch.nn.Linear(self.dimensions, 1)
        self.loss = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(self.parameters())

        self.node2vec_model = Node2Vec(
            initial_embedding_param['graph'], dimensions=dimensions,
            walk_length= initial_embedding_param['walk_length'],
            num_walks=initial_embedding_param['num_walks'], workers=4)
        self.has_learned_node_embeddings = False
        self.initial_embedding_param = initial_embedding_param

    def _reset_parameters(self):
        """Reinitializes manually created torch parameters."""
        torch.nn.init.xavier_normal_(self.initial_embeddings)

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
        node_embeddings = np.zeros(shape=(
            len(self.initial_embedding_param['graph'].nodes), self.dimensions))
        for node in self.initial_embedding_param['graph'].nodes:
            node_embeddings[node] = model.wv[str(node)]
        node_embeddings = torch.Tensor(node_embeddings)
        mask_embedding = torch.zeros(self.dimensions).unsqueeze(dim=0)
        self.initial_embeddings = torch.cat(
            [node_embeddings, mask_embedding], dim=0).to(utils.get_device())
        self.has_learned_node_embeddings = True

    def forward(self, hyperedges, hyperedge_subsets, H):
        """Forward pass of NeuralHyperNetwork.

        Args:
            hyperedges: List(torch.LongTensor). Hyperedges.
            hyperedge_subsets: List(List(torch.LongTensor)) subsets of
                hyperedges.
        
        Returns:
            torch.Tensor. Probability of hyperedge prediction.
        """
        n1 = self.initial_embeddings
        e1 = self.hedge1(n1, hyperedges, hyperedge_subsets)
        n2 = self.dnode1(e1, H)
        e2 = self.hedge2(n2, hyperedges, hyperedge_subsets)
        x = torch.cat([e1, e2], dim=1)
        x = torch.relu(self.classif(x))
        preds = torch.sigmoid(self.classif2(x)).squeeze()
        return preds
    
    def trainer(self, batch_gen, test_gen, max_epoch=10):
        """Trains Hyperlink prediction model.

        Args:
            batch_gen: BatchGenerator. Batch generator object that generates
                a batch of hyperedges and corresponding labels.
            test_gen: BatchGenerator. A batch generator that generates batches
                of test data (only inputs).
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
            hyperedges, hyperedge_subsets, H, labels, epoch_bool = batch_gen.next()
            self.optim.zero_grad()
            preds = self.forward(hyperedges, hyperedge_subsets, H)
            loss = self.loss(preds, labels)
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
            hyperedges, hyperedge_subsets, H, is_last_batch = test_gen.next()
            preds = self.forward(hyperedges, hyperedge_subsets, H)
            predictions.append(preds.squeeze().detach())
            test_iterator.update()
        predictions = torch.cat(predictions)
        self.train()
        return predictions
    
