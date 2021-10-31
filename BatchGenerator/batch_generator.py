from utils import utils

import numpy as np
import torch

# pylint: disable=no-member

device = ('cuda' if torch.cuda.is_available() else 'cpu')

class BatchGenerator(object):
    def __init__(self, inputs, outputs, batch_size, test_generator=False):
        self.batch_size = batch_size
        self.inputs = inputs
        self.outputs = outputs
        self._cursor = 0
        self.test_generator = test_generator
        self.shuffle()
    
    def eval(self):
        self.test_generator = True

    def train(self):
        self.test_generator = False

    def shuffle(self):
        idcs = np.arange(len(self.inputs))
        np.random.shuffle(idcs)
        self.inputs = [self.inputs[i] for i in idcs]
        self.outputs = [self.outputs[i] for i in idcs]        

    @property
    def total_size(self):
        return len(self.inputs)

    def get_labels(self):
        return torch.Tensor(self.outputs).to(utils.get_device())

    def process_batch(self, batch_inputs, batch_outputs=None):
        batch_inputs = [
            torch.LongTensor(list(ip)).to(utils.get_device())
            for ip in batch_inputs]
        if batch_outputs:
            batch_outputs = torch.Tensor(batch_outputs).to(utils.get_device())
            return batch_inputs, batch_outputs
        else:
            return batch_inputs
    
    def __iter__(self):
        self._cursor = 0
        return self
    
    def next(self):
        return self.__next__()

    def __next__(self):
        if self.test_generator:
            return self.next_test_batch()
        else:
            return self.next_train_batch()
    
    def next_train_batch(self):
        ncursor = self._cursor+self.batch_size
        if ncursor >= len(self.inputs):
            batch_inputs = self.inputs[self._cursor:] + self.inputs[
                :ncursor - len(self.inputs)]
            batch_outputs = self.outputs[self._cursor:] + self.outputs[
                :ncursor - len(self.outputs)]
            self._cursor = ncursor - len(self.inputs)
            batch_inputs, batch_outputs = self.process_batch(
                batch_inputs, batch_outputs)
            return batch_inputs, batch_outputs, True
        batch_inputs = self.inputs[self._cursor:self._cursor + self.batch_size]
        batch_outputs = self.outputs[
            self._cursor:self._cursor + self.batch_size]
        self._cursor = ncursor % len(self.inputs)
        batch_inputs, batch_outputs = self.process_batch(
            batch_inputs, batch_outputs)
        return batch_inputs, batch_outputs, False

    def next_test_batch(self):
        ncursor = self._cursor+self.batch_size
        if ncursor >= len(self.inputs):
            batch_inputs = self.inputs[self._cursor:]
            batch_inputs = self.process_batch(batch_inputs)
            self._cursor = 0
            return batch_inputs, True
        batch_inputs = self.inputs[self._cursor:self._cursor + self.batch_size]
        self._cursor = ncursor % len(self.inputs)
        batch_input = self.process_batch(batch_inputs)
        return batch_input, False


class HyperedgeGroupBatchGenerator(object):
    """A custom batch generator for hyperedge group based data."""
    def __init__(
            self, hyperedges, hyperedge_subsets, labels, batch_size, mask,
            test_generator=False):
        """Creates an instance of HyperedgeGroupBatchGenerator.
        
        Args:
            n_nodes: int. Number of nodes in hypergraph.
            hyperedges: List(frozenset). List of hyperedges.
            hyperedge_subsets: List(List(sets)). List of subsets of hyperedges.
            labels: list. Labels of hyperedges.
            batch_size. int. Batch size of each batch.
            mask: int. ID of mask node which is used for masking purpose.
            test_generator: bool. Whether batch generator is test generator.
        """
        self.batch_size = batch_size
        self.H = utils.get_hypermatrix(hyperedges, n_nodes=mask+1)
        self.hyperedges = hyperedges
        self.hyperedge_subsets = hyperedge_subsets
        self.labels = labels
        self._cursor = 0
        self.test_generator = test_generator
        self.MASK = mask
        self.shuffle()
    
    def eval(self):
        self.test_generator = True

    def train(self):
        self.test_generator = False

    def shuffle(self):
        idcs = np.arange(len(self.hyperedges))
        np.random.shuffle(idcs)
        self.H = self.H[:, idcs]
        self.hyperedges = [self.hyperedges[i] for i in idcs]
        self.hyperedge_subsets = [self.hyperedge_subsets[i] for i in idcs]
        self.labels = [self.labels[i] for i in idcs]

    @property
    def total_size(self):
        return len(self.hyperedges)

    def get_labels(self):
        return torch.Tensor(self.labels).to(utils.get_device())

    def get_hypergraph_incidence_matrix(self, hyperedges):
        idcs = []
        vals = []
        for hidx, hedge in enumerate(hyperedges):
            for node in hedge:
                idcs.append([node, hidx])
                vals.append(1)
        idcs = torch.LongTensor(idcs)
        vals = torch.ByteTensor(vals)
        return torch.sparse.ByteTensor(idcs.t(), vals)

    def process_batch(self, hyperedges, hyperedge_subsets, Hb, batch_outputs=None):
        Hb = torch.Tensor(Hb).to(utils.get_device())
        hyperedges = [
            torch.LongTensor(list(ip)).to(utils.get_device())
            for ip in hyperedges]
        hsubs = []
        for subgroup in hyperedge_subsets:
            max_size = max(len(grp) for grp in subgroup)
            subgroup = [list(grp) for grp in subgroup]
            hsubs.append([
                [grp[i] if i < len(grp) else self.MASK for i in range(max_size)]
                for grp in subgroup])
        if batch_outputs:
            batch_outputs = torch.Tensor(batch_outputs).to(utils.get_device())
            return hyperedges, hsubs, Hb, batch_outputs
        else:
            return hyperedges, hsubs, Hb
    
    def __iter__(self):
        self._cursor = 0
        return self
    
    def next(self):
        return self.__next__()

    def __next__(self):
        if self.test_generator:
            return self.next_test_batch()
        else:
            return self.next_train_batch()
    
    def next_train_batch(self):
        ncursor = self._cursor+self.batch_size
        if ncursor >= len(self.hyperedges):
            hyperedges = self.hyperedges[self._cursor:] + self.hyperedges[
                :ncursor - len(self.hyperedges)]
            
            hyperedge_subsets = (
                self.hyperedge_subsets[self._cursor:] +
                self.hyperedge_subsets[:ncursor - len(self.hyperedge_subsets)])
            
            labels = self.labels[self._cursor:] + self.labels[
                :ncursor - len(self.labels)]
            
            Hb = np.concatenate(
                [self.H[:, self._cursor:].toarray(),
                 self.H[:, :ncursor - len(self.hyperedges)].toarray()],
                axis=1)
            
            self._cursor = ncursor - len(self.hyperedges)
            
            hyperedges, hyperedge_subsets, Hb, labels = self.process_batch(
                hyperedges, hyperedge_subsets, Hb, labels)
            return hyperedges, hyperedge_subsets, Hb, labels, True
        
        hyperedges = self.hyperedges[
            self._cursor:self._cursor + self.batch_size]
        
        hyperedge_subsets = self.hyperedge_subsets[
            self._cursor:self._cursor + self.batch_size]
        
        labels = self.labels[
            self._cursor:self._cursor + self.batch_size]
        
        Hb = self.H[:, self._cursor: self._cursor + self.batch_size].toarray()
        
        self._cursor = ncursor % len(self.hyperedges)
        
        hyperedges, hyperedge_subsets, Hb, labels = self.process_batch(
                hyperedges, hyperedge_subsets, Hb, labels)
        return hyperedges, hyperedge_subsets, Hb, labels, False

    def next_test_batch(self):
        ncursor = self._cursor+self.batch_size
        
        if ncursor >= len(self.hyperedges):
            hyperedges = self.hyperedges[self._cursor:]
            hyperedge_subsets = self.hyperedge_subsets[self._cursor:]

            Hb = self.H[:, self._cursor:].toarray()
            self._cursor = 0
            
            hyperedges, hyperedge_subsets, Hb = self.process_batch(
                hyperedges, hyperedge_subsets, Hb)
            return hyperedges, hyperedge_subsets, Hb, True
        
        hyperedges = self.hyperedges[
            self._cursor:self._cursor + self.batch_size]
        
        hyperedge_subsets = self.hyperedge_subsets[
            self._cursor:self._cursor + self.batch_size]
        
        Hb = self.H[:, self._cursor: self._cursor + self.batch_size].toarray()
        
        self._cursor = ncursor % len(self.hyperedges)
        hyperedges, hyperedge_subsets, Hb = self.process_batch(
                hyperedges, hyperedge_subsets, Hb)
        
        return hyperedges, hyperedge_subsets, Hb, False
