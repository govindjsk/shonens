#!/usr/bin/env python
# This file declares a set of custom exceptions.

class NodeNotFoundException(Exception):
    """An exception that is raised when a node is not found in the graph."""
    def __init__(self, node):
        super(NodeNotFoundException, self).__init__(
            "Node %d not found in the graph" % node)


class NodeEmbeddingsNotFoundError(Exception):
    """An exception that is rased when node embeddings are not trained for
    model."""
    def __init__(self):
        super(NodeEmbeddingsNotFoundError, self).__init__(
            'Node embeddings are not trained. Please execute ' +
            'learn_node_embeddings method before training the model.')


class MethodNotImplementedError(Exception):
    """An exception that is raised of a method is not implemented."""
    def __init__(self, method_name, cls_name):
        """Initializes exception.

        Args:
            method_name: str. Method which is not found.
            cls_name: str. Class name of the object.
        """
        super(MethodNotImplementedError, self).__init__(
            self, 'Method %s is not implemented for objects of the class %s' % (
                method_name, cls_name))


class BestModelNotTrackedError(Exception):
    """An exception that is rased when best model is not available and attempt
    is made to load it.
    """
    def __init__(self):
        super(BestModelNotTrackedError, self).__init__(
            'Attempt is made to load best model which has not been tracked')
