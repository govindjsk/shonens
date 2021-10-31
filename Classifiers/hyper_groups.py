from collections import defaultdict
import itertools
from functools import reduce

from tqdm import tqdm

def _get_subgroup_importance(
        hyperedge, node_to_hyperedges, max_subset_size=None, memory_opt=False):
    """Generates subgroup importance scores for a given hyperedge.

    Args:
        hyperedge: frozenset. Set of nodes incident on hyperedge.
        node_to_hyperedges: dict(set). A dictionary mapping nodes to hyperedges
            incident on nodes.
        max_subset_size: int|None. If None then it considers all subsets
            otherwise only given number of nodes in subset.
        memory_opt: bool. Whether to use more memory to optimize performance.
            This can also cause reduce in performance for small datasets.
    
    Returns:
        dict. A dictionary mapping subset of nodes to their importance scores.
    """
    hyperedge = list(hyperedge)
    importance_score = {}
    memory = {}
    max_size = max_subset_size or len(hyperedge)
    for subsetsize in range(1, max_size + 1):
        for subset in itertools.combinations(hyperedge, subsetsize):
            subnodes = list(subset)
            othernodes = [node for node in hyperedge if node not in subnodes]
            incident_hdges = (
                set() if not memory_opt
                else memory.get(frozenset(subnodes), set()))
            if not incident_hdges:
                incident_hedges = set(reduce(
                    lambda x, y: x.intersection(y),
                    [node_to_hyperedges[node] for node in subnodes]))
                if memory_opt:
                    memory[frozenset(subnodes)] = incident_hedges
            
            non_incident_hedges = set()
            for node in othernodes:
                incident_with_node = set() if not memory_opt else memory.get(
                    frozenset(subnodes + [node]), set())
                if not incident_with_node:
                    incident_with_node = (
                        incident_hedges.intersection(node_to_hyperedges[node]))
                    if memory_opt:
                        memory[frozenset(subnodes + [node])] = (
                            incident_with_node)
                non_incident_hedges.update(incident_with_node)

            importance_score[frozenset(subnodes)] = len(
                incident_hedges.difference(non_incident_hedges))
    
    total = sum(importance_score.values())
    importance_score = {
        k: v / float(total) if total > 0 else v
        for k, v in importance_score.items()}
    return importance_score

def get_hyperedge_subset_importance(
        eval_edges, ground_edges, max_subset_size=None):
    """Generates importance scores of subsets for each hyperedge in hyperedges.

    Args:
        eval_edges: List(frozenset). Subset scores are calculated for
            eval_edges.
        ground_edges: List(frozenset). Ground edges are sued for scores
            calculation of eval_edges.
        max_subset_size: int|None. If None then it considers all subsets
            otherwise only given number of nodes in subset.
    
    Returns:
        List(dict). Each dict maps a subgroup of nodes
            to their importance for a given hyperedge.
    """
    node_to_hyperedges = defaultdict(set)
    for hidx, hedge in enumerate(ground_edges):
        for node in hedge:
            node_to_hyperedges[node].add(hidx)
    
    importance_scores = []
    for hedge in tqdm(
            eval_edges, desc='Calculating Hyperedge Scores', leave=False):
        scores = _get_subgroup_importance(
            hedge, node_to_hyperedges, max_subset_size)
        importance_scores.append(scores)
    
    return importance_scores

def get_central_subgroups_from_importance_scores(
        hyperedges, subset_importance, threshold=0.7):
    """Returns central subgroups of a hyperedge based on subgroup scores.

    Args:
        hyperedges: List(frozenset). List of hyperedges.
        subset_importance: List(dict). List of subset_importance scores for
            each hyperedge. Dict maps subgroup to its score.
        threshold: float. Add subgroups until threshold amount of scores are
            covered or all nodes are covered. Both should be satisfied.
    
    Returns:
        List(list). Central subgroups for each hyperedge.
    """
    central_subgroups = []
    
    for hedge, scores in tqdm(
            zip(hyperedges, subset_importance),
            desc='Finding central subgroups', leave=False):
        groups = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        csubgroups = set()
        coverage = set()
        score = 0
        for grp, scr in groups:
            if threshold and score >= threshold and coverage == set(hedge):
                break
            elif not threshold and coverage == set(hedge):
                break
            coverage.update(set(grp))
            csubgroups.add(frozenset(grp))
            score += scr
        for node in hedge:
            csubgroups.add(frozenset({node}))
        central_subgroups.append(csubgroups)
    
    return central_subgroups
