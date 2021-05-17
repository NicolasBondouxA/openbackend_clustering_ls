import json
import csv
import logging
import pickle
from abc import ABC, abstractmethod

import networkx as nx
import matplotlib
import numpy as np

matplotlib.use('agg')
import matplotlib.pyplot as plt

import random

class Instance(object):

    def __init__(self, exchanges):
        """

        :param exchanges: Dict[str, Dict[str, int]]; source -> target -> count
        """
        self.exchanges = exchanges

        self.indices = self._build_indices()
        self.sorted_edges = self._build_sorted_edges()
        self.solution = None

    def get_all_sources(self):
        return sorted(self.exchanges.keys())

    def get_all_targets(self):
        return sorted(set([target for source in self.exchanges for target in self.exchanges[source]]))

    def get_all_nodes(self):
        return sorted(set(self.get_all_sources() + self.get_all_targets()))

    def _build_indices(self):
        return {node: node_idx for node_idx, node in enumerate(self.get_all_nodes())}

    def _build_sorted_edges(self):
        edges = [(source, target, self.exchanges[source][target])
                 for source in self.exchanges for target in self.exchanges[source]]
        return sorted(edges, key=lambda x: x[2], reverse=True)

    def set_solution(self, solution):
        self.solution = solution
        self.compress_solution()

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    def __eq__(self, other):
        if isinstance(other, Instance):
            return self.exchanges == other.exchanges
        return NotImplemented

    def __hash__(self):
        return hash(self.exchanges)

    def get_leaf_nodes(self):
        """
        Leaf node: linked to one and only one other node from the graph.
        :return:
        """
        single_target = set()
        for source in self.exchanges:
            targets = set(self.exchanges[source].keys())
            if source in targets:
                targets.remove(source)
            if len(self.exchanges[source]) == 1:
                single_target.add(source)

    def retrieve_clusters(self):
        """
        Returns a view of the clustering solution of a dictionary with cluster index as key and the set of nodes from
        that cluster as value. Used to compare clustering solutions.

        :return:
        """

        clusters = {}
        if self.solution is None:
            return clusters
        for node in self.solution:
            if self.solution[node] not in clusters:
                clusters[self.solution[node]] = set()
            clusters[self.solution[node]].add(node)
        return clusters

    def _find_best_cluster_mapping(self, self_clusters, other_clusters):
        """
        Finds a mapping of cluster ids between 2 solutions. Returns a dict which keys correspond to the cluster indices
        of the other solution and the values to the cluster indices of the current solution. It can then be used to
        reassign cluster indices in the other solution so that it matches the current clustering indices.

        :param self_clusters: clusters from the current instance object, result of a call to retrieve_clusters.
        :param other_clusters: clusters from the other instance object, result of a call to retrieve_clusters.
        :return:
        """

        if not self_clusters or not other_clusters:
            return None
        best_matches = {}
        for self_cluster in self_clusters:
            best_matches[self_cluster] = {}
            for other_cluster in other_clusters:
                current_count = len(self_clusters[self_cluster] - other_clusters[other_cluster]) \
                                + len(other_clusters[other_cluster] - self_clusters[self_cluster])
                best_matches[self_cluster][other_cluster] = current_count

        def find_min_match():
            best_val = float('inf')
            best_match = None
            for self_cluster in best_matches:
                for other_cluster in best_matches[self_cluster]:
                    if best_matches[self_cluster][other_cluster] < best_val:
                        best_val = best_matches[self_cluster][other_cluster]
                        best_match = (self_cluster, other_cluster)
            return best_match

        mapping = {}
        while best_matches:
            self_cluster, other_cluster = find_min_match()
            best_matches.pop(self_cluster)
            for cluster in best_matches:
                if other_cluster in best_matches[cluster]:
                    best_matches[cluster].pop(other_cluster)
            mapping[other_cluster] = self_cluster

        # mapping = {best_matches[key]: key for key in best_matches}
        next_idx = len(self_clusters.keys())
        for cluster in other_clusters:
            if cluster not in mapping:
                mapping[cluster] = next_idx
                next_idx += 1

        return mapping

    def reassign_solution(self, mapping):
        """
        Change the assignment of cluster numbers.
        :param mapping: dict-like key is current cluster number, value is desired cluster value
        :return:
        """
        if self.solution is None:
            return
        temp_solution = {node: mapping[self.solution[node]] for node in self.solution}
        self.solution = temp_solution

    def compress_solution(self):
        """
        For a clustering with N clusters, uses the numbers from 0 to N-1 as cluster numbers. Used when the optimization
        algorithm assigns non consecutive numbers for clusters.
        :return:
        """
        self.reassign_solution(
            {cluster: cluster_idx for cluster_idx, cluster in enumerate(sorted(self.retrieve_clusters()))}
        )

    def compare_solutions(self, other):
        """
        Produces logs to help comparing 2 solutions
        :param other: other instance
        :return:
        """

        self_clusters = self.retrieve_clusters()
        other_clusters = other.retrieve_clusters()
        mapping = self._find_best_cluster_mapping(self_clusters, other_clusters)
        other.reassign_solution(mapping)
        other_clusters = other.retrieve_clusters()

        def find_cluster(clusters, node):
            for cluster in clusters:
                if node in clusters[cluster]:
                    return cluster

        for cluster in sorted(self_clusters):
            logging.info('self cluster {}: {}'.format(cluster, sorted(self_clusters[cluster])))
            logging.info('        - OBEs: {}'.format(sorted(self_clusters[cluster] - other_clusters[cluster])))
            for node in sorted(self_clusters[cluster] - other_clusters[cluster]):
                logging.info('              : {} to cluster {}'.format(node, find_cluster(other_clusters, node)))
            logging.info('        + OBEs: {}'.format(sorted(other_clusters[cluster] - self_clusters[cluster])))
            for node in sorted(other_clusters[cluster] - self_clusters[cluster]):
                logging.info('              : {} from cluster {}'.format(node, find_cluster(self_clusters, node)))
            logging.info('   = cluster {}: {}'.format(cluster, sorted(other_clusters[cluster])))

        for other_cluster in other_clusters:
            if other_cluster not in self_clusters:
                logging.info('new cluster: {}'.format(sorted(other_clusters[other_cluster])))
                for node in sorted(other_clusters[other_cluster]):
                    logging.info('           : {} from cluster {}'.format(node, find_cluster(self_clusters, node)))

    def print_kpis(self):

        if not self.solution:
            return

        # size of clusters
        cluster_sizes = {}
        for node in self.solution:
            if self.solution[node] not in cluster_sizes:
                cluster_sizes[self.solution[node]] = 1
            else:
                cluster_sizes[self.solution[node]] += 1

        # traffic between clusters
        inter_cluster_traffic = 0
        intra_cluster_traffic = 0
        for source in self.exchanges:
            for target in self.exchanges[source]:
                if self.solution[source] != self.solution[target]:
                    inter_cluster_traffic += self.exchanges[source][target]
                else:
                    intra_cluster_traffic += self.exchanges[source][target]

        # traffic between 2 given clusters
        inter_cluster_per_cluster_traffic = {}
        for source in self.exchanges:
            for target in self.exchanges[source]:
                source_cluster = self.solution[source]
                target_cluster = self.solution[target]
                if source_cluster not in inter_cluster_per_cluster_traffic:
                    inter_cluster_per_cluster_traffic[source_cluster] = {}
                if target_cluster not in inter_cluster_per_cluster_traffic[source_cluster]:
                    inter_cluster_per_cluster_traffic[source_cluster][target_cluster] = 0
                inter_cluster_per_cluster_traffic[source_cluster][target_cluster] += self.exchanges[source][target]

        logging.info('number of clusters: {}\n{}\n{}'.format(len(cluster_sizes),
                                                             ','.join(['cluster'] + [str(cluster) for cluster in sorted(cluster_sizes)]),
                                                             ','.join(['size'] + [str(cluster_sizes[cluster]) for cluster in sorted(cluster_sizes)])))

        logging.info('total traffic intra clusters: {}'.format(intra_cluster_traffic))
        logging.info('total traffic inter clusters: {}'.format(inter_cluster_traffic))

        traffic_matrix_str = ','.join(['Traffic'] + ['C' + str(cluster) for cluster in sorted(cluster_sizes)]) + '\n'
        for source_cluster in sorted(cluster_sizes):
            traffic_matrix_str += 'C' + str(source_cluster) + ','
            traffic_matrix_str += ','.join([str(inter_cluster_per_cluster_traffic[source_cluster][target_cluster])
                                            if source_cluster in inter_cluster_per_cluster_traffic
                                               and target_cluster in inter_cluster_per_cluster_traffic[source_cluster]
                                            else ''
                                            for target_cluster in sorted(cluster_sizes)])
            traffic_matrix_str += '\n'
        logging.info('traffic detail: \n{}'.format(traffic_matrix_str))

    def total_exchanges(self):
        return sum([self.exchanges[source][target] for source in self.exchanges for target in self.exchanges[source]])

    def log_edge_weights(self):
        logging.info('logging edge weights')
        for source in self.exchanges:
            for target in self.exchanges[source]:
                logging.info(','.join([source, target, str(self.exchanges[source][target])]))

    def _get_sorted_edge_weights(self, max_cluster_size=None):
        if max_cluster_size is None:
            max_cluster_size = len(self.get_all_nodes())
        return sorted([self.exchanges[source][target]
                       for source in self.exchanges
                       for target_idx, target in enumerate(self.exchanges[source]) if target_idx < max_cluster_size],
                      reverse=True)

    def _get_max_edges_in_clusters(self, max_cluster_size):
        nb_clusters_of_max_size = len(self.get_all_nodes()) // max_cluster_size
        remaining_cluster_size = len(self.get_all_nodes()) - nb_clusters_of_max_size * max_cluster_size
        return nb_clusters_of_max_size * max_cluster_size * (max_cluster_size - 1) \
               + remaining_cluster_size * (remaining_cluster_size - 1)

    def get_max_solution_value(self, max_cluster_size):
        """
        Returns an upper bound on the value of the optimal solution if a maximum cluster size is provided.
        :param max_cluster_size:
        :return:
        """
        max_edges = self._get_max_edges_in_clusters(max_cluster_size)
        max_value = sum(self._get_sorted_edge_weights(max_cluster_size)[:max_edges])
        logging.info('maximum number of edges in the solution: {}'.format(max_edges))
        logging.info('maximum value of the objective function: {}'.format(max_value))
        return max_value

    def get_max_cluster_value(self, max_cluster_size):
        """
        Returns an upper bound on the contribution of any cluster to the objective function, if a maximum cluster size is provided.
        :param max_cluster_size:
        :return:
        """
        max_value = sum(self._get_sorted_edge_weights(max_cluster_size)[:max_cluster_size * (max_cluster_size - 1)])
        logging.info('maximum contribution of any cluster: {}'.format(max_value))
        return max_value

    def get_unique_edge_weights_sorted(self):
        """
        Returns unique edge weights sorted by descending value.
        :return:
        """
        return sorted(set([self.exchanges[source][target]
                           for source in self.exchanges
                           for target in self.exchanges[source]]),
                      reverse=True)

    def write_matrix(self,f):
        import sys
        for idx,source in enumerate(self.get_all_nodes()):
            sys.stderr.write('{}: node_{}\n'.format(source,str(idx)))
            
            is_first = True
            if source in self.exchanges:
                exchanges_s = self.exchanges[source]
            else:
                exchanges_s = None

            for target in self.get_all_nodes():
                if not is_first:
                    f.write(',')
                else:
                    is_first = False
                if exchanges_s == None or target not in exchanges_s:
                    f.write('X')
                else:
                    f.write(str(exchanges_s[target]))
            f.write('\n')



class AbstractInstanceParser(ABC):

    def __init__(self, path):
        self.path = path
        self.exchanges = self._parse_file()

    @abstractmethod
    def _parse_file(self):
        return NotImplemented

    def _filter_exchanges(self, ignore, kept_nodes, min_edge_weight):
        filtered_exchanges = {}
        for source in self.exchanges:
            if kept_nodes is not None:
                if source not in kept_nodes:
                    continue
            if source in ignore:
                continue
            for target in self.exchanges[source]:
                if kept_nodes is not None and target not in kept_nodes:
                    continue
                if target in ignore or self.exchanges[source][target] < min_edge_weight:
                    continue
                if source not in filtered_exchanges:
                    filtered_exchanges[source] = {}
                filtered_exchanges[source][target] = self.exchanges[source][target]

        all_nodes = set()
        for source in self.exchanges:
            all_nodes.add(source)
            all_nodes.update(self.exchanges[source])

        kept_nodes = set()
        for source in filtered_exchanges:
            kept_nodes.add(source)
            kept_nodes.update(filtered_exchanges[source])

        logging.info('all nodes: {}'.format(sorted(all_nodes)))
        logging.info('kept nodes: {}'.format(sorted(kept_nodes)))
        logging.info('nodes filtered out: {}'.format(sorted(all_nodes - kept_nodes)))
        return filtered_exchanges

    def create_instance(self, ignore=None,  kept_nodes=None, min_edge_weight=1):
        filtered_exchanges = self._filter_exchanges(ignore if ignore is not None else [], kept_nodes, min_edge_weight)
        return Instance(filtered_exchanges)


class InstanceMatrixParser(AbstractInstanceParser):
    """
    Class implementing a parser to process the toy examples in resources/matrix_instance_*.
    """
    def __init__(self, path, use_perturbation=False):
        self.use_perturbation = use_perturbation
        super().__init__(path)


    def _parse_file(self):
        with open(self.path) as f:
            exchanges = {}
            for line_idx, line in enumerate(f.readlines()):
                splitted = line.rstrip().split(',')
                for elem_idx, elem in enumerate(splitted):
                    if elem == 'X' or elem == '0':
                        continue
                    source_id = 'node_{}'.format(line_idx)
                    target_id = 'node_{}'.format(elem_idx)
                    if source_id not in exchanges:
                        exchanges[source_id] = {}
                    if target_id not in exchanges[source_id]:
                        exchanges[source_id][target_id] = 0
                    elem = float(elem)
                    if self.use_perturbation:
                        elem = (float(10*elem)+random.random()) / 10.
                    exchanges[source_id][target_id] += elem
            return exchanges


class InstancePickleParser(AbstractInstanceParser):
    """
    Class implementing a parser to load pickled instances.
    """

    def _parse_file(self):
        return pickle.load(open(self.path, "rb"))

    def create_instance(self, ignore=None, kept_nodes=None, min_edge_weight=1):
        return self.exchanges


def circular_position(graph, graph_rad, cluster_rad):
    """
    Assign a position to each node with the following process:
    1. assign a center for each cluster, positioned around a circle of radius graph_rad
    2. assign a position for each node, positioned around a circle of radius cluster_rad and centered at the center of
    the cluster fixed in step 1
    :param graph:
    :param graph_rad: radius of the circle formed by the centers of the different clusters
    :param cluster_rad: radius of the circle formed by a single cluster
    :return:
    """
    # prep center points (along circle perimeter) for the clusters
    clusters = sorted(set([graph.nodes()[node]['cluster'] for node in graph.nodes()]))
    graph_angs = np.linspace(0, 2 * np.pi, 1 + len(clusters))[1:]
    pos = {}
    for cluster_idx, cluster in enumerate(clusters):
        cluster_nodes = sorted([node for node in graph.nodes() if graph.nodes()[node]['cluster'] == cluster])
        cluster_nodes_angs = np.linspace(0, 2 * np.pi, 1 + len(cluster_nodes))[1:]
        for node_idx, node in enumerate(cluster_nodes):
            pos[node] = np.array([graph_rad * np.cos(graph_angs[cluster_idx])
                                  + cluster_rad * np.cos(cluster_nodes_angs[node_idx]),
                                  graph_rad * np.sin(graph_angs[cluster_idx])
                                  + cluster_rad * np.sin(cluster_nodes_angs[node_idx])])
    return pos


def display_graph(instance, path, graph_rad=2.5, cluster_rad=1, min_edge_weight=0):
    """
    Saves an image of the instance.

    Edge width and color depend on the edge weight (number of transactions).

    If the instance does not contain a solution a default method from the networkx library is used. Nodes that should be
    separated in different clusters have different colors.

    If a solution is present, the cluster centers are positioned along a circle, and the nodes of each clusters are
    positioned along a smaller circle. Nodes belonging to the same clusters have the same color

    :param instance:
    :param path: path where to store the png file
    :param graph_rad: radius of the cluster centers
    :param cluster_rad: radius of the clusters
    :param min_edge_weight: display only edges with edge weight big enough
    :return:
    """

    # palette = ['#ff0000', '#00ff00', '#ffff00', '#00ffff', '#ff00ff', '#c0c0c0', '#808080', '#800000',
    #            '#808000', '#008000', '#800080', '#008080', '#000080', '#0000ff']
    palette = [
        '#D98880', '#C39BD3', '#7FB3D5', '#76D7C4', '#7DCEA0', '#F7DC6F', '#F0B27A', '#F4F6F7', '#BFC9CA', '#85929E',
        '#7B241C', '#633974', '#1A5276', '#117864', '#196F3D', '#9A7D0A', '#935116', '#979A9A', '#5F6A6A', '#212F3C'
    ]

    g = nx.DiGraph()
    g.add_nodes_from(instance.get_all_nodes())
    if instance.solution is not None:
        for node in instance.solution:
            g.nodes()[node]['cluster'] = palette[instance.solution[node]]
    else:
        for node in g.nodes():
            if node in instance.separated_nodes:
                g.nodes()[node]['cluster'] = palette[instance.separated_nodes.index(node)]
            else:
                g.nodes()[node]['cluster'] = '#1f78b4'

    for cluster in sorted(instance.retrieve_clusters()):
        logging.info('C{}: color -> {}'.format(cluster, palette[cluster]))

    # logging.info([source for source in instance.get_all_sources()])
    g.add_edges_from([(source, target, {'weight': instance.exchanges[source][target]})
                      for source in instance.get_all_sources()
                      for target in instance.exchanges[source]
                      if instance.exchanges[source][target] >= min_edge_weight])
    # e.g. to add cluster tag to node: G.nodes['ACL']['cluster'] = k
    edge_colors = [float(g[u][v]['weight']) for u, v in g.edges()]
    weights = [float(g[u][v]['weight']) for u, v in g.edges()]
    node_colors = [g.nodes()[n]['cluster'] for n in g.nodes()]
    options = {
        # "node_color": "black",
        "node_size": 500,
        "linewidths": 0,
        # "width": 0.1,
        "pos": circular_position(g, graph_rad, cluster_rad) if instance.solution is not None else nx.spring_layout(g, k=0.6),
        # "arrows": False,
        "with_labels": True,
        "edge_color": edge_colors,
        "node_color": node_colors,
        # "width": [max(w / max(weights) * 10, 0.1) for w in weights]  # normalize_weights()
        "width": [(w - min(weights)) / (max(weights) - min(weights)) * 10 + 0.1 for w in weights]
    }

    plt.figure(figsize=(20, 20))
    nx.draw(g, **options)
    plt.savefig(path)
