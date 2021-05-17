"""
No unit tests in this class.

Class used to run a test campaign with different nb_clusters and min_edge_weight.
"""

import logging
import pathlib
import unittest

from openbackend_clustering.instance.instance import InstanceMatrixParser, display_graph
from openbackend_clustering.model.local_solver_model import LSModel

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(message)s', level=logging.INFO)

current_path = pathlib.Path(__file__).parent.absolute()


def create_instance(min_edge_weight):
    parser = InstanceMatrixParser(str(current_path) + '/../openbackend_clustering/resources/CA_extract.matrix')
    return parser.create_instance(ignore=['node_22', 'node_8', 'node_67', 'node_46'], min_edge_weight=min_edge_weight)


def run_test_min_edge(min_edge_weight, nb_clusters):
    # create instance
    instance = create_instance(min_edge_weight)
    separated_nodes = [node for node in ['node_16', 'node_65', 'node_39', 'node_57', 'node_68'] if node in instance.get_all_nodes()]
    min_cluster_weight = int(instance.total_exchanges() / nb_clusters * 0.25)
    # create model and solve
    model = LSModel(instance, nb_clusters=nb_clusters, separated_nodes=separated_nodes,
                     min_cluster_weight=min_cluster_weight)
    model.solve()
    model.print_solution()
    instance.print_kpis()
    # save instance with solution
    instance.set_solution(model.solution)
    instance.save('instance_min_edge_weight_{}_w_{}_clusters.pickle'.format(min_edge_weight, nb_clusters))
    # save a graphical representation of the solution
    display_graph(instance, 'solution_min_edge_weight_{}_w_{}_clusters.png'.format(min_edge_weight, nb_clusters))


class MinEdgeWeightTest(unittest.TestCase):

    def test_min_edge_weight_20_w_6_clusters(self):
        run_test_min_edge(20, 6)

    def test_min_edge_weight_20_w_5_clusters(self):
        run_test_min_edge(20, 5)

    def test_min_edge_weight_10_w_6_clusters(self):
        run_test_min_edge(10, 6)

    def test_min_edge_weight_10_w_5_clusters(self):
        run_test_min_edge(10, 5)

    def test_min_edge_weight_5_w_6_clusters(self):
        run_test_min_edge(5, 6)

    def test_min_edge_weight_5_w_5_clusters(self):
        run_test_min_edge(5, 5)
