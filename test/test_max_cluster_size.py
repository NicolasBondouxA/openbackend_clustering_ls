"""
No unit tests in this class.

Class used to run a test campaign with different max_cluster_size and min_edge_weight.
"""

import logging
import pathlib
import unittest

from openbackend_clustering.instance.instance import InstanceMatrixParser, display_graph, InstancePickleParser
from openbackend_clustering.model.local_solver_model import LSModel

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(message)s', level=logging.INFO)

current_path = pathlib.Path(__file__).parent.absolute()


def create_instance(min_edge_weight):
    parser = InstanceMatrixParser(str(current_path) + '/../openbackend_clustering/resources/CA_extract.matrix')
    return parser.create_instance(ignore=['node_22', 'node_8', 'node_67', 'node_46'], min_edge_weight=min_edge_weight)


def run_test_max_cluster_size(min_edge_weight, max_cluster_size, graph_rad=5):
    # create instance
    instance = create_instance(min_edge_weight)
    separated_nodes = [node for node in ['node_16', 'node_65', 'node_39', 'node_57', 'node_68'] if node in instance.get_all_nodes()]
    # min_cluster_weight = int(instance.total_exchanges() / nb_clusters * 0.25)
    # create model and solve
    model = LSModel(instance, nb_clusters=20, separated_nodes=separated_nodes,
                     max_cluster_size=max_cluster_size)
    model.solve()
    model.print_solution()
    instance.print_kpis()
    # save instance with solution
    instance.set_solution(model.solution)
    instance.save('instance_min_w_{}_max_c_{}.pickle'.format(min_edge_weight, max_cluster_size))
    # save a graphical representation of the solution
    display_graph(instance, 'solution_min_w_{}_max_c_{}.png'.format(min_edge_weight, max_cluster_size),
                  graph_rad=graph_rad)


class MaxClusterSizeTest(unittest.TestCase):

    def test_min_w_1_max_c_5(self):
        run_test_max_cluster_size(min_edge_weight=1, max_cluster_size=5, graph_rad=6)

    def test_min_w_1_max_c_8(self):
        run_test_max_cluster_size(min_edge_weight=1, max_cluster_size=8, graph_rad=6)

    def test_min_w_5_max_c_5(self):
        run_test_max_cluster_size(min_edge_weight=5, max_cluster_size=5)

    def test_min_w_5_max_c_8(self):
        run_test_max_cluster_size(min_edge_weight=5, max_cluster_size=8)

    def test_min_w_10_max_c_5(self):
        run_test_max_cluster_size(min_edge_weight=10, max_cluster_size=5)

    def test_min_w_10_max_c_8(self):
        run_test_max_cluster_size(min_edge_weight=10, max_cluster_size=8)

    def test_min_w_20_max_c_5(self):
        run_test_max_cluster_size(min_edge_weight=20, max_cluster_size=5)

    def test_min_w_20_max_c_8(self):
        run_test_max_cluster_size(min_edge_weight=20, max_cluster_size=8)

    # @unittest.skip("")
    def test_redraw_solutions(self):
        instances = {}
        for w in [1, 20]:  # [1, 5, 10, 20]:
            instances[w] = {}
            for c in [5, 8]:

                instance_name = 'instance_min_w_{}_max_c_{}'.format(w, c)
                pickle_path = str(current_path) \
                              + '/../openbackend_clustering/resources/{}.pickle'.format(instance_name)
                parser = InstancePickleParser(pickle_path)
                instance = parser.create_instance()
                instance.compress_solution()
                instances[w][c] = instance

        logging.info('self = [w == 20] and [c == 8]')
        logging.info('other  = [w == 20] and [c == 5]')
        instances[20][8].compare_solutions(instances[20][5])

        logging.info('self  = [w == 20] and [c == 8] ')
        logging.info('other = [w == 1] and [c == 8]')
        instances[20][8].compare_solutions(instances[1][8])

        logging.info('self  = [w == 20] and [c == 5]')
        logging.info('other = [w == 1] and [c == 5]')
        instances[20][5].compare_solutions(instances[1][5])

        for w in [1, 20]:  # [1, 5, 10, 20]:
            for c in [5, 8]:
                instance_name = 'instance_min_w_{}_max_c_{}'.format(w, c)
                logging.info('kpis for instance: {}'.format(instance_name))
                instances[w][c].print_kpis()
                display_graph(instances[w][c], '{}.png'.format(instance_name), graph_rad=6 if w == 1 else 5)

    def test_filter_instance_w_20(self):
        instance = create_instance(20)
        self.assertEqual(34, len(instance.get_all_nodes()))
