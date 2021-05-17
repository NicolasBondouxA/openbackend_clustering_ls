import itertools
import logging
import pathlib
import unittest

from openbackend_clustering.model.local_solver_model import LSModel
from openbackend_clustering.instance.instance import  InstanceMatrixParser, InstancePickleParser, \
    display_graph

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(message)s', level=logging.INFO)

current_path = pathlib.Path(__file__).parent.absolute()
path = str(current_path) + '/../openbackend_clustering/resources/CA_extract.matrix'


class LSModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        parser = InstanceMatrixParser(path)
        cls.instance = parser.create_instance()

        matrix_path_3 = str(current_path) + '/../openbackend_clustering/resources/matrix_instance_3'
        parser_matrix_3 = InstanceMatrixParser(matrix_path_3)
        cls.instance_3 = parser_matrix_3.create_instance()

        matrix_path_5 = str(current_path) + '/../openbackend_clustering/resources/matrix_instance_5'
        parser_matrix_5 = InstanceMatrixParser(matrix_path_5)
        cls.instance_5 = parser_matrix_5.create_instance()

    def test_model(self):
        model = LSModel(self.instance, nb_clusters=5, separated_nodes=['node_16', 'node_65', 'node_39', 'node_57', 'node_68'],
                         min_cluster_size=5)
        model.solve()
        model.print_solution()
        self.instance.print_kpis()
        # save instance with solution
        self.instance.set_solution(model.solution)
        self.instance.save('instance.pickle')
        # load saved instance and check if equality holds
        parser = InstancePickleParser('instance.pickle')
        pickled_instance = parser.create_instance()
        self.assertEqual(self.instance, pickled_instance)
        # save a graphical representation of the solution
        display_graph(self.instance, 'solution_instance.png')

    def test_model_matrix_instance_3(self):
        model = LSModel(self.instance_3, nb_clusters=3)
        model.solve()
        model.print_solution()
        self.instance_3.print_kpis()
        # save instance with solution
        self.instance_3.set_solution(model.solution)
        self.instance_3.save('instance_3.pickle')
        # load saved instance and check if equality holds
        parser = InstancePickleParser('instance_3.pickle')
        pickled_instance = parser.create_instance()
        self.assertEqual(self.instance_3, pickled_instance)
        # save a graphical representation of the solution
        display_graph(self.instance_3, 'solution_instance_3.png')

    def test_model_matrix_instance_5(self):
        model = LSModel(self.instance_5, nb_clusters=5, separated_nodes=['node_0', 'node_4'], min_cluster_size=2)
        model.solve()
        model.print_solution()
        self.instance_5.print_kpis()
        # save instance with solution
        self.instance_5.set_solution(model.solution)
        self.instance_5.save('instance_5.pickle')
        # load saved instance and check if equality holds
        parser = InstancePickleParser('instance_5.pickle')
        pickled_instance = parser.create_instance()
        self.assertEqual(self.instance_5, pickled_instance)
        # save a graphical representation of the solution
        display_graph(self.instance_5, 'solution_instance_5.png')

    def test_load_instance_results(self):
        parser = InstancePickleParser('instance.pickle')
        pickled_instance = parser.create_instance()
        # save a graphical representation of the solution
        display_graph(pickled_instance, 'solution_instance_pickle.png')
        pickled_instance.print_kpis()

    def _experimentation_plan(self, separated_nodes_plan, nb_clusters_plan, optimize_min_cluster_weight_plan,
                              min_cluster_weight_plan, ignore_plan, path, parser_class):

        plan_nb = 0
        parser = parser_class(path)

        for separated_nodes, nb_clusters, optimize_min_cluster_weight, min_cluster_weight, ignore in \
                itertools.product(separated_nodes_plan, nb_clusters_plan, optimize_min_cluster_weight_plan,
                                  min_cluster_weight_plan, ignore_plan):
            if separated_nodes and nb_clusters < 5:
                continue
            if optimize_min_cluster_weight and min_cluster_weight is not None:
                continue
            file_name = 'sep[{}]-nb[{}]-opt_min[{}]-min_w[{}]-ignore[{}]'.format(
                len(separated_nodes) != 0,
                nb_clusters,
                optimize_min_cluster_weight,
                min_cluster_weight,
                len(ignore)
            )

            instance = parser.create_instance(ignore=ignore)
            min_cluster_weight = None if min_cluster_weight is None \
                else int(instance.total_exchanges() / nb_clusters * min_cluster_weight)

            plan_nb += 1
            logging.info('plan nb: {}'.format(plan_nb))
            logging.info(' - separated_nodes: {}'.format(separated_nodes))
            logging.info(' - nb_clusters: {}'.format(nb_clusters))
            logging.info(' - optimize_min_cluster_weight: {}'.format(optimize_min_cluster_weight))
            logging.info(' - min_cluster_weight: {}'.format(min_cluster_weight))
            logging.info(' - ignore: {}'.format(ignore))
            logging.info(' - file_name: {}'.format(file_name))

            model = LSModel(instance, nb_clusters=nb_clusters, separated_nodes=separated_nodes, force_nb_clusters=True,
                             min_cluster_weight=min_cluster_weight,
                             optimize_min_cluster_weight=optimize_min_cluster_weight)
            model.solve()
            if model.infeasible:
                logging.info('NOT FEASIBLE')
                continue
            model.print_solution()
            instance.print_kpis()
            # save instance with solution
            instance.set_solution(model.solution)
            instance.save('{}.pickle'.format(file_name))
            # save graphical representation of the solution
            display_graph(instance, '{}.png'.format(file_name))

    def test_experimentation_plan_instance(self):
        """
        Not a test. Experimenting different optimization setups with the complete instance.
        :return:
        """

        # always forcing the nb of clusters
        separated_nodes_plan = [['node_16', 'node_65', 'node_39', 'node_57', 'node_68'], []]
        nb_clusters_plan = [3, 4, 5, 6]
        optimize_min_cluster_weight_plan = [False, True]
        min_cluster_weight_plan = [None, 0.25, 0.5]  # percentage of total graph weight / nb clusters
        ignore_plan = [[], ['node_22', 'node_8'],['node_67', 'node_46', 'node_22', 'node_8']]

        self._experimentation_plan(separated_nodes_plan, nb_clusters_plan, optimize_min_cluster_weight_plan,
                                   min_cluster_weight_plan, ignore_plan, path, InstanceMatrixParser)

    def test_experimentation_plan_instance_5(self):
        """
        More or less a test. Experimenting different optimization setups with a toy instance.
        :return:
        """

        # always forcing the nb of clusters
        separated_nodes_plan = [[]]
        nb_clusters_plan = [2, 3]
        optimize_min_cluster_weight_plan = [False, True]
        min_cluster_weight_plan = [None, 0.25, 0.5]  # percentage of total graph weight / nb clusters
        ignore_plan = [[]]

        self._experimentation_plan(separated_nodes_plan, nb_clusters_plan, optimize_min_cluster_weight_plan,
                                   min_cluster_weight_plan, ignore_plan,
                                   str(current_path) + '/../openbackend_clustering/resources/matrix_instance_5',
                                   InstanceMatrixParser)

    def test_max_objective_function_value_matrix_instance_5(self):
        model = LSModel(self.instance_5, nb_clusters=5, max_cluster_size=2, objective_constraint=True)
        model.solve()
        model.print_solution()
        self.instance_5.print_kpis()

    def test_lex_objective_function_matrix_instance_5(self):
        model = LSModel(self.instance_5, nb_clusters=5, max_cluster_size=3, lex_objective=True)
        model.solve()
        model.print_solution()
        self.instance_5.print_kpis()
