import time

import logging
import localsolver
import math

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(message)s', level=logging.INFO)



class LSModel(object):

    def __init__(self, instance, nb_clusters=None, separated_nodes=None, min_cluster_size=None, force_nb_clusters=False,
                 min_cluster_weight=None, optimize_min_cluster_weight=False, max_cluster_size=None,
                 objective_constraint=False, cluster_index_weight=False, lex_objective=False):
        """
        :param instance: Instance to be solved
        :param nb_clusters: int; number of clusters in the instance
        :param separated_nodes: List[str]; list of backend that must be in different clusters
        :param min_cluster_size: int; minimum size of each cluster
        :param force_nb_clusters: bool; if True forces to use all clusters
        :param min_cluster_weight: int; minimum weight of cluster internal exchanges
        :param optimize_min_cluster_weight: bool; change optimization function to maximize the minimum of internal
        exchanges over all clusters, all nb_clusters will be used
        :param max_cluster_size: int; maximum size of a cluster
        :param objective_constraint: bool; if true adds a constraint on the max value of the objective function
        :param cluster_index_weight: bool; if true weights clusters in the objective function to favor smaller indices
        :param lex_objective: bool; if true considers edge weights as categories and optimize by category
        """

        self.instance = instance
        self.nb_clusters = nb_clusters
        self.nb_nodes = len(self.instance.get_all_nodes())
        self.solution = {}

        self.separated_nodes = separated_nodes if separated_nodes is not None else []
        self.min_cluster_size = min_cluster_size
        self.force_nb_clusters = force_nb_clusters
        self.min_cluster_weight = min_cluster_weight
        self.optimize_min_cluster_weight = optimize_min_cluster_weight
        if optimize_min_cluster_weight and not force_nb_clusters:
            logging.error('can only optimize min cluster weight when forcing to use all clusters')
            logging.error('optimize_min_cluster_weight == True => force_nb_clusters == True')
            raise ValueError
        self.max_cluster_size = max_cluster_size
        if max_cluster_size is None and objective_constraint:
            logging.error('can only add a constraint on the value of the objective function if max_cluster_size is set')
            logging.error('objective_constraint == True => max_cluster_size is not None')
            raise ValueError
        self.objective_constraint = objective_constraint
        self.cluster_index_weight = cluster_index_weight
        self.lex_objective = lex_objective
        self.time_limit = None
        self.infeasible = False
            


    def solve(self):
        """
        Solves the instance.
        :return:
        """
        
        '''
        #calculate a new equivalent non directed exchanges,
        # (in order to remove the cases where we have the double arrow source -> target and target -> source

        exchanges2 = {}
        for source in self.instance.get_all_sources():
            exchanges_s =  self.instance.exchanges[source]
            for target,v  in exchanges_s.items():
                s = source
                t = target
                if source > target:
                   s,t = t,s

                if s not in exchanges2:
                    exchanges2[s] = {}
                exchanges2_s = exchanges2[s]
                if t not in exchanges2_s:
                    exchanges2_s[t] = v
                else:
                    exchanges2_s[t] += v
        '''
        exchanges2 = self.instance.exchanges
        # Maximum number of clusters, to be set to number of OBEs
        nb_max_cluster = len(self.instance.get_all_nodes())

        with localsolver.LocalSolver() as ls:
            # Declares the optimization model
            self.model = ls.model
            # Set decisions: cluster_list[k] represents the OBEs in cluster k
          
            
            clusters_list = [self.model.set(self.nb_nodes) for k in range(self.nb_clusters)]

            # Each OBE must be in one cluster and one cluster only
            self.model.constraint(self.model.partition(clusters_list))

            # translation int to OBE name:
            obeToInt = {}
            intToObe = []

            for count,val in enumerate(self.instance.get_all_nodes()):
                obeToInt[val]=count
                intToObe.append(val)

            #x[n][k]: is n in cluster #k
            x={}
            for n in self.instance.get_all_nodes():
                x[n] = []
                for k in clusters_list:
                    x[n].append(self.model.contains(k,obeToInt[n]))

            #y[k][s][t]: (cluster, source name, target name): is the arrow s->t inner to the cluster k
            #z[s][t] (not used): indexed by OBE names (source and target): is the arrow s->t inner to the same cluster
            y = {}
            z = {}
            for source in exchanges2:
                z[source] = {}
                z_s = z[source]
                for target in exchanges2[source]:
                    z_s_t_k = []
                    for k in range(0,self.nb_clusters):
                        if k not in y:
                            y[k] = {}
                        y_k = y[k]
                        if source not in y_k:
                            y_k[source] = {}
                        y_k_s = y_k[source]
                        a = self.model.and_(x[source][k],x[target][k])
                        y_k_s[target] = a
                        z_s_t_k.append(a)
                    z_s[target] = self.model.or_(z_s_t_k)

            '''
            #internal traffic: definition using z (not used)
            internal_traffic = []
          
            for source in exchanges2:
                for target,exchange in exchanges2[source].items():
                    internal_traffic.append(
                        self.model.iif(z[source][target],
                            exchange,
                            0))
            total_traffic = self.model.sum(internal_traffic)
            '''
            

            #cluster sizes:
            cluster_sizes = []
            for k in clusters_list:
                cluster_sizes.append(self.model.count(k))


            #internal traffic per cluster:
            total_traffic_k = []
            
            for k in range(0,len(clusters_list)):
                weights_in_cluster = []
                for source in exchanges2:
                    for target,exchange in exchanges2[source].items():
                        weights_in_cluster.append(
                            self.model.iif(y[k][source][target],
                            exchange,
                            0))
                total_traffic_k.append(self.model.sum(weights_in_cluster))

            #internal traffic
            total_traffic = self.model.sum(total_traffic_k)

            #nb elmt in cluster
            cluster_count = [self.model.count(k) for k in clusters_list]

            #min_cluster_size (if cluster is not empty)
            min_cluster_size = self.min_cluster_size

            if self.min_cluster_size is None and self.force_nb_clusters:
                min_cluster_size = 1
            if min_cluster_size is not None and min_cluster_size > 0:
                for k_idx,k in enumerate(clusters_list):
                    if self.force_nb_clusters:
                        self.model.add_constraint(cluster_count[k_idx] >= min_cluster_size)
                    else:
                        self.model.add_constraint(self.model.or_(cluster_count[k_idx] >= min_cluster_size,cluster_count[k_idx] == 0))

            #max_cluster_size
            if self.max_cluster_size is not None:
                for k in clusters_list:
                    self.model.add_constraint(self.model.count(k) <= self.max_cluster_size)

            #min_cluster_weight (if cluster is not empty)
            if self.min_cluster_weight is not None and self.min_cluster_weight > 0:
                for k_idx,k in enumerate(total_traffic_k):
                    if self.force_nb_clusters:
                        self.model.add_constraint(k >= self.min_cluster_weight)
                    else:
                        self.model.add_constraint(self.model.or_(k >= self.min_cluster_weight,cluster_count[k_idx] == 0))

            for count, node in enumerate(self.separated_nodes):
                self.model.add_constraint(self.model.contains(clusters_list[count],obeToInt[node]))

            self.model.maximize(total_traffic)


            
            # heuristic: we add an equilibrium score in order to have clusters with similar weights
            # let's try to optimize the sum -ln(traffic_proportion_in_cluster)*cluster_traffic
            '''
            equilibrum_score = 1-4*self.model.sum([k*self.model.log((k+0.000000001)/total_traffic)
                                                   for k in total_traffic_k])/(total_traffic*math.log(self.nb_clusters))
            heuristic_total_traffics = total_traffic*equilibrum_score

            self.model.maximize(heuristic_total_traffics)
            '''
            
            if self.time_limit is not None:
                phase = ls.create_phase()
                phase.set_optimized_objective(0)
                phase.set_time_limit(self.time_limit)



            # close the model
            self.model.close()

            # Parameterizes the solver
 

            # solve model
            start = time.time()
            ls.solve()
            end = time.time()

            self.running_time = end - start
            logging.info("Running time (sec.) = {}".format(self.running_time))
    

            status = ls.solution.get_status()
            if status in (localsolver.LSSolutionStatus.FEASIBLE,localsolver.LSSolutionStatus.OPTIMAL):
                logging.info("sum internal traffic = {}".format(total_traffic.value))
                #logging.info("equilibrium score " + str(equilibrum_score.value))
                for node in self.instance.get_all_nodes():
                    for k in range(self.nb_clusters):
                        if x[node][k].value == 1:
                            self.solution[node] = k
                            break
                self.instance.set_solution(self.solution)

                # Writes the solution in a file
                with open('./LS_solution.txt', 'w') as f:
                    f.write("%d\n" % total_traffic.value)
                    for cluster in range(self.nb_clusters):
                        for obe in self.instance.get_all_nodes():
                            f.write("%s " % x[obe][cluster].value)
                        f.write("\n")
            else:
                self.infeasible = True
                logging.warning("Problem has no solution")
       

    def set_parameters(self, time_limit):
        self.time_limit = time_limit

    def print_solution(self):
        if self.infeasible:
            return

        # x_{v,k} : 1 if vertex v is in cluster k, 0 otherwise
        for node in self.instance.get_all_nodes():
            logging.info('{} = {}'.format(node, self.solution[node]))

        # y_{u,v,k} : 1 if u and v are connected and are part of cluster k, 0 otherwise
        for source in self.instance.get_all_sources():
            for target in self.instance.exchanges[source]:
                k1 = self.solution[source]
                k2 = self.solution[target]
                if k1 == k2:
                    logging.info('{}_{} = {}'.format(source, target, k1))

    def write_to_file(self, path):
        pass
        '''
        with get_environment().get_output_stream(path) as fp:
            self.model.solution.export(fp, "json")
        '''

