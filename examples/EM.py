from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import time
import numpy as np
import pandas as pd
import re
import itertools

from string import find, join, split
import xml.etree.ElementTree as ET




# ToDO: add error messages and throw exceptions
# ToDO: When calculating likelihood, ESS table entries is multiplied with CPD table entries and then sum out. One problem is when some ESS entries are zero, CPT entries become NAN. In this case, product of a pair of ESS and CPT entries are one instead of zero.
# ToDO: Create a function that estimate the time of whole process by running one iteration of evidence.
# ToDo: Use utilities instead of likelihood to evaluate results.
# ToDo: Create an interface and a wrapper that Priyanka and Angel can use. This wrapper should take how much weight one relation should change?

'''
log_levels = {
    'DEBUG': 0,
    'INFO': 1,
}
'''


class EM:

    def __init__(self, edges = None, frag_name = None, evidence_filename = None, verbose = False):

        if edges == None:
            if frag_name == None:
                print "Specify the Bayesian Network by providing a list of edges or a frag file"
                quit()
            else:
                edges = self.get_model_from_frag(frag_name)

        self.verbose = verbose
        self.model = BayesianModel(edges)
        self.evidences = self.read_evidence(evidence_filename)
        self.ess_matrices = None

        # variables = model.nodes()
        # edges = model.edges()

    def get_model(self):
        return self.model.copy()

    # Initialize CPDs randomly
    def add_random_cpds(self):

        nodes = self.model.nodes()
        # Todo: define the variable for number of random evidence somewhere
        # Todo: find a better way to generate evidence from a random distribution any time
        evidence = np.random.randint(low=0,high=1000,size=(100,len(nodes)))
        evidence = (evidence > np.random.randint(low=0,high=1000)).astype(int)
        # Since the learned CPT will not be complete if some variables have not been assigned all values in its domain,
        # an array of all zeros and of all ones are appended to evidence.
        evidence = np.vstack((evidence, np.zeros((1, len(nodes)))))
        evidence = np.vstack((evidence, np.ones((1, len(nodes)))))
        values = pd.DataFrame(evidence, columns=nodes)

        self.model.fit(values)


    # Construct Expected Sufficient Statistics (ESS) tables.
    def construct_ess(self):

        cpds = self.model.get_cpds()

        ess_matrices = []
        for cpd in cpds:
            # Create a ess_matrix to store ess values for every child, parent values
            # this matrix has the same format as CPD
            # ToDo: create a separate class for ess_matrix
            ess_matrix = cpd.copy()
            ess_matrix.values.fill(0)

            ess_matrices.append(ess_matrix)
            # print ess_matrix.values
            # print ess_matrix.variables

        self.ess_matrices = ess_matrices


    # ToDO: how to select best results from random starts
    def group_expectation_maximization(self, number_of_starts=10, difference_threshold=0.001, max_number_of_iterations=1000):

        max_likelihood = 0
        best_cpds = None

        for n in range(number_of_starts):

            # delete CPD tables and ESS tables from previous run

            # In each iteration, reinitialize CPD tables randomly and fill ESS tables with zeros
            self.add_random_cpds()
            self.construct_ess()

            if self.verbose:
                print "\n\n==================One run===================="
                print "\n------------------Before--------------------"
                for cpd in self.model.get_cpds():
                    print cpd

            # Select the best theta by comparing the likelihood
            likelihood = self.expectation_maximization(difference_threshold, max_number_of_iterations)
            if likelihood > max_likelihood:
                best_cpds = self.model.get_cpds()
                max_likelihood = likelihood

            if self.verbose:
                print "\n------------------After--------------------"
                for cpd in self.model.get_cpds():
                    print cpd


        print '\n\n==================Best set of parameters of likelihood: ', max_likelihood, "====================="
        for cpd in best_cpds:
            print cpd

    def expectation_maximization(self, difference_threshold, max_number_of_iterations):

        # ToDo: check if the convergence test is correct

        start = time.time()
        i = 0
        for i in range(max_number_of_iterations):

            prev_cpds = [cpd.copy() for cpd in self.model.get_cpds()]

            self.compute_ess()
            print 'here'
            self.compute_cpds()

            diff_max = 0
            cpds = self.model.get_cpds()
            for ci in range(len(cpds)):

                diff = np.max(abs(prev_cpds[ci].values - self.model.get_cpds()[ci].values))
                if diff > diff_max:
                    diff_max = diff

            if diff_max < difference_threshold:
                break

            #print diff_max


            #print self.model.get_cpds()[0]
            #for cpd in em.model.get_cpds():
            #    print cpd

        end = time.time()
        print "time for ", i, " iterations is ", end - start

        # Find "likelihood" of evidence given theta P(V|theta)
        # This can be done by using ESS tables and CPD tables
        cpds = self.model.get_cpds()
        ess_matrices = self.ess_matrices
        likelihood = 0.0
        for i in range(len(cpds)):
            l = np.sum(cpds[i].values ** ess_matrices[i].values)
            likelihood = likelihood + l

        return likelihood

    def compute_ess(self):

        inference = VariableElimination(self.model)
        
        for evidence in self.evidences:

            self.display("\n\n============================Evidence===========================")
            self.display(evidence, "Evidence")
            evidence_variables = evidence.keys()
            #print evidence.keys()

            for ess_matrix in self.ess_matrices:

                self.display("\n\n-----------------(Child, Parents)------------------")
                self.display(ess_matrix, 'ESS table before update')

                # variables: all variables in this ess table (The order matters)
                # query_variables: variables that need to query
                # query_indices: all possible values assigned to variables
                # Use deep copy of list: slicing [:]

                variables = ess_matrix.variables[:]
                query_variables = ess_matrix.variables[:]
                query_indices = [i for i in itertools.product(range(2), repeat = len(query_variables))]
                self.display(variables, 'Variables in this ESS table')

                # eliminate query variables if they are in evidence
                for v in variables:
                    if v in evidence_variables:
                        query_variables.remove(v)

                        new_query_indices = []
                        value = evidence[v]
                        index = variables.index(v)
                        for qi in query_indices:
                            if qi[index] == value:
                                new_query_indices.append(qi)
                        query_indices = new_query_indices


                self.display(query_variables, 'Variables not in evidence')
                self.display(query_indices, 'Query indices for variables')

                # Updating ESS table
                # 1. no query variables means every variables have an assigned value in the evidence
                #    add 1 to the ess table where variable values match
                if len(query_variables) == 0:
                    ess_matrix.values[query_indices[0]] = ess_matrix.values[query_indices[0]] + 1.0
                # 2. use inference to get Probability
                else:
                    query_result = inference.query(variables = query_variables, evidence = evidence)['all']
                    self.display(query_result, 'Query result')

                    # Note: Variable order is different in ESS table and query result
                    # query_result_index: all possible values assigned to queried variables in the order of all variables
                    query_result_indices = []
                    for qi in query_indices:
                        qri = []
                        for qrv in query_result.variables:
                            qri.append(qi[variables.index(qrv)])
                        query_result_indices.append(tuple(qri))
                    self.display(query_result_indices, 'Query result indices for query result')

                    for i in range(len(query_indices)):
                        ess_matrix.values[query_indices[i]] = ess_matrix.values[query_indices[i]] + query_result.values[query_result_indices[i]]

                self.display(ess_matrix, 'ESS table after update')


    def compute_cpds(self):

        cpds = self.model.get_cpds()

        # Note: ESS table list and CPT table list have the same order of (Child, Parents)
        for i in range(len(cpds)):

            cpd = cpds[i]
            self.display("\n\n===================CPD=====================")
            self.display("\n--------------before update-----------------")
            self.display(cpd)
            ess = self.ess_matrices[i]
            self.display("\n--------------ess table-----------------")
            self.display(ess)

            # Normalize ESS tables to get CPDs (Easily)
            cpd.values = ess.normalize(inplace = False).values
            self.display("\n-----------------after update------------------")
            self.display(cpd)

            # Resolve nan values for some CPT entries by assigning uniform distribution
            cpd.values[np.isnan(cpd.values)] = 1.0 / cpd.values.shape[0]
            self.display("\n-----------------after rationalize----------------")
            self.display(cpd)





    # Read blogdb and store in evidences.
    #
    # evidences: a list of dictionaries where each dictionary has format {'node_name': node_value}
    #            each dictionary can be directly used in pgmpy's VE inference as evidence
    #            node without an assigned value will not be in this dictionary

    def read_evidence(self, evidence_filename):

        evidences = []

        blogdb = open(evidence_filename, 'r')
        prev_obj_id = -1
        
        evidence = {}
        for line in blogdb:
            
            if line == '\n':
                continue

            # delete any space
            line = line.replace(' ','')

            # obj_id: o1, o2, o3 in blogdb
            obj_id = int(re.split('O|,',line)[1])
            
            # if object id does not start at 0 in blogdb
            if prev_obj_id == -1:
                prev_obj_id = obj_id

            if obj_id != prev_obj_id:
                evidences.append(evidence)
                evidence = {}

            # replace object id with x to match standard node name
            rel = re.split('=',line)[0].replace('O' + str(obj_id),'x')

            if 'True' == re.split('=|\n',line)[1]:
                evidence[rel] = 1
            elif 'False' == re.split('=|\n',line)[1]:
                evidence[rel] = 0

            prev_obj_id = obj_id

        # append the last set of evidence    
        evidences.append(evidence)

        blogdb.close()

        return evidences


    def get_model_from_frag(self, frag_name):

        edges = []

        fh = open(frag_name, 'r')
        for line in fh:
            if find(line, '//') == 0:
                continue

            if ''.join(split(line)) == '':
                continue

            if find(line, '->') != -1:
                line = ''.join(split(line))
                lhs_rhs = split(line, '->')
                lhs = lhs_rhs[0]
                rhs = lhs_rhs[1]

                parents = split(lhs, '~')
                for parent in parents:
                    edges.append((parent, rhs))
            #else:
            #    add_node(line)
        fh.close()

        return edges


    '''
    def _add_node(self, line):
        line = ''.join(split(line))
        net.node(line)
    '''

    # Write CPDs to a PMML file
    def write_cpds(self, pmml_name):

        ET.register_namespace('', "http://www.dmg.org/PMML-3_0")

        #tree = ET.parse(pmml_name)
        #root = tree.getroot()
        #root.remove(root.find('{http://www.dmg.org/PMML-3_0}DataDictionary'))

        root = ET.Element('PMML', {'version': "3.0"})
        root.append(ET.Element('{http://www.dmg.org/PMML-3_0}Header', {"copyright": "Georgia Institute of Technology"}))
        root.append(ET.Element('{http://www.dmg.org/PMML-3_0}DataDictionary', {}))
        data_dic = root[1]

        variable_ids = {}
        id_count = 0
        for ic, cpd in enumerate(self.model.get_cpds()):

            child = cpd.variables[0]
            parents = []
            if len(cpd.variables) > 1:
                parents = cpd.variables[1:]

            # Assign a node id to all new variables
            for var in cpd.variables:
                if var not in variable_ids:
                    variable_ids[var] = id_count
                    id_count = id_count + 1

            data_dic.append(ET.Element('{http://www.dmg.org/PMML-3_0}DataField',
                                       {'name': child, 'optype': 'categorical', 'id': str(variable_ids[child])}))
            data = data_dic[ic]
            data.append(ET.Element('{http://www.dmg.org/PMML-3_0}Extension', {}))
            data.append(ET.Element('{http://www.dmg.org/PMML-3_0}Value', {'value': "True"}))
            data.append(ET.Element('{http://www.dmg.org/PMML-3_0}Value', {'value': "False"}))
            ext = data[0]

            ext.append(ET.Element('{http://www.dmg.org/PMML-3_0}X-NodeType', {}))
            ext[0].text = 'chance'
            ext.append(ET.Element('{http://www.dmg.org/PMML-3_0}X-Position', {'x': '0', 'y': '0'}))
            ext.append(ET.Element('{http://www.dmg.org/PMML-3_0}X-Definition', {}))
            definition = ext[2]

            if len(parents) != 0:
                for ip, parent in enumerate(parents):
                    definition.append(ET.Element('{http://www.dmg.org/PMML-3_0}X-Given', {}))
                    definition[ip].text = str(variable_ids[parent])


            # Table indices are in different orders for pgmpy standard and pmml file standard
            #
            # For pgmpy
            # Child                F                                T
            # Parent1         F            T                 F                 T
            # Parent2     F       T     F       T       F         T        F        T
            # index: table[child_idx][parent1_idx][parent2_idx]
            #
            # For pmml
            # child       T       F     T       F       T         F        T        F
            # parent1                T                                F
            # Parent2         T            F                 T                 F
            # order of child_idx, parent1_idx, parent2_idx: TTT, FTT, TTF, FTF, TFT, FFT, TFF,FFF

            definition.append(ET.Element('{http://www.dmg.org/PMML-3_0}X-Table', {}))
            table_index = [i for i in itertools.product(range(2), repeat = len(cpd.variables))]
            table_index.reverse()
            pmml_table_index = []
            for idx in table_index:
                idx = list(idx)
                idx.insert(0, idx[-1])
                idx.pop()
                pmml_table_index.append(tuple(idx))

            table_str = str([cpd.values[idx] for idx in pmml_table_index])
            table_str = table_str.replace('[', '')
            table_str = table_str.replace(']', '')
            table_str = table_str.replace(',', ' ')
            definition[len(definition) - 1].text = table_str

        tree = ET.ElementTree(root)
        tree.write(pmml_name, xml_declaration=True, encoding='US-ASCII', method='xml')


    def display(self, variable, variable_name = ""):
        if self.verbose:
            if variable_name != "":
                print "*** ", variable_name + " ***"
            print variable




if __name__ == '__main__':

    # ToDo: edit examples


    edges = [('isA(x,Cup)', 'atLocation(x,Shelf)'),
                                  ('isA(x,Bowl)', 'atLocation(x,Shelf)'),
                                  ('atLocation(x,Shelf)', 'atLocation(x,Kitchen)'),
                                  ('atLocation(x,Shelf)', 'atLocation(x,Bathroom)')]


    cpd_poll = TabularCPD(variable='isA(x,Cup)', variable_card=2,
                          values=[[0.9], [0.1]])
    cpd_smoke = TabularCPD(variable='isA(x,Bowl)', variable_card=2,
                           values=[[0.3], [0.7]])
    cpd_cancer = TabularCPD(variable='atLocation(x,Shelf)', variable_card=2,
                            values=[[0.03, 0.05, 0.001, 0.02],
                                    [0.97, 0.95, 0.999, 0.98]],
                            evidence=['isA(x,Bowl)', 'isA(x,Cup)'],
                            evidence_card=[2, 2])
    cpd_xray = TabularCPD(variable='atLocation(x,Kitchen)', variable_card=2,
                          values=[[0.9, 0.2], [0.1, 0.8]],
                          evidence=['atLocation(x,Shelf)'], evidence_card=[2])
    cpd_dysp = TabularCPD(variable='atLocation(x,Bathroom)', variable_card=2,
                          values=[[0.65, 0.3], [0.35, 0.7]],
                          evidence=['atLocation(x,Shelf)'], evidence_card=[2])

    #print cpd_cancer
    #print cpd_cancer.variables
    #print cpd_cancer.values[1]
    # Associating the parameters with the model structure.
    #model.add_cpds(cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp)
    
    #em = EM(edges=edges, evidence_filename='test.blogdb', verbose = False)


    em = EM(frag_name='kitchen.frag', evidence_filename='kitchen.blogdb', verbose=False)
    em.group_expectation_maximization(number_of_starts=1, difference_threshold=0.1, max_number_of_iterations=30)
    model = em.get_model()
    infer = VariableElimination(model)
    q = infer.query(variables=['AtLocation(x,Table)','AtLocation(x,Bathroom)','AtLocation(x,Sink)','AtLocation(x,Dog)','AtLocation(x,Bedroom)'], evidence={'IsA(x,Mug.n.01)': 1})
    print "inference result"
    print q['AtLocation(x,Table)']
    print q['AtLocation(x,Bathroom)']
    print q['AtLocation(x,Sink)']
    print q['AtLocation(x,Dog)']
    print q['AtLocation(x,Bedroom)']
    em.write_cpds('new.pmml')