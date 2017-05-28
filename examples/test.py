from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import time
import numpy as np
import re


cancer_model = BayesianModel([('Pollution', 'Cancer'), 
                              ('Smoker', 'Cancer'),
                              ('Cancer', 'Xray'),
                              ('Cancer', 'Dyspnoea')])




cpd_poll = TabularCPD(variable='Pollution', variable_card=2,
                      values=[[0.9], [0.1]])
cpd_smoke = TabularCPD(variable='Smoker', variable_card=2,
                       values=[[0.3], [0.7]])
cpd_cancer = TabularCPD(variable='Cancer', variable_card=2,
                        values=[[0.03, 0.05, 0.001, 0.02],
                                [0.97, 0.95, 0.999, 0.98]],
                        evidence=['Smoker', 'Pollution'],
                        evidence_card=[2, 2])
cpd_xray = TabularCPD(variable='Xray', variable_card=2,
                      values=[[0.9, 0.2], [0.1, 0.8]],
                      evidence=['Cancer'], evidence_card=[2])
cpd_dysp = TabularCPD(variable='Dyspnoea', variable_card=2,
                      values=[[0.65, 0.3], [0.35, 0.7]],
                      evidence=['Cancer'], evidence_card=[2])
#print cpd_cancer
#print cpd_cancer.variables
#print cpd_cancer.values[1]
# Associating the parameters with the model structure.
cancer_model.add_cpds(cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp)


# Checking if the cpds are valid for the model.
#print cancer_model.check_model()
# print cancer_model.edges()
# print cancer_model.nodes()
#print cancer_model.get_cpds()


cpds = cancer_model.get_cpds()

ess_matrices = []
for cpd in cpds:
	ess_matrix = cpd.copy()
	ess_matrix.values.fill(0)
	print ess_matrix.values




cancer_infer = VariableElimination(cancer_model)


start = time.time()
q = cancer_infer.query(variables=['Smoker', 'Cancer'], evidence={'Dyspnoea':1, 'Pollution':0})
end = time.time()

# #q = cancer_infer.query(variables=['Pollution'])
# # print q['Cancer']
# print q['Smoker']
# print q['Cancer']
#print q['all']

a = q['all']
print a
print a.variables
#print a.values
#print end - start
# from pgmpy.factors.discrete import State
# from pgmpy.sampling import BayesianModelSampling

# inference = BayesianModelSampling(cancer_model)
# #evidence = []
# evidence = [State('Cancer',0), State('Dyspnoea', 1)]
# # samples = inference.likelihood_weighted_sample(evidence, 1000)
# samples = inference.rejection_sample(evidence, 100)
# print samples['Smoker'].sum()/ 100.0


# from pgmpy.sampling import GibbsSampling
# gibbs = GibbsSampling(cancer_model)
# samples = gibbs.sample(size = 1000)
# print samples['Pollution'].sum()/ 1000.0
