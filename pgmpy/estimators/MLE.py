from itertools import combinations

import numpy as np
from scipy import stats

from pgmpy.estimators import BaseEstimator
from pgmpy.factors import TabularCPD
from pgmpy.models import BayesianModel


class MaximumLikelihoodEstimator(BaseEstimator):
    """
    Class used to compute parameters for a model using Maximum Likelihood Estimate.

    Parameters
    ----------
    model: pgmpy.models.BayesianModel or pgmpy.models.MarkovModel or pgmpy.models.NoisyOrModel
        model for which parameter estimation is to be done

    data: pandas DataFrame object
        datafame object with column names same as the variable names of the network

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.models import BayesianModel
    >>> from pgmpy.estimators import MaximumLikelihoodEstimator
    >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
    ...                       columns=['A', 'B', 'C', 'D', 'E'])
    >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
    >>> estimator = MaximumLikelihoodEstimator(model, values)
    """
    def __init__(self, model, data):
        if not isinstance(model, BayesianModel):
            raise NotImplementedError("Maximum Likelihood Estimate is only implemented of BayesianModel")

        super().__init__(model, data)

    def get_parameters(self):
        """
        Method used to get parameters.

        Returns
        -------
        parameters: list
            List containing all the parameters. For Bayesian Model it would be list of CPDs'
            for Markov Model it would be a list of factors

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> estimator = MaximumLikelihoodEstimator(model, values)
        >>> estimator.get_parameters()
        """
        parameters = []

        for node in self.model.nodes():
            parents = self.model.get_parents(node)
            if not parents:
                state_counts = self.data.ix[:, node].value_counts()
                cpd = TabularCPD(node, self.node_card[node],
                                 state_counts.values[:, np.newaxis])
                cpd.normalize()
                parameters.append(cpd)
            else:
                parent_card = np.array([self.node_card[parent] for parent in parents])
                var_card = self.node_card[node]
                state_counts = self.data.groupby([node] + self.model.predecessors(node)).size()
                values = state_counts.values.reshape(var_card, np.product(parent_card))
                cpd = TabularCPD(node, var_card, values,
                                 evidence=parents,
                                 evidence_card=parent_card.astype('int'))
                cpd.normalize()
                parameters.append(cpd)

        return parameters

    def get_model(self, threshold=0.95):
        nodes = self.data.columns
        self.model.add_nodes_from(nodes)
        edges = []
        for u, v in combinations(nodes, 2):
            f_exp = self.data.groupby([u, v]).size().values
            u_f_obs = self.data.ix[:, u].value_counts().values
            v_f_obs = self.data.ix[:, v].value_counts().values
            if stats.chisquare(f_obs=[i * j for i in u_f_obs for j in v_f_obs], f_exp=f_exp) < threshold:
                edges.append((u, v))
        self.model.add_edges_from(edges)
        return nodes, edges
