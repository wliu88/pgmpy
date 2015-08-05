import numpy as np
from scipy import stats
from scipy import integrate

from pgmpy.estimators import BaseEstimator
from pgmpy.factors import TabularCPD
from pgmpy.models import BayesianModel


class BayesianEstimator(BaseEstimator):
    """
    Class used to compute parameters for a model using Bayesian Estimate.

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

    @staticmethod
    def _integral_function(theta, prior, state_count, number_of_states):
        return (theta ** state_count) * prior.pdf([])

    def get_parameters(self, prior='dirichlet', **kwargs):
        """
        Method for getting all the learned CPDs of the model.

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
        if prior == 'dirichlet' and 'alpha' not in kwargs:
            alpha = {node: [1] * (self.node_card[node] * (np.product([self.node_card[_node]
                                                                      for _node in self.model.predecessors(node)])
                                                          if self.model.predecessors(node) else 1))
                     for node in self.model.nodes()}
        else:
            alpha = kwargs['alpha']

        parameters = []

        for node in self.model.nodes():
            if prior == 'dirichlet':
                parents = self.model.get_parents(node)
                if not parents:
                    state_counts = self.data.ix[:, node].value_counts()
                    node_alpha = np.array(alpha[node])

                    values = (state_counts.values + node_alpha) / (state_counts.values.sum() + node_alpha.sum())
                    cpd = TabularCPD(node, self.node_card[node], values[:, np.newaxis])
                    cpd.normalize()
                    parameters.append(cpd)
                else:
                    parent_card = np.array([self.node_card[parent] for parent in parents])
                    var_card = self.node_card[node]
                    state_counts = (self.data.groupby([node] + self.model.predecessors(node)).size()).values
                    node_alpha = np.array(alpha[node])

                    values = (state_counts.values + node_alpha) / (state_counts.values.sum() + node_alpha.sum())
                    values = values.reshape(var_card, np.product(parent_card))
                    cpd = TabularCPD(node, var_card, values,
                                     evidence=parents,
                                     evidence_card=parent_card.astype('int'))
                    cpd.normalize()
                    parameters.append(cpd)

        return parameters
