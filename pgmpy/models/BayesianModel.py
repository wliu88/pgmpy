def fit(self, data, estimator=None):
        """
        Computes the CPD for each node from a given data in the form of a pandas dataframe.

        Parameters
        ----------
        data : pandas DataFrame object
            A DataFrame object with column names same as the variable names of network

        estimator: Estimator class
            Any pgmpy estimator. If nothing is specified, the default Maximum Likelihood
            estimator would be used

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> model.get_cpds()
        [<pgmpy.factors.CPD.TabularCPD at 0x7fd173b2e588>,
         <pgmpy.factors.CPD.TabularCPD at 0x7fd173cb5e10>,
         <pgmpy.factors.CPD.TabularCPD at 0x7fd173b2e470>,
         <pgmpy.factors.CPD.TabularCPD at 0x7fd173b2e198>,
         <pgmpy.factors.CPD.TabularCPD at 0x7fd173b2e2e8>]
        """

        from pgmpy.estimators import MaximumLikelihoodEstimator, BaseEstimator, BayesianEstimator

        if estimator is None or estimator == 'mle':
            estimator_type = MaximumLikelihoodEstimator
        elif estimator == 'bayes':
            estimator_type = BayesianEstimator

        estimator = estimator_type(self, data)
        if not isinstance(estimator, BaseEstimator):
            raise TypeError("Estimator object should be a valid pgmpy estimator.")

        if not self.nodes():
            nodes, edges = estimator.get_model()
            self.add_nodes_from(nodes)
            self.add_edges_from(edges)

        cpds_list = estimator.get_parameters()
        self.add_cpds(*cpds_list)

    def predict(self, data):
        """
        Predicts states of all the missing variables.

        Parameters
        ----------
        data : pandas DataFrame object
            A DataFrame object with column names same as the variables in the model.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> train_data = values[:800]
        >>> predict_data = values[800:]
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> predict_data = predict_data.copy()
        >>> predict_data.drop('E', axis=1, inplace=True)
        >>> y_pred = model.predict(predict_data)
        >>> y_pred
        array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1,
               1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1,
               1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
               1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
               1, 1, 1, 0, 0, 0, 1, 0])
        """
        from pgmpy.inference import VariableElimination

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("data has variables which are not in the model")

        missing_variables = set(self.nodes()) - set(data.columns)
        pred_values = defaultdict(list)

        model_inference = VariableElimination(self)
        for index, data_point in data.iterrows():
            states_dict = model_inference.map_query(variables=missing_variables, evidence=data_point.to_dict())
            for k, v in states_dict.items():
                pred_values[k].append(v)
        return pd.DataFrame(pred_values, index=data.index)

    def get_factorized_product(self, latex=False):
        # TODO: refer to IMap class for explanation why this is not implemented.
        pass

    def is_iequivalent(self, model):
        pass

    def is_imap(self, independence):
        pass
