from preprocessing import PPGraph
from sorted_heap import SortedHeap
from genetic_algorithm import GeneticAlgorithm
from analytical_model import AnalyticalModel
import statsmodels.api as sm
import multiprocessing as mp
import pandas as pd
import numpy as np
import scipy.stats
import itertools
import copy
import time
import os
import random


use_parallel = True


class AttributeTransformation:
    default_transformation = {
        'nodes': {
            1: {'function': 'log', 'args': {'c': 'x1'}},
            2: {'function': 'log10', 'args': {'c': 'x1'}},
            3: {'function': 'square', 'args': {'c': 'x1'}},
            6: {'function': 'reciprocal', 'args': {'c': 'x1'}},
            7: {'function': 'polynomial', 'args': {'c': ['x1'], 'r': 'Response'}},
            8: {'function': 'power', 'args': {'c': 'x1', 'p': 3}},
            9: {'function': 'squareroot', 'args': {'c': 'x1'}},
        },
        'edges': []
    }

    def __init__(self, data, response_data):
        self._data = copy.copy(data)
        if type(self._data) is not pd.DataFrame:
            self._data = self._data.to_frame()
        self._data["Response"] = response_data.values
        self.transformed_data = {}
        for n, v in self.default_transformation["nodes"].items():
            v['args']['c'] = data.name
            if 'r' in v['args'].keys():
                v['args']['r'] = response_data.columns.values.tolist()[0]
        self.data = PPGraph(self._data, self.default_transformation).data
        self.columns = self.data.columns
        d1 = response_data.to_numpy().flatten()
        self.data = self.data.drop("Response", axis=1)
        for f in self.data.columns:
            if not np.any(np.isnan(self.data[[f]])):
                d2 = self.data[[f]].to_numpy().flatten()
                if d2.min() == d2.max():            # d2 is constant
                    pearsons = [np.nan, np.nan]
                else:
                    pearsons = scipy.stats.pearsonr(d1, d2)
                self.transformed_data[f] = pearsons[0]

    def evaluate(self, transformation_threshold):
        best_pearson = 0
        best_transform = None
        for t, v in self.transformed_data.items():
            if abs(v) > best_pearson:
                best_pearson = abs(v)
                best_transform = t
        if best_transform != self.columns[0]:
            if best_pearson / abs(self.transformed_data[self.columns[0]]) > (1. + transformation_threshold):
                return [
                    best_transform,
                    self.transformed_data[best_transform],
                    self.data[best_transform]
                ]
        return [
            self.columns[0],
            self.transformed_data[self.columns[0]],
            self.data[self.columns[0]]
        ]


class AttributeInteraction:

    interaction_polynomial = {
        'nodes': {
            1: {'function': 'product', 'args': {'c': ['x1', 'x2'], 'r': 'Response'}},
        },
        'edges': []
    }

    def __init__(self, data, response_data):
        self._data = copy.copy(data)
        self._data["Response"] = copy.copy(response_data.values)
        self._columns = self._data.columns
        self.transformed_data = {}

        self.interaction_polynomial["nodes"][1]["args"]["c"] = list(self._columns)
        self.data = PPGraph(self._data, self.interaction_polynomial).data
        self.columns = self.data.columns
        self.interaction_column = [c for c in list(self.columns) if "IP" in c][0]
        d1 = response_data.to_numpy().flatten()
        d2 = self.data[self.interaction_column].to_numpy().flatten()
        pearsons = scipy.stats.pearsonr(d1, d2)
        self.transformed_data[self.interaction_column] = round(pearsons[0], 5)

    def evaluate(self):
        return [
            self.interaction_column,
            self.transformed_data[self.interaction_column],
            self.data[self.interaction_column]
        ]


parallel_results = []


class AutomatedModelBuilder:
    pearsons_threshold = 0.1
    pearsons_change_threshold = 0.0
    vif_threshold = 5.0
    training_split = 0.8
    data_attribute_ratio = 10
    parallel_threshold = 10000

    def __init__(
            self,
            data,
            response,
            genetic=False,
            evaluation="rmse",
            skip_transformations=False,
            skip_interactions=False,
            one_out=False,
            model_config=None,
            ga_config=None
    ):
        print("Starting automated multiple linear regression")
        self.response = response
        self.data = data
        self.evaluation = evaluation
        self.response_data = copy.copy(self.data[[response]])
        self.attribute_data = copy.copy(self.data).drop(response, axis=1)
        self.interaction_data = pd.DataFrame()
        self.one_out = one_out
        self.attribute_transformations = None
        self.valid_transformations = None
        self.valid_attributes = None
        self.model_config = model_config
        self.ga_config = ga_config
        if not skip_interactions:
            self.process_interactions()
            self.attribute_data = pd.concat([self.attribute_data, self.interaction_data], axis=1)
        self.process_attributes(bypass=skip_transformations)           # step 1: Generate all transformation, validate transformations
        self.all_combinations = []
        self.best_models = SortedHeap(n=10, target=0.0)
        # step 2: Generate all combinations from valid/best transformations
        print("{} total attributes.".format(len(self.valid_attributes)))
        # self.attribute_data.to_csv(os.path.join("data", "attribute_data.csv"))
        # self.data.to_csv(os.path.join("data", "transformed_data.csv"))
        # if self.valid_attributes.shape[1] > 15 or genetic:     # 5 base variables
        if genetic:
            # use genetic algorithm
            print("Genetic algorithm flag active.")
            self.run_genetic_algorithm()
        else:
            for i in range(1, int(self.data.shape[0]/self.data_attribute_ratio)):
                self.all_combinations += list(itertools.combinations(list(self.valid_attributes.columns), i))
            self.valid_combinations = []
            print("Validating all {} potential attribute combinations...".format(len(self.all_combinations)))
            global use_parallel
            if len(self.all_combinations) >= self.parallel_threshold:
                use_parallel = True
                print("Potential attribute combinations above {}, turning parallel functions on.".format(self.parallel_threshold))
            else:
                use_parallel = False
            if use_parallel:
                self.validate_combinations_parallel()               # step 3: Validate all combinations in parallel
            else:
                self.validate_combinations()                        # step 3: Validate all combinations
            print("Found {} valid combinations".format(len(self.valid_combinations)))
            self.results = []
            print("Building models...")
            self.build_models()                                     # step 4: Build models and validate results

    def process_attributes(self, bypass=False):
        if bypass:
            self.valid_attributes = self.attribute_data
            self.valid_transformations = []
            for ad in list(self.attribute_data.columns):
                self.valid_transformations.append([ad, np.nan, self.data[ad]])
            return
        # step 1.a Perform all possible transformations on attribute data
        print("Calculating attribute transformations...")
        self.attribute_transformations = [
            AttributeTransformation(self.attribute_data[a], self.response_data) for a in self.attribute_data.columns
        ]
        # step 1.b Evalute all attribute transformations for best selection
        print("Evaluating all attribute transformations...")
        self.valid_transformations = [
            t.evaluate(self.pearsons_change_threshold) for t in self.attribute_transformations
        ]
        self.valid_attributes = pd.DataFrame()
        for vt in self.valid_transformations:
            if vt is not None:
                if vt[0] not in self.data.columns:
                    self.data[vt[0]] = vt[2]
                self.valid_attributes[vt[0]] = vt[2]

    def process_interactions(self):
        print("Calculating attribute interactions...")
        attribute_combinations = list(itertools.combinations(self.attribute_data.columns, 2))
        print("Found {} attribute interactions".format(len(attribute_combinations)))
        attribute_interactions = [
            AttributeInteraction(self.attribute_data[list(a)], self.response_data) for a in attribute_combinations
        ]
        # step 1.b Evalute all attribute transformations for best selection
        print("Evaluating all attribute interactions...")
        valid_interactions = [
            t.evaluate() for t in attribute_interactions
        ]
        valid_interactions = list(filter(None, valid_interactions))
        for vt in valid_interactions:
            if vt is not None:
                if vt[0] not in self.interaction_data.columns:
                    self.interaction_data[vt[0]] = vt[2]
                if vt[0] not in self.data.columns:
                    self.data[vt[0]] = vt[2]
        del attribute_combinations
        del attribute_interactions
        del valid_interactions

    def validate_combinations_p(self, c):
        subset = self.data[list(c)]
        valid = True
        if len(c) > 1:
            for i in range(0, len(c)):
                subset_data = subset.drop(c[i], axis=1)
                mod = sm.OLS(subset[c[i]], sm.add_constant(subset_data))
                res = mod.fit()
                vif2 = 1. / (1. - res.rsquared)
                if vif2 > self.vif_threshold:
                    valid = False
                    break
        if valid:
            return c

    def validate_combinations_parallel(self):
        all_combinations = copy.copy(self.all_combinations)
        pool = mp.Pool(mp.cpu_count())
        parallel_results = pool.map(self.validate_combinations_p, [c for c in all_combinations])
        pool.close()
        pool.join()
        parallel_results = list(filter(None, parallel_results))
        self.valid_combinations = parallel_results

    def validate_combinations(self):
        for c in self.all_combinations:
            valid = True
            subset = self.data[list(c)]
            if len(c) > 1:
                for i in range(0, len(c)):
                    subset_data = subset.drop(c[i], axis=1)
                    mod = sm.OLS(subset[c[i]], sm.add_constant(subset_data))
                    res = mod.fit()
                    vif2 = 1./(1. - res.rsquared)
                    if vif2 > self.vif_threshold:
                        valid = False
                        break
            if valid:
                self.valid_combinations.append(list(c))

    def build_models_parallel(self, c, model_config):
        m_data = copy.copy(self.data[list(c)])
        m_data[self.response] = self.response_data.values
        m = AnalyticalModel(m_data, self.response, one_out=self.one_out, model_config=model_config)
        return m

    def build_models(self):
        if use_parallel:
            all_combinations = copy.copy(self.valid_combinations)
            pool = mp.Pool(mp.cpu_count())
            pool_results = [pool.apply_async(self.build_models_parallel, (c, self.model_config)) for c in all_combinations]
            parallel_results = []
            for p in pool_results:
                parallel_results.append(p.get())
            pool.terminate()
            pool.join()
            [self.best_models.add(m, m.evaluate(use=self.evaluation)) for m in parallel_results]
            self.results = self.best_models
        else:
            for c in self.valid_combinations:
                m_data = copy.copy(self.data[list(c)])
                m_data[self.response] = self.response_data.values
                m = AnalyticalModel(m_data, self.response, training_split=0.80, one_out=self.one_out, model_config=self.model_config)
                self.best_models.add(m, m.evaluate(use=self.evaluation))
            self.results = self.best_models
        if self.results.s_heap[0][1] is not None:
            self.results.s_heap[0][1].plot_results()
            self.results.s_heap[0][1].print_summary()
            self.results.print(self.evaluation)
        else:
            print("No valid models found or all model outputs failed evaluation.")

    def set_ga_config(self):
        if self.ga_config is None and use_parallel:
            self.ga_config = {
                "config":
                    {
                        "mutate_percent": 0.2,
                        "no_change_threshold": 10,
                        "population_size": 50
                    },
                "mutate_config":
                    {
                        "add_percent": 0.2,
                        "delete_percent": 0.2
                    }
            }
        else:
            self.ga_config = {
                "config":
                    {
                        "mutate_percent": 0.2,
                        "no_change_threshold": 20,
                        "population_size": 100
                    },
                "mutate_config":
                    {
                        "add_percent": 0.2,
                        "delete_percent": 0.2
                    }
            }

    def run_genetic_algorithm_parallel(self, i):
        validated_data = copy.copy(self.data[list(self.valid_attributes.columns)])
        validated_data[self.response] = self.response_data.values

        ga = GeneticAlgorithm(
            data=validated_data,
            response=self.response,
            attributes=list(self.valid_attributes.columns),
            evaluation=self.evaluation,
            seed=i,
            one_out=self.one_out,
            model_config=self.model_config,
            **self.ga_config["config"]
        )
        ga.execute(**self.ga_config["mutate_config"])
        return ga.best_models.s_heap

    def run_genetic_algorithm(self):
        use_parallel = False if len(self.valid_transformations) <= 21 else True
        # use_parallel = False
        self.set_ga_config()
        if use_parallel:
            concurrent_ga = mp.cpu_count()
            # concurrent_ga = 4
            pool = mp.Pool(concurrent_ga)
            pool_results = [pool.apply_async(self.run_genetic_algorithm_parallel, (i,)) for i in range(0, concurrent_ga)]
            parallel_results = []
            for r in pool_results:
                parallel_results.append(r.get())
            pool.terminate()
            pool.join()
            for m in parallel_results:
                for mi in m:
                    metric = mi[1].evaluate(use=self.evaluation)
                    self.best_models.add(mi[1], metric)
            self.results = self.best_models
        else:
            validated_data = copy.copy(self.data[list(self.valid_attributes.columns)])
            validated_data[self.response] = self.response_data.values
            ga = GeneticAlgorithm(
                data=validated_data,
                response=self.response,
                attributes=list(self.valid_attributes.columns),
                evaluation=self.evaluation,
                model_config=self.model_config,
                one_out=self.one_out,
                **self.ga_config["config"]
            )
            ga.execute(**self.ga_config["mutate_config"])
            self.results = ga.best_models
        if self.results.s_heap[0][1] is not None:
            self.results.s_heap[0][1].plot_results()
            self.results.s_heap[0][1].print_summary()
            self.results.print(self.evaluation)
        else:
            print("No valid models found or all model outputs failed evaluation.")


if __name__ == "__main__":
    t0 = time.time()

    # ------------ Data input and setup configuration ---------- #
    _raw_data = pd.read_excel(os.path.join("data", "VB_Data_1a.xlsx"))                  # Data source
    _target = 'Response'                                                                # Column in data source to use as target attribute
    _raw_data = _raw_data.drop("ID", axis=1)
    _raw_data = _raw_data.drop(["x5","x6","x7","x8","x9"], axis=1)
    # _raw_data = _raw_data.drop(["x6","x7","x8","x9"], axis=1)
    #_raw_data = _raw_data.drop(["x1","x2","x3"], axis=1)

    model_config = {"type": "GBR"}
    ga_config = {
        "config":
            {
                "mutate_percent": 0.2,
                "no_change_threshold": 10,
                "population_size": 50
            },
        "mutate_config":
            {
                "add_percent": 0.2,
                "delete_percent": 0.2
            }
    }
    amlr = AutomatedModelBuilder(
        _raw_data,
        _target,

        genetic=False,
        skip_transformations=False,
        skip_interactions=False,
        evaluation="rmse",
        one_out=False,
        model_config=model_config,
        ga_config=None
    )

    t1 = time.time() - t0
    print("Time elapsed: {} sec".format(round(t1, 3)))
