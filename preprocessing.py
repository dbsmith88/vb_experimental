from sklearn import linear_model
import networkx as nx
import numpy as np
import pandas as pd
import copy
import os


class DAGFunctions:

    @staticmethod
    def add(df, c1, c2):
        """
        :param df: Pandas DataFrame
        :param c1: Column 1 Name
        :param c2: Column 2 Name
        :return: df with c1+c2 column added
        """
        data1 = df[[c1]].to_numpy()
        data2 = df[[c2]].to_numpy()
        result = (data1 + data2).flatten()
        df.insert(df.columns.size, c1+"+"+c2, result, True)
        return df

    @staticmethod
    def subtract(df, c1, c2):
        """
        :param df: Pandas DataFrame
        :param c1: Column 1 Name
        :param c2: Column 2 Name
        :return: df with c1-c2 column added
        """
        data1 = df[[c1]].to_numpy()
        data2 = df[[c2]].to_numpy()
        result = (data1 - data2).flatten()
        df.insert(df.columns.size, c1+"-"+c2, result, True)
        return df

    @staticmethod
    def absolute(df, c):
        """
        :param df: Pandas DataFrame
        :param c: Column Name
        :return: df with c absolute value column added
        """
        data = df[[c]].to_numpy()
        result = np.abs(data).flatten()
        df.insert(df.columns.size, c+"a", result, True)
        return df

    @staticmethod
    def multiply(df, c, scalar):
        """
        :param df: Pandas DataFrame
        :param c: Column Name
        :param scalar: Scalar value to multiple with column
        :return: df with c column multiplied by scalar
        """
        data = df[[c]].to_numpy()
        result = (scalar * data).flatten()
        df.insert(df.columns.size, c+"*s", result, True)
        return df

    @staticmethod
    def normalize(df, c):
        """
        :param df: Pandas DataFrame
        :param c: Column Name
        :return: df with c normalized column added
        """
        data = df[[c]].to_numpy()
        norm = np.linalg.norm(data)
        result = (data/norm).flatten()
        df.insert(df.columns.size, c+"n", result, True)
        return df

    @staticmethod
    def negate(df, c):
        """
        :param df: Pandas DataFrame
        :param c: Column Name
        :return: df with c negation column added
        """
        data = df[[c]].to_numpy()
        result = data * (-1.0)
        df.insert(df.columns.size, "neg("+c+")", result, True)
        return df

    @staticmethod
    def square(df, c):
        """
        :param df: Pandas DataFrame
        :param c: Column Name
        :return: df with c squared column added
        """
        data = df[[c]].to_numpy().flatten()
        coef = np.zeros(len(data))
        coef[data > 0.0] = 1.0
        coef[data < 0.0] = -1.0
        result = (coef * np.square(data))
        df.insert(df.columns.size, "("+c+")^2", result, True)
        return df

    @staticmethod
    def power(df, c, p):
        """
        :param df: Panda DataFrame
        :param c: Column Name
        :param p: raise column to power p
        :return: df with c column raised to power p
        """
        data = df[[c]].to_numpy().flatten()
        exponent = np.empty(len(data))
        exponent.fill(p)
        result = np.power(data, exponent)
        df.insert(df.columns.size, "("+c+")^" + str(p), result, True)
        return df

    @staticmethod
    def squareroot(df, c):
        """
        :param df: Pandas DataFrame
        :param c: Column Name
        :return: df with c square root value column added
        """
        data = df[[c]].to_numpy().flatten()
        coef = np.zeros(len(data))
        coef[data > 0.0] = 1.0
        coef[data < 0.0] = -1.0
        data = np.abs(data)
        result = np.multiply(coef, np.sqrt(data))
        df.insert(df.columns.size, "("+c+")^1/2", result, True)
        return df

    @staticmethod
    def log(df, c):
        """
        :param df: Pandas DataFrame
        :param c: Column Name
        :return: df with natural log(c) column added
        """
        data = df[[c]].to_numpy().flatten()
        coef = np.zeros(len(data))
        coef[data > 0.0] = 1.0
        coef[data < 0.0] = -1.0
        data = np.abs(data)
        data[data < 1.0] = 1.0
        result = np.multiply(coef, np.log(data))
        df.insert(df.columns.size, "ln("+c+")", result, True)
        return df

    @staticmethod
    def log10(df, c):
        """
        :param df: Pandas DataFrame
        :param c: Column Name
        :return: df with log10(c) column added
        """
        data = df[[c]].to_numpy().flatten()
        coef = np.zeros(len(data))
        coef[data > 0.0] = 1.0
        coef[data < 0.0] = -1.0
        data = np.abs(data)
        data[data < 1.0] = 1.0
        result = np.multiply(coef, np.log10(data))
        df.insert(df.columns.size, "log10("+c+")", result, True)
        return df

    @staticmethod
    def reciprocal(df, c):
        """
        :param df: Pandas DataFrame
        :param c: Column Name
        :return: df with reciprocal value, 1/x, column added
        """
        data = df[[c]].to_numpy()
        result = np.reciprocal(data).flatten()
        df.insert(df.columns.size, "recip("+c+")", result, True)
        return df

    @staticmethod
    def product(df, c, r):
        """
        :param df: Pandas DataFrame
        :param c: Column Name list
        :param r: Response column in df
        :return: Product of the c[0] and c[1] terms in the df appended to the df
        """
        if len(c) < 2:
            return df
        if r in c:
            c.remove(r)
        x = df[c].to_numpy()
        x0 = x[:, 0]
        x1 = x[:, 1]
        p_x = np.multiply(x0, x1)
        df["IP("+c[0]+c[1]+")"] = p_x
        return df

    @staticmethod
    def polynomial(df, r, c, signed=True):
        """
        :param df: Pandas DataFrame
        :param r: Response Column Name
        :param c: Column Name list
        :param signed: carry the sign of the squared term
        :return: df with polynomial transformation of c
        """
        y = df[[r]].to_numpy().flatten()
        if type(c) is not list:
            c = [c]
        if r in c:
            c.remove(r)
        x = df[c].to_numpy()
        if len(x) == 1:
            x = x.reshape(-1, 1)
        if signed:
            xp = np.column_stack((x, np.square(x) * np.sign(x)))
        else:
            xp = np.column_stack((x, np.square(x)))
        lg = linear_model.LinearRegression(fit_intercept=True, normalize=False)
        s = 0
        n = x.shape[0]
        lg.fit(xp[s:n], y[s:n])
        coef = lg.coef_
        intercept = lg.intercept_
        result = intercept + (coef[0] * xp[:, 0]) + (coef[1] * xp[:, 1])
        df["P(" + c[0] + ")"] = result
        return df


class PPNode:
    parameters = None
    function = None

    def __init__(self, f, p):
        self.function = f
        self.parameters = p

    def execute(self, df):
        self.parameters["df"] = df
        results = getattr(DAGFunctions, self.function)(**self.parameters)
        return results


class PPGraph:

    def __init__(self, data, parameters):
        self.graph = nx.DiGraph()
        self.data = copy.copy(data)
        self.parameters = parameters
        self.generate_graph()
        self.traverse()

    def generate_graph(self):
        for k, v in self.parameters["nodes"].items():
            self.graph.add_node(k, data=PPNode(v["function"], v["args"]))
        for e in self.parameters["edges"]:
            self.graph.add_edge(e[0], e[1])
        # pos = nx.spring_layout(self.graph)
        # nx.draw_networkx_nodes(self.graph, pos, node_size=700)
        # nx.draw_networkx_edges(self.graph, pos, edgelist=self.parameters["edges"], width=3)
        # nx.draw_networkx_labels(self.graph, pos, font_size=20)
        # plt.show()

    def traverse(self):
        order = list(nx.topological_sort(self.graph))
        for o in order:
            n = self.graph.nodes[o]
            self.data = n['data'].execute(self.data)


if __name__ == "__main__":
    _raw_data = pd.read_excel(os.path.join("data", "VB_Data_1a.xlsx"))                  # Data source

    # two inputs for the request: csv data and json configuration for preprocessing
    input_parameters = {
        'nodes': {
            # 1: {'function': 'polynomial', 'args': {'c': ['x1', 'x2'], 'r': 'Response'}},
            1: {'function': 'polynomial', 'args': {'c': ['x1'], 'r': 'Response'}},
            # 1: {'function': 'add', 'args': {'c1': 'x1', 'c2': 'x3'}},
            # 2: {'function': 'square', 'args': {'c': 'x3'}},
            # 3: {'function': 'square', 'args': {'c': 'x1+x3'}},
            # 4: {'function': 'normalize', 'args': {'c': 'x5'}},
        },
        'edges': []
    }

    pp_data = PPGraph(_raw_data, input_parameters).data
    print(pp_data.to_string())
