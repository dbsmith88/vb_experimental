import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import sklearn.metrics as skm
import sklearn as sk
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
import statsmodels.api as sm
import statsmodels.stats.outliers_influence as smo
import scipy.stats


class AnalyticalModel:

    def __init__(self, data: pd.DataFrame, target: str, training_split=0.8, one_out=False, model_config=None):
        self.target = target
        self.data = data
        self.attribute_data = self.data.drop(target, axis=1)
        self.target_data = self.data[[target]]
        self.attributes = self.attribute_data.columns
        self.one_out = one_out
        self.training_split = training_split
        self.train_n = int(self.data.shape[0] * self.training_split) if not self.one_out else int(self.data.shape[0])
        # self.train_n = int(self.data.shape[0])
        self.model = None
        self.model_configs = model_config
        self.results = None
        self.confusion = None
        self.coef = None
        self.r2 = None
        self.r2_adjusted = None
        self.mse = None
        self.rmse = None
        self.anderson = None
        self.anderson_p = None
        self.residuals = None
        self.predictions = None
        self.aic = None
        self.aaic = None
        self.bic = None
        self.eval = None
        self.build_model()

    def build_mlr(self, train_x, train_y, test_x, test_y, params):
        """
        Build, fit and predict with a multiple linear regression model.
        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :param params:
        :return:
        """
        self.model = sk.linear_model.LinearRegression(fit_intercept=True, **params)
        self.results = self.model.fit(train_x, train_y)
        pred_y = self.results.predict(test_x)
        self.predictions = pred_y
        test_y = test_y.to_numpy().flatten()
        self.coef = self.results.coef_
        res = test_y - pred_y
        self.residuals = res

    def build_linear_svr(self, train_x, train_y, test_x, test_y, params):
        """
        Build, fit and predict with a Linear Support Vector Regressor
        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :param params:
        :return:
        """
        n_train_x = sk.preprocessing.scale(train_x, axis=1)
        self.model = sk.svm.LinearSVR(fit_intercept=True, loss='squared_epsilon_insensitive', random_state=0, **params)
        self.results = self.model.fit(n_train_x, train_y)
        pred_y = self.results.predict(test_x)
        self.predictions = pred_y
        test_y = test_y.to_numpy().flatten()
        self.coef = self.results.coef_
        res = test_y - pred_y
        self.residuals = res

    def build_gbr(self, train_x, train_y, test_x, test_y, params):
        """
        Build, fit and predict with a Gradient Boost Regressor
        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :param params:
        :return:
        """
        self.model = GradientBoostingRegressor(random_state=0, **params)
        self.results = self.model.fit(train_x, train_y)
        pred_y = self.results.predict(test_x)
        self.predictions = pred_y
        test_y = test_y.to_numpy().flatten()
        self.coef = None
        res = test_y - pred_y
        self.residuals = res

    def build_elastic_net(self, train_x, train_y, test_x, test_y, params):
        """
        Build, fit and predict with an Elastic Net CV
        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :param params:
        :return:
        """
        self.model = sk.linear_model.ElasticNetCV(**params)
        self.results = self.model.fit(train_x, train_y)
        pred_y = self.results.predict(test_x)
        self.predictions = pred_y
        test_y = test_y.to_numpy().flatten()
        self.coef = self.results.coef_
        res = test_y - pred_y
        self.residuals = res

    def build_rfr(self, train_x, train_y, test_x, test_y, params):
        """
        Build, fit and predict with a Random Forest Regressor
        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :param params:
        :return:
        """
        self.model = RandomForestRegressor(random_state=0, **params)
        self.results = self.model.fit(train_x, train_y)
        pred_y = self.results.predict(test_x)
        self.predictions = pred_y
        test_y = test_y.to_numpy().flatten()
        self.coef = None
        res = test_y - pred_y
        self.residuals = res

    def build_stacker(self, train_x, train_y, test_x, test_y, params):
        """
        Build, fit and predict with a stacking regressor ensemble.
        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :param params:
        :return:
        """
        # n_train_x = sk.preprocessing.scale(train_x, axis=1)
        if "estimators" in params.keys():
            estimators = []
            for e in params["estimators"]:
                # example estimator would be 'linear_model.RidgeCV', where the group and type must match the scikit-learn model
                sm = e.split(".")
                estimator = (sm[1], getattr(getattr(sk, sm[0]), sm[1]))
                estimators.append(estimator)
        else:
            estimators = [
                ('lr', sk.linear_model.LinearRegression()),
                # ('svr', sk.svm.LinearSVR(random_state=42)),
                ('enet', sk.linear_model.ElasticNetCV()),
                ('ridge', sk.linear_model.RidgeCV())
            ]
        self.model = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(random_state=42),  passthrough=False, n_jobs=-1)
        self.results = self.model.fit(train_x, train_y)
        pred_y = self.results.predict(test_x)
        self.predictions = pred_y
        test_y = test_y.to_numpy().flatten()
        self.coef = None
        res = test_y - pred_y
        self.residuals = res

    def build_model(self, weights=None):
        test_n = self.train_n if not self.one_out else 0
        test_m = self.data.shape[0] if not self.one_out else self.data.shape[0]
        train_data = self.data[0:self.train_n]
        test_data = self.data[test_n:test_m]
        train_x = train_data[self.attributes]
        train_y = train_data[[self.target]]
        test_x = test_data[self.attributes]
        test_y = test_data[[self.target]]
        weights = np.ones(train_x.shape[0]) if weights is None else weights

        y = train_y.to_numpy().flatten()
        x = train_x.to_numpy()
        model_configs = {} if self.model_configs is None else self.model_configs
        if "type" in model_configs.keys():
            params = model_configs["params"] if "params" in model_configs.keys() else {}
            if model_configs["type"] == "MLR":
                self.build_mlr(x, y, test_x, test_y, params)
            elif model_configs["type"] == "LinearSVR":
                self.build_linear_svr(x, y, test_x, test_y, params)
            elif model_configs["type"] == "GBR":
                self.build_gbr(x, y, test_x, test_y, params)
            elif model_configs["type"] == "RFR":
                self.build_rfr(x, y, test_x, test_y, params)
            elif model_configs["type"] == "ElasticNetCV":
                self.build_elastic_net(x, y, test_x, test_y, params)
            elif model_configs["type"] == "Stacker":
                self.build_stacker(x, y, test_x, test_y, params)
            else:
                self.build_mlr(x, y, test_x, test_y, params)
        else:
            model_configs["type"] = "MLR"
            self.model_configs = model_configs
            self.build_mlr(x, y, test_x, test_y, {})

        n = float(self.data.shape[0])
        p = float(self.data.shape[1] - 1.)
        sse = np.sum(np.power(self.residuals, 2))
        sst = np.sum(np.power(test_y - np.mean(test_y), 2))
        self.r2 = ((sst - sse) / sst).round(4)
        self.r2_adjusted = (self.r2 - (1. - self.r2) * 2. / (n - 3.)).round(4)
        self.rmse = (np.sqrt(sse / (n - p - 1.))).round(4)
        self.mse = (np.power(self.rmse, 2)).round(4)
        self.aic = (n * np.log(sse / n) + (2. * p) + n + 2.).round(4)
        self.aaic = (self.aic + (2. * (p + 1.) * (p + 2.))/(n - p - 2.)).round(4)
        self.bic = ((n * np.log(sse/n)) + (p * np.log(n))).round(4)
        self.results.aic = self.aic
        self.results.bic = self.bic

        self.anderson = scipy.stats.anderson(self.residuals)
        self.anderson_pvalue(replicate=True)

    def anderson_pvalue(self, replicate=True):
        ad = self.anderson.statistic
        if replicate:
            if ad < 2:
                p = 1. - np.exp(-1.2337141/ad) / np.sqrt(ad) * (2.00012+(.247105-(.0649821-(.0347962-(.011672-.00168691*ad)*ad)*ad)*ad)*ad)
            else:
                p = 1. - np.exp(-1.*np.exp(1.0776-(2.30695-(.43424-(.082433-(.008056 -.0003146*ad)*ad)*ad)*ad)*ad))
        else:
            # https://www.spcforexcel.com/knowledge/basic-statistics/anderson-darling-test-for-normality
            ad = ad * (1. + (.75/50.) + 2.25/(50.**2))
            if ad >= 0.6:
                p = 1. - np.exp(1.2937 - 5.709*ad + 0.0186*(ad**2))
            elif 0.34 < ad < 0.6:
                p = 1. - np.exp(0.9177 - 4.279*ad - 1.38*(ad**2))
            elif 0.2 < ad < 0.34:
                p = 1.0 - np.exp(-8.318 + 42.796*ad - 59.938*(ad**2))
            else:
                p = 1.0 - np.exp(-13.436 + 101.14*ad - 223.73*(ad**2))
        self.anderson_p = p

    def evaluate_VIF(self, threshold=5.0):
        valid = True
        subset = self.data[list(self.attributes)]
        if len(self.attributes) > 1:
            for i in range(0, len(self.attributes)):
                subset_data = subset.drop(self.attributes[i], axis=1)
                mod = sm.OLS(subset[self.attributes[i]], sm.add_constant(subset_data))
                res = mod.fit()
                vif2 = 1. / (1. - res.rsquared)
                if vif2 > threshold:
                    valid = False
                    break
        if valid:
            return True
        else:
            return False

    def evaluate(self, use="rmse", ad=True, check_VIF=False, exclude=True):
        use = use.lower()
        self.eval = use
        if use == "r2":
            metric = abs(self.r2) - 1.0
        elif use == "r2a":
            metric = abs(self.results.rsquared_adj) - 1.0
        elif use == "rmse":
            metric = self.rmse
        elif use == "press":
            r = smo.OLSInfluence(self.results)
            metric = r.ess_press
        elif use == "aic":
            metric = self.aic
        elif use == "caic":
            k = self.data.shape[1] - 1
            n = self.results.nobs
            metric = self.aic + ((2*(k*k) + 2*k)/(n - k - 1))
        elif use == "bic":
            metric = self.bic
        else:
            metric = self.mse
        if ad:
            if self.anderson_p < 0.05:
                if exclude:
                    metric = float("inf")
                else:
                    metric = 10000
        if check_VIF:
            if not self.evaluate_VIF():
                if exclude:
                    metric = float("inf")
                else:
                    metric = 10000      # Allows for model to still be on the list but will let better models get added.
        return metric

    def plot_results(self):
        test_data = self.data[self.train_n:] if not self.one_out else self.data[0:self.train_n]
        test_y = test_data[[self.target]]
        pred_y = self.predictions

        plt.subplot(2, 1, 1)
        plt.title("{} Model Results ({}: {}) \n Attributes: {}".format(
            self.model_configs["type"], self.eval, getattr(self, self.eval), ", ".join(list(self.attributes)))
        )
        plot_x = np.arange(0, self.residuals.shape[0])
        plt.scatter(plot_x, test_y, color='gray', linewidth=1)
        plt.scatter(plot_x, pred_y, color='red', linewidth=1)
        plt.ylabel("Prediction/Actual")
        plt.axhline(y=np.mean(pred_y), linewidth=0.5, color='black')
        red_patch = mpatches.Patch(color='red', label='Prediction')
        gray_patch = mpatches.Patch(color='gray', label='Actual')
        plt.legend(handles=[gray_patch, red_patch])

        plt.subplot(2, 1, 2)
        plt.scatter(pred_y, self.residuals, facecolors='none', edgecolors='blue')
        plt.axhline(linewidth=0.5, color='black')
        plt.ylabel("Fitted vs Residuals")
        plt.show()

    def print_summary(self):
        test_data = self.data[self.train_n:] if not self.one_out else self.data[0:self.train_n]
        test_y = test_data[[self.target]]
        pred_y = self.predictions
        max_error = skm.max_error(test_y, pred_y)
        mean_absolute_error = skm.mean_absolute_error(test_y, pred_y)
        median_absolute_error = skm.median_absolute_error(test_y, pred_y)
        print("\n----------------- Model Summary ----------------")
        print("Type: {}\t\tEvaluation Criteria: {}".format(self.model_configs["type"], self.eval).expandtabs(15))
        print("Response: {}\t\tAttributes: {}".format(self.target, ", ".join(list(self.attributes))).expandtabs(15))
        print("Total Data Records: {}\t\tTraining Data Split: {}".format(self.data.shape[0], self.training_split).expandtabs(15))
        print("Total Training Records: {}\t\tTotal Testing Records: {}".format(self.train_n, test_data.shape[0]).expandtabs(15))
        print("R Squared: {}\t\tMean Squared Error: {}\t\tRoot Mean Squared Error: {}".format(round(self.r2,4), round(self.mse,4), round(self.rmse,4)).expandtabs(15))
        print("Max Error: {}\t\tMean Absolute Error: {}\t\tMedian Absolute Error: {}".format(
            round(max_error, 4), round(mean_absolute_error,4), round(median_absolute_error,4)).expandtabs(15))

    def print_summary2(self):
        print(self.results.summary())
