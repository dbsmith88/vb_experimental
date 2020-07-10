import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import sklearn.metrics as skm
import statsmodels.api as sm
import statsmodels.stats.outliers_influence as smo
import scipy.stats


class MultipleLinearRegression:

    def __init__(self, data: pd.DataFrame, target: str, training_split=0.8, one_out=False):
        self.target = target
        self.data = data
        self.attribute_data = self.data.drop(target, axis=1)
        self.target_data = self.data[[target]]
        self.attributes = self.attribute_data.columns
        self.one_out = one_out
        self.training_split = training_split
        self.train_n = int(self.data.shape[0] * self.training_split) if not one_out else int(self.data.shape[0])
        self.mlr = None
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
        self.build_model(replicate=one_out)

    def build_model(self, weights=None, args=None, replicate=False):
        test_n = self.train_n if not self.one_out else 0
        test_m = self.data.shape[0] if not self.one_out else self.data.shape[0]
        train_data = self.data[0:self.train_n]
        test_data = self.data[test_n:test_m]
        train_x = train_data[self.attributes]
        train_y = train_data[[self.target]]
        test_x = test_data[self.attributes]
        test_y = test_data[[self.target]]
        weights = np.ones(train_x.shape[0]) if weights is None else weights

        # ---- Statsmodel parameters ---- #
        self.mlr = sm.OLS(train_y, sm.add_constant(train_x))
        self.results = self.mlr.fit()
        pred_y = self.results.predict()
        self.predictions = pred_y
        test_y = test_y.to_numpy().flatten()
        self.residuals = self.results.resid.to_numpy()
        self.coef = self.results.params

        if replicate:
            pred_y = self.coef[0] + np.dot(train_x.to_numpy(), self.coef.to_numpy()[1:])
            test = np.sum(pred_y - self.predictions)
            res = test_y - pred_y
            self.residuals = res
            n = float(self.data.shape[0])
            p = float(self.data.shape[1] - 1.)
            # sse = np.dot(self.residuals, self.residuals)
            sse = self.results.ssr
            # mean_y = np.mean(test_y)
            # sst = np.dot(test_y - mean_y, test_y - mean_y)
            sst = self.results.centered_tss
            self.r2 = ((sst - sse) / sst).round(4)
            self.r2_adjusted = (self.r2 - (1. - self.r2) * 2. / (n - 3.)).round(4)
            self.rmse = (np.sqrt(sse / (n - p - 1.))).round(4)
            self.mse = (np.power(self.rmse, 2)).round(4)
            self.aic = (n * np.log(sse / n) + (2. * p) + n + 2.).round(4)
            self.aaic = (self.aic + (2. * (p + 1.) * (p + 2.))/(n - p - 2.)).round(4)
            self.bic = ((n * np.log(sse/n)) + (p * np.log(n))).round(4)
            self.results.aic = self.aic
            self.results.bic = self.bic
        else:
            # ---- Sci-kit Learn parameters ---- #
            # self.mlr = linear_model.LinearRegression(fit_intercept=True)
            # self.mlr.fit(train_x, train_y)
            # self.coef = self.mlr.coef_
            # pred_y = self.mlr.predict(test_x)
            # self.predictions = pred_y.flatten()
            # self.residuals = (test_y - pred_y).flatten()

            self.mse = skm.mean_squared_error(test_y, pred_y, squared=True)
            self.rmse = skm.mean_squared_error(test_y, pred_y, squared=False)
            self.r2 = skm.r2_score(test_y, pred_y)
            self.aic = self.results.aic
            self.bic = self.results.bic
        self.anderson = scipy.stats.anderson(self.residuals)
        self.anderson_pvalue(replicate=replicate)

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
                    metric = 10000      # Allows for model to still be on the list but will lets better models get added.
        return metric

    def plot_results(self):
        test_data = self.data[self.train_n:] if not self.one_out else self.data[0:self.train_n]
        test_y = test_data[[self.target]]
        pred_y = self.predictions

        plt.subplot(2, 1, 1)
        plt.title("Model Results")
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
        print("Response: {}\t\tAttributes: {}".format(self.target, ", ".join(list(self.attributes))).expandtabs(15))
        print("Total Data Records: {}\t\tTraining Data Split: {}".format(self.data.shape[0], self.training_split).expandtabs(15))
        print("Total Training Records: {}\t\tTotal Testing Records: {}".format(self.train_n, test_data.shape[0]).expandtabs(15))
        print("R Squared: {}\t\tMean Squared Error: {}\t\tRoot Mean Squared Error: {}".format(round(self.r2,4), round(self.mse,4), round(self.rmse,4)).expandtabs(15))
        print("Max Error: {}\t\tMean Absolute Error: {}\t\tMedian Absolute Error: {}".format(
            round(max_error, 4), round(mean_absolute_error,4), round(median_absolute_error,4)).expandtabs(15))

    def print_summary2(self):
        print(self.results.summary())
