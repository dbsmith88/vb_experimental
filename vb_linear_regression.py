from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from vb_helper import ShrinkBigKTransformer, None_T, LogP1_T
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter('ignore')


class LinearRegressionVB:

    def __init__(self):
        pass


class LinearRegressionAutomatedVB:
    name = "Linear Regression Automated"
    id = "lra"
    description = "Automated pipeline with feature evaluation and selection for a linear regression estimator."

    def __init__(self, test_split=0.2, cv_folds=10, cv_reps=10, seed=42, one_out=False):
        self.test_split = test_split
        self.cv_folds = cv_folds
        self.cv_reps = cv_reps
        self.seed = seed
        self.one_out = one_out

        self.k = None
        self.n = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.lr_estimator = None
        self.attr = None
        self.results = None
        self.residuals = None

    def set_data(self, x, y):
        if self.one_out:
            self.x_train, self.x_test, self.y_train, self.y_test = (x, x, y, y)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                x, y,
                test_size=self.test_split,
                random_state=self.seed
            )
        self.n, self.k = self.x_train.shape

    def set_pipeline(self):
        transformer_list = [None_T(), LogP1_T()]
        steps = [
            ('scaler', StandardScaler()),
            ('shrink_k1', ShrinkBigKTransformer()),
            ('polyfeat', PolynomialFeatures(interaction_only=1)),
            ('shrink_k2', ShrinkBigKTransformer(selector='elastic-net')),
            ('reg', make_pipeline(StandardScaler(), LinearRegression(fit_intercept=1)))
        ]

        inner_params = {'polyfeat__degree': [2]}
        if self.k > 4:
            interv = -(-self.k // 3)
            np.arange(2, self.k + interv, interv)
            inner_params['shrink_k1__max_k'] = np.arange(4, self.k, 4)
        inner_cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=self.seed)
        X_T_pipe = GridSearchCV(Pipeline(steps=steps), param_grid=inner_params, cv=inner_cv)

        Y_T_X_T_pipe = Pipeline(steps=[('ttr', TransformedTargetRegressor(regressor=X_T_pipe))])
        Y_T__param_grid = {'ttr__transformer': transformer_list}
        lin_reg_Xy_transform = GridSearchCV(Y_T_X_T_pipe, param_grid=Y_T__param_grid, cv=inner_cv)

        self.lr_estimator = lin_reg_Xy_transform
        self.lr_estimator.fit(self.x_train, self.y_train)
        self.attr = pd.DataFrame(self.lr_estimator.cv_results_)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        # generates the model that is saved

    def predict(self):
        self.results = self.lr_estimator.predict(self.x_test)
        self.residuals = self.results - self.y_test.to_numpy().flatten()


def evaluate_results(predicted, actual):
    import sklearn.metrics as skm
    metrics = {
        "max_error": skm.max_error(actual, predicted),
        "mean_absolute_error": skm.mean_absolute_error(actual, predicted),
        "mean_squared_error": skm.mean_squared_error(actual, predicted),
        "root_mean_squared_error": skm.mean_squared_error(actual, predicted, squared=False),
        "r2": skm.r2_score(actual, predicted)
    }
    return metrics


if __name__ == "__main__":
    import pandas as pd
    import time
    import json
    import os
    t0 = time.time()

    # ------------ Data input and setup configuration ---------- #
    _raw_data = pd.read_excel(os.path.join("data", "VB_Data_1a.xlsx"))                  # Data source
    _target = 'Response'                                                                # Column in data source to use as target attribute
    _raw_data = _raw_data.drop("ID", axis=1)

    y = _raw_data[_target]
    x = _raw_data.drop(_target, axis=1)
    lr = LinearRegressionAutomatedVB(cv_folds=2, cv_reps=2)
    lr.set_data(x, y)
    lr.set_pipeline()
    lr.predict()
    metrics = evaluate_results(lr.results, lr.y_test)
    print(json.dumps(metrics, indent=2))
    t1 = time.time() - t0
    print("Time elapsed: {} sec".format(round(t1, 3)))
