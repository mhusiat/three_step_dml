from time import time
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

from dml import DoubleMachineLearner, ensemble_weights_cv

np.random.seed(42)


def m(X: np.array) -> np.array:
    """ helper function to get non-linear relations """
    f1_ = (X - 2 * X ** 2 - 2 * X ** 3) / 10
    f2_ = np.sin(X) * 2
    f_ = f1_ + f2_
    f_[X > 0] = -1 * X[X > 0]
    return np.clip(f_, a_min=-5, a_max=3)


def g(X: np.array) -> np.array:
    """ helper function to get non-linear relations """
    f_ = -np.sin(X)
    f_[(X > 1)] = 2
    f_[(X > 3)] = 1
    return f_


def plot_mX_gX(path="dml_simulations_data_graph.pdf") -> None:
    """ a wrapper to plot graphs of non-linear functions for reference """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax = axes.flatten()

    x = np.linspace(-5, 5, 1000)

    ax[0].plot(x, g(x))
    ax[0].set_title("g(x) function on linear feature T")
    ax[1].plot(x, m(x))
    ax[1].set_title("m(x) function on target feature Y")

    plt.savefig(path)
    plt.clf()
    return None


def gen_sim_data(n=1000, k=4, param=[0.5]) -> np.array:
    """
    functional form of data-generating process fixed here; for estimating
    2+ linear parameters this function needs to be adjusted manually
    """
    X = np.random.randn(n, k)
    T = g(X[:, 0]) + np.random.randn(n)
    Y = param * T + m(X[:, 0]) + np.random.randn(n)

    # return a "target | linear_features | nuisance features" ordered array
    return np.hstack([Y.reshape(-1, 1), T.reshape(-1, 1), X])


def sum_square(b: np.array, X: np.array, y: np.array) -> float:
    """ compute sum of squared errors for given beta vector """
    return np.sum((y.reshape(-1, 1) - X.dot(b).reshape(-1, 1)) ** 2)


class ExactWeights:
    def __init__(self):
        """ sklearn-like object; estimates weights where sum(weights)=1 """
        self.name = "custom weight estimator"

    def fit(self, X: np.array, Y: np.array):
        # constraint weights to be between 0 and 1 and sum up to 1
        bounds = Bounds(0, 1)
        constraints = LinearConstraint(np.ones(X.shape[1]), 1, 1)

        # ignore particular warning about setting the hessian to zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # set starting values at equal weights for each estimator
            self.coefs_ = minimize(
                sum_square,
                np.ones(X.shape[1]) * 0.5,
                args=(X, Y),
                method="trust-constr",
                constraints=constraints,
                bounds=bounds,
            ).x

    def predict(self, Xp):
        return Xp.dot(self.coefs_)


"""

if executed as the main script, will run simulations for 3 variants of the
ensemble Double Machine Learning algorithm as in Chernozhukov et al. (2017):

- 'classic' ensemble DML2 as in Chernozhukov et al. (2017) with ensemble
   weights summing up to 1 pre-computed with 5-fold cross-validation

- 'classic with half step' ensemble weights are computed using the same
   fold from cross-fitting as the parameter of interest

- 'custom' ensemble DML2 similar to Chernozhukov et al. (2017) but with
   step in the middle to compute cross-fit-run specific ensemble weights;
   ensemble weights also sum up to 1 and are restricted to be positive
   this is similar approach to one taken in Bajari et al. (2015)

note that custom estimator takes an average of the estimates from given
ensemble methods if more than 1 ensemble method is provided

References:
   Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
   Newey, W., and Robins, J. (2017) Double/debiased machine learning for
   treatment and structural parameters. The Econometrics Journal, 21(1), 1-68.

   Bajari, P., Nekipelov, D., Ryan, S.P., and Yang, M. (2015) Machine learning
   methods for demand estimation. American Economic Review, 105(5), 481-485.
"""

# specify tuples of estimators for nuisance and ensemble estimators
NUISANCE_ESTIMATORS = [
    ("Elastic Net", ElasticNetCV(cv=3)),
    ("K nearest neighbors", KNeighborsRegressor(n_neighbors=50, weights="distance")),
    ("Support Vector Machines", SVR(gamma="auto")),
    ("Random Forest", RandomForestRegressor(n_estimators=100, max_features="sqrt")),
    ("Boosted Trees", GradientBoostingRegressor(n_estimators=80)),
    ("Kernel Ridge", KernelRidge()),
]
ENSEMBLE_ESTIMATORS = [ExactWeights()]

SIMULATIONS, CROSSFIT_RUNS, PARAM_VALUE = 2, 2, [0.5]
CORES_USED = 10  # multiprocessing is used in parallel at crossfiting level

# specify combinations of sample x nr_features for simulations
# this means that there will be len(N)*len(K)*SIMULATIONS runs
N = [100, 200]
K = [2, 3]

SIMULATION_DML_ESTIMATORS = OrderedDict(
    [
        (
            "classic",
            DoubleMachineLearner(
                [estimator for _, estimator in NUISANCE_ESTIMATORS],
                crossfit_runs=CROSSFIT_RUNS,
                nfolds=(1, 3),
            ),
        ),
        (
            "classic_in_weights",
            DoubleMachineLearner(
                [estimator for _, estimator in NUISANCE_ESTIMATORS],
                crossfit_runs=CROSSFIT_RUNS,
                nfolds=(1, 3),
                in_ensemble_weights=True,
            ),
        ),
        (
            "custom",
            DoubleMachineLearner(
                [estimator for _, estimator in NUISANCE_ESTIMATORS],
                ensemble_estimators=[ENSEMBLE_ESTIMATORS[0]],
                crossfit_runs=CROSSFIT_RUNS,
                nfolds=(1, 1, 3),
            ),
        ),
    ]
)

if __name__ == "__main__":
    # plot the m() and g() functions to have a visual reference
    # on what kind of non-linear functions are fitted
    plot_mX_gX()

    # create a table of output that will store results across n-k combinations
    RESULTS_TABLE_PATH = "dml_simulation_results.txt"
    multi_index, multi_column = [], []
    for n in N:
        multi_index.append((n, "bias"))
        multi_index.append((n, "acc"))
    for k in K:
        (multi_column.append((k, name)) for name in SIMULATION_DML_ESTIMATORS.keys())

    index = pd.MultiIndex.from_tuples(multi_index, names=["size", "measure"])
    column = pd.MultiIndex.from_tuples(multi_column, names=["features", "estimator"])
    res_frame = pd.DataFrame({}, index=index, columns=column)

    start = time()
    for n in N:
        for k in K:

            # initiate objects to store the results of this n-k combination
            SIMULATION_RESULTS = {name: [] for name in SIMULATION_DML_ESTIMATORS.keys()}
            TABLE_BIAS = {name: [] for name in SIMULATION_DML_ESTIMATORS.keys()}
            TABLE_ACC = {name: 0 for name in SIMULATION_DML_ESTIMATORS.keys()}

            for run in range(SIMULATIONS):
                data = gen_sim_data(n=n, k=k, param=PARAM_VALUE)

                # precompute weights that will be used across
                # cross-fitting for the classic method
                weights = ensemble_weights_cv(
                    data[:, 1 + len(PARAM_VALUE) :],
                    data[:, : 1 + len(PARAM_VALUE)],
                    [estimator for _, estimator in NUISANCE_ESTIMATORS],
                    ENSEMBLE_ESTIMATORS[0],
                    nfolds=5,
                )

                for estimator_name, dml_estimator in SIMULATION_DML_ESTIMATORS.items():
                    if estimator_name in ["classic", "classic_in_weights"]:
                        # classic version needs to be provided with weights
                        dml_estimator.fit(
                            data[:, 1 + len(PARAM_VALUE) :],
                            data[:, 1 : 1 + len(PARAM_VALUE)],
                            data[:, [0]],
                            cores_used=CORES_USED,
                            ensemble_weights=weights,
                        )
                    else:
                        # custom version estimates weights with middle step
                        dml_estimator.fit(
                            data[:, 1 + len(PARAM_VALUE) :],
                            data[:, 1 : 1 + len(PARAM_VALUE)],
                            data[:, [0]],
                            cores_used=CORES_USED,
                        )

                    SIMULATION_RESULTS[estimator_name].append(
                        (
                            dml_estimator.averaged_estimate,
                            np.sqrt(dml_estimator.mean_corrected_variance),
                        )
                    )

                # this variable only provides the estimated runtime for the
                # current n-k combination (runtime differs largely by n-k)
                _avg_runtime = (time() - start) / (run + 1)
                est_time = round(_avg_runtime * (SIMULATIONS - run + 1) / 3600, 2)
                print(f"n: {n}; k: {k}; run {run}; remaining: {est_time}h\r", end="")

            for name, results in SIMULATION_RESULTS.items():
                for estimate, standard_errors in results:

                    # store the bias as average mean squared error
                    TABLE_BIAS[name].append(PARAM_VALUE - estimate)

                    # also check if the confidence intervals calculated
                    # capture the true value (should be ~ 95%)
                    if (
                        estimate - 1.96 * standard_errors
                        <= PARAM_VALUE
                        <= estimate + 1.96 * standard_errors
                    ):
                        TABLE_ACC[name] += 1 / SIMULATIONS

            # simulations can take a while so print results as they arrive
            # note that the results are stored within a dataframe regardless
            for name, bias in TABLE_BIAS.items():
                res_frame.loc[(n, "bias"), (k, name)] = np.mean(np.array(bias) ** 2)
                print(
                    f"n={n}, k={k}, estimator={name}; bias: "
                    f"{np.round(np.mean(np.array(bias)**2), 7)}"
                )

            for name, accuracy in TABLE_ACC.items():
                res_frame.loc[(n, "acc"), (k, name)] = f"({accuracy})"
                print(
                    f"n={n}, k={k}, estimator={name}; "
                    f"accuracy: {np.round(accuracy, 7)}"
                )

    # store the results table in the main folder for reference
    with open(RESULTS_TABLE_PATH, "w") as f:
        f.write(res_frame.to_latex(multicolumn_format="c"))
