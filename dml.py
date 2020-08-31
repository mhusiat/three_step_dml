from multiprocessing import Pool

import numpy as np
import pandas as pd


def _two_step_orthogonalization(
    nfolds: int,
    tsize: int,
    df_folds: list,
    fold_combinations: tuple,
    nuisance_estim: list,
    ensemble_weights: np.array,
    in_ensemble_weights=False,
) -> tuple:
    """
    orthogonalize features with an ensemble of estimators using precomputed
    set of ensemble weights (following Chernozhukov et al., 2017)
    """
    # initiate the list storage for orthogonalized features
    orthogonalized_target_and_treatment = []

    for cbn in fold_combinations:
        # determine what folds have what task in the current run of estimation
        linear_folds = cbn[: nfolds[0]]
        nuisance_folds = cbn[nfolds[0] :]

        # split samples into 2 parts: training the nuisance parameters and
        # estimating the parameters of interest on orthogonalized features
        df_train = np.vstack([df_folds[c] for c in nuisance_folds])
        df_params = np.vstack([df_folds[c] for c in linear_folds])

        # initialize fitted values of treatment regressors
        fitted_values = np.zeros([df_params.shape[0], tsize + 1, len(nuisance_estim)])
        estimators_linear = np.zeros([df_params.shape[0], tsize + 1])

        # fit each variable of interest seperately against the nuisance params
        # note that there are tsize treatment features + 1 target feature
        for t in range(tsize + 1):
            for which, estim in enumerate(nuisance_estim):
                # train the model using nuisance sample
                estim.fit(df_train[:, tsize + 1 :], df_train[:, t])

                # fit values using the linear sample
                fitted_values[:, t, which] = estim.predict(df_params[:, tsize + 1 :])

            if in_ensemble_weights:
                tX = fitted_values[:, t, :]
                ensemble_weights[:, t] = np.linalg.inv(tX.T.dot(tX)).dot(
                    tX.T.dot(df_params[:, t])
                )
            # use pre-computed weights to combine the nuisance estimators
            estimators_linear[:, t] = fitted_values[:, t, :].dot(ensemble_weights[:, t])

        # initialize orthogonalized features for each ensemble estimator
        orthogonal_features = df_params[:, : tsize + 1] - estimators_linear
        orthogonalized_target_and_treatment.append(orthogonal_features)

    # return stacked orthogonalized features; note that order
    # of observations needs to be preserved here
    return np.vstack(orthogonalized_target_and_treatment)


def _three_step_orthogonalization(
    nfolds: int,
    tsize: int,
    df_folds: list,
    fold_combinations: tuple,
    nuisance_estim: list,
    ensemble_estim: list,
) -> tuple:
    """
    orthogonalize features with an average over ensembles of estimators which
    are computed using an extra fold (hence 3 steps); this is a similar
    procedure as DML in Chernozhukov et al. (2017) but with an extra step
    in the middle instead of cross-validation prior to estimation
    """
    # initiate the list storage for orthogonalized features
    orthogonalized_target_and_treatment = []

    # routine is rerun nfold times so that each fold is used
    # in different tasks the same amount of times
    for cbn in fold_combinations:

        # determine what folds have what task in the current run of estimation
        linear_folds = cbn[: nfolds[0]]
        ensemble_folds = cbn[nfolds[0] : nfolds[0] + nfolds[1]]
        nuisance_folds = cbn[nfolds[0] + nfolds[1] :]

        # split samples into 3 parts: training the nuisance parameters;
        # determining ensemble weights; estimating the parameters of interest
        df_train = np.vstack([df_folds[c] for c in nuisance_folds])
        df_ensemble = np.vstack([df_folds[c] for c in ensemble_folds])
        df_params = np.vstack([df_folds[c] for c in linear_folds])

        # initialize fitted values for target and treatment features
        estimators_ensemble = np.zeros(
            [df_ensemble.shape[0], tsize + 1, len(nuisance_estim)]
        )
        estimators_linear_nuisance = np.zeros(
            [df_params.shape[0], tsize + 1, len(nuisance_estim)]
        )
        estimators_linear_ensemble = np.zeros(
            [df_params.shape[0], tsize + 1, len(ensemble_estim)]
        )

        # fit each variable of interest seperately against the nuisance params
        # and predict orthogonalized features using ensemble and linear samples
        for i in range(tsize + 1):
            for which, estim in enumerate(nuisance_estim):
                # train the model using the train sample only
                estim.fit(df_train[:, tsize + 1 :], df_train[:, i])

                # predict on both ensemble and linear params samples
                estimators_ensemble[:, i, which] = estim.predict(
                    df_ensemble[:, tsize + 1 :]
                )
                estimators_linear_nuisance[:, i, which] = estim.predict(
                    df_params[:, tsize + 1 :]
                )

            for which, estim in enumerate(ensemble_estim):
                # train ensemble using fitted values from previous step
                estim.fit(estimators_ensemble[:, i, :], df_ensemble[:, i])

                # and predict the features using fitted values on linear
                # parameters sample and trained weights on ensemble sample
                estimators_linear_ensemble[:, i, which] = estim.predict(
                    estimators_linear_nuisance[:, i, :]
                )
        # average over the predictions of different ensemble methods used
        averaged_ensembles = np.mean(estimators_linear_ensemble, axis=2)

        # orthonalize the target and linear features against fitted values
        orthogonal_features = df_params[:, : tsize + 1] - averaged_ensembles

        # note that order of linear folds needs to be preserved here
        orthogonalized_target_and_treatment.append(orthogonal_features)

    # combine list of orthogonalized features into a single array
    return np.vstack(orthogonalized_target_and_treatment)


def _run_double_machine_learning(
    df: np.array,
    tsize: int,
    nuisance_estim: list,
    ensemble_estim: list,
    ensemble_weights: np.array,
    nfolds,
    in_ensemble_weights,
):
    """
    wrapper function that fits a single cross-fitting run of the model
    used by the .fit() method on the DoubleMachineLearner class object
    """
    # create sum(nfolds) combinations of folds so that each piece of data is
    # used the same amount of times in each part throughout the estimation
    fold_combinations = [
        list(range(i, sum(nfolds))) + list(range(0, i)) for i in range(sum(nfolds))
    ]

    # determine fold size and fold the dataset (approximately) evenly
    fold_size = int(np.floor(df.shape[0] / sum(nfolds)))
    df_folds = np.split(df, [fold_size * which for which in range(1, sum(nfolds))])

    if len(nfolds) == 2:
        orthogonalized_features = _two_step_orthogonalization(
            nfolds,
            tsize,
            df_folds,
            fold_combinations,
            nuisance_estim,
            ensemble_weights,
            in_ensemble_weights=in_ensemble_weights,
        )
    elif len(nfolds) == 3:
        orthogonalized_features = _three_step_orthogonalization(
            nfolds, tsize, df_folds, fold_combinations, nuisance_estim, ensemble_estim
        )
    else:
        raise ValueError("there should be either 2 or 3 sets of folds")

    # split the results into target and treatment features
    Y = orthogonalized_features[:, 0]
    T = orthogonalized_features[:, 1:]

    DML_estimates = np.linalg.inv(np.dot(T.T, T)).dot(np.dot(T.T, Y))

    # note that variance estimates still need a finite sample correction
    residuals = Y - T.dot(DML_estimates)
    asymptotic_variance_estimates = np.mean(residuals ** 2) / T.T.dot(T)

    return DML_estimates, np.diag(asymptotic_variance_estimates)


def ensemble_weights_cv(
    X: np.array,
    y: np.array,
    nuisance_estimators: list,
    ensemble_estimator: object,
    nfolds=5,
) -> np.array:
    """
    helper function to pre-estimate ensemble weights for k features in
    Double Machine Learning algorithm using nfolds cross-validation

    takes:
        X: n x m numpy array of nuisance features used for training

        y: n x k numpy array of k features that are estimated

        nuisance_estimators: list of sklearn-like estimator objects
                             used to estimate the k features of interest

        ensemble_estimator: sklearn-like estimator for ensemble weights

        nfolds: number of folds used in the cross-validation routine

    returns:
        n x k array of weights; where n = nr of estimators used in the ensemble
    """
    # stack features together for consistent splitting in cross-validation
    df = np.hstack([y, X])

    # create sum(nfolds) combinations of folds so that each piece of data is
    # used the same amount of times throughout the estimation
    fold_combinations = [
        list(range(i, nfolds)) + list(range(0, i)) for i in range(nfolds)
    ]

    # determine fold size and fold the dataset (approximately) evenly
    sample_fold = int(np.floor(df.shape[0] / nfolds))
    df_folds = np.split(df, [sample_fold * i for i in range(1, nfolds)])

    # initiate final weights matrix
    final_weights = np.zeros([len(nuisance_estimators), y.shape[1]])

    for cbn in fold_combinations:
        # assign roles to folds in the current run
        ensemble_sample = df_folds[0]
        train_sample = np.vstack(df_folds[1:])

        # initiate the weights for each ensemble and feature in this run
        current_run_weights = np.zeros([len(nuisance_estimators), y.shape[1]])
        for t in range(y.shape[1]):
            # initiate fitted values array
            fitted_values = np.zeros(
                [ensemble_sample.shape[0], len(nuisance_estimators)]
            )

            for which, estimator in enumerate(nuisance_estimators):
                # train the nuisance parameter estimator
                estimator.fit(train_sample[:, y.shape[1] :], train_sample[:, t])

                # fit the values on the ensemble sample
                fitted_values[:, which] = estimator.predict(
                    ensemble_sample[:, y.shape[1] :]
                )
            # estimate weights of fitted values against ensemble sample target
            ensemble_estimator.fit(fitted_values, ensemble_sample[:, t])

            # store the weights for the feature t of the current run
            current_run_weights[:, t] = ensemble_estimator.coefs_

        # update final weights with set of weights for each of the k features
        # estimated divided by the number of nfold cross-validation runs
        final_weights += current_run_weights / nfolds

    return final_weights


class DoubleMachineLearner:
    """
    double machine (DML2) learning as presented by Chernozhukov et al. (2017)
    paper with additional intermediate step to fit a list of ensemble methods
    similar to Bajari et al. (2015) paper
    """

    def __init__(
        self,
        nuisance_estimators,
        ensemble_estimators=None,
        ensemble_weights=None,
        crossfit_runs=50,
        nfolds=(1, 1, 3),
        in_ensemble_weights=False,
    ):
        """
        initialize the estimator

        takes:

            nuisance_estimators: list of sklearn or sklearn-like estimators
            used to fit the nuisance paramaters

            ensemble_estimators: list of sklearn or sklearn-like estimators
            used to fit the ensemble weights across models

            crossfit_runs: number of random splits of datasets and consequent
            estimation of parameters to average out the bias resulting from a
            given choice of splitting the data

        the dependance on sklearn-like estimators is necessary since functions
        often call for .fit() and .predict() methods of the estimators
        """
        # ensure that folds are integers for correct splitting of the dataset
        for _fold_size in nfolds:
            if type(_fold_size) != int:
                raise TypeError("fold size should only be expressed in integers")

        # NOTE: temporary fix on linear fold to be exactly of size 1; this may be
        # changed in future version but for now it decreases computational burden
        # of the estimator and avoids another layer of prediction averaging
        if nfolds[0] != 1:
            raise ValueError("linear param fold should be fixed at 1 for now")

        self.nuisance_estimators = nuisance_estimators
        self.ensemble_estimators = ensemble_estimators
        self.crossfit_runs = crossfit_runs
        self.nfolds = nfolds
        self.in_ensemble_weights = in_ensemble_weights

    def fit(
        self, X: np.array, T: np.array, Y: np.array, cores_used=1, ensemble_weights=None
    ):
        """
        fit the Double Machine Learning estimator

        takes:

            Y: dependant variable (target); N x 1 column numpy array

            T: features for which linear parameters are estimates; N x T array
               (also sometimes referred to as 'treatment' within the code)

            X: features used to estimate nuisance parameters; N x M array

            cores_used: how many multiple processes are used at the same time
            since crossfit runs are independent of each other, multiprocessing
            package is used to leverage this and speed up the estimation
        """
        # ensure that features are provided in a correct format and size
        if Y.shape[0] != T.shape[0] or Y.shape[0] != X.shape[0]:
            raise ValueError("length of observations does not match for inputs")

        if Y.shape[1] != 1:
            raise ValueError("dependant variable should be a Nx1 column vector")

        if self.ensemble_estimators is None and ensemble_weights is None:
            raise ValueError(
                "you need to either provide pre-computed ensemble "
                "weights or specify a set of estimators for them"
            )

        # stack features together and convert into a dataframe; this simplifies
        # consistent multiple splits of the dataset across estimation
        dataframe = pd.DataFrame(np.hstack([Y, T, X]))
        treatment_size = T.shape[1]  # record nr of the treatment features

        # generate multiple shuffled index orderings for random data splits
        # across crossfit runs; this is done to average out the bias due to
        # making a random split in the data for different parts of the estimator
        # (done beforehand to avoid identical splits in multiprocessing)
        shuffled_index = [
            np.random.choice(dataframe.index, dataframe.shape[0], replace=False)
            for _ in range(self.crossfit_runs)
        ]

        # create empty arrays for storing crossfit results across estimators
        estimates_array = np.zeros([self.crossfit_runs, treatment_size])
        variance_array = np.zeros([self.crossfit_runs, treatment_size])

        # use multiprocessing for simultaenous model estimation across crossfit
        # runs; since these are unrelated, asynchronous multiprocessing allows
        # to speed up the estimation process substantially
        with Pool(processes=cores_used) as mp_pool:
            pool_of_tasks = [
                mp_pool.apply_async(
                    _run_double_machine_learning,
                    args=(
                        dataframe.loc[shuffled_index[i], :],
                        treatment_size,
                        self.nuisance_estimators,
                        self.ensemble_estimators,
                        ensemble_weights,
                        self.nfolds,
                        self.in_ensemble_weights,
                    ),
                )
                for i in range(self.crossfit_runs)
            ]

            parallel_results = [p.get() for p in pool_of_tasks]

        # unpack the results after finishing all parallel crossfit runs
        for which, results in enumerate(parallel_results):
            estimates_array[which, :], variance_array[which, :] = results

        # create average estimates across the ensemble estimators
        self.averaged_estimate = np.mean(estimates_array, axis=0)

        # estimate variance for each estimate; note that this is a
        # finite-sample mean or median corrected variance that corrects for
        # random splits within each cross-fit run of the estimator
        self.mean_corrected_variance = np.mean(
            variance_array + (estimates_array - np.mean(estimates_array, axis=0)) ** 2,
            axis=0,
        )
        self.median_corrected_variance = np.mean(
            variance_array
            + (estimates_array - np.median(estimates_array, axis=0)) ** 2,
            axis=0,
        )
