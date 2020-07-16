A work-in-progress implementation of Double Machine Learning algorithm as in Chernozhukov et al. (2017) where an ensemble of ML estimators is used to fit the nuisance parameters, with an additional middle step to fit the ensemble weights using part of sample. This is similar to the approach taken by Bajari et al., 2015, and is done at the level of cross-fitting. Although this approach reduces the sample size used for nuisance/linear parameters estimation in each run, it allows to fit each estimator with different sample part.

files:

dml.py contains DoubleMachineLearner class that allows to run the algorithm in two variants:
    - either supplying precomputed weights for the ensemble learner (computed with cross-validation prior to the estimation) which is similar to the original approach
    - or supplying a set of ensemble learners that will be fitted using a part of the sample. If more than 1 ensemble learner is provided, an average over the predictions made by each ensemble will be used.

dml_simulations.py allows to run a set of simulations to compare performance of the two variants of the algorithm

it generates:
    - plots of functional forms of features used in the estimation to highlight the non-linearities
    - latex table with comparison of algorithms in terms of bias and confidence interval coverage

Note that simulations can be quite computationally expensive due to cross-fitting.

References:
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2017) Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1), 1-68.
Bajari, P., Nekipelov, D., Ryan, S.P., and Yang, M. (2015) Machin learning methods for demand estimation. American Economics Review, 105(5), 481-485.


