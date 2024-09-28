#!/usr/bin/python
#-*- coding:utf-8 -*-

########################################################################

import math

from skopt.space import Real, Categorical, Integer

from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost.sklearn import XGBClassifier, XGBRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor

from OMCModels import DecisionTreeRegressorOMC, LinearRegressionOMC, RandomForestRegressorOMC, XGBRegressorOMC, SVROMC, LGBMRegressorOMC
from OMCModels import DecisionTreeClassifierOMC, LogisticRegressionOMC, RandomForestClassifierOMC, XGBClassifierOMC, SVCOMC, LGBMClassifierOMC


def getModelByName(classifier_name, OMC=True, nfeature=750, nsample=100, random_seed=0, njob=20, npoint=4, cv=5, niter=100, static_feature_pos=[]):
    match classifier_name:

        ## with default parameter
        case "CART-R-D":
            if OMC:
                opt = GridSearchCV(
                    estimator=DecisionTreeRegressorOMC(random_state=random_seed, static_feature_position=static_feature_pos),
                    param_grid={'k_omc': list(range(2, round(nsample / 2)))},
                    cv=cv,
                    n_jobs=njob
                )
            else:
                opt = DecisionTreeRegressor(random_state=random_seed)

        case "CART-C-D":
            if OMC:
                opt = GridSearchCV(
                    estimator=DecisionTreeClassifierOMC(random_state=random_seed, static_feature_position=static_feature_pos),
                    param_grid={'k_omc': list(range(2, round(nsample / 2)))},
                    cv=cv,
                    n_jobs=njob
                )
            else:
                opt = DecisionTreeClassifier(random_state=random_seed)

        case "RF-R-D":
            if OMC:
                opt = GridSearchCV(
                    estimator=RandomForestRegressorOMC(n_estimators=1000, random_state=random_seed, static_feature_position=static_feature_pos),
                    param_grid={'k_omc': list(range(2, round(nsample / 2)))},
                    cv=cv,
                    n_jobs=njob
                )
            else:
                opt = RandomForestRegressor(n_estimators=1000, random_state=random_seed)

        case "RF-C-D":
            if OMC:
                opt = GridSearchCV(
                    estimator=RandomForestClassifierOMC(n_estimators=1000, random_state=random_seed, static_feature_position=static_feature_pos),
                    param_grid={'k_omc': list(range(2, round(nsample / 2)))},
                    cv=cv,
                    n_jobs=njob
                )
            else:
                opt = RandomForestClassifier(n_estimators=1000, random_state=random_seed)

        case "XGB-R-D":
            if OMC:
                opt = GridSearchCV(
                    estimator=XGBRegressorOMC(random_state=random_seed, static_feature_position=static_feature_pos),
                    param_grid={'k_omc': list(range(2, round(nsample / 2)))},
                    cv=cv,
                    n_jobs=njob
                )
            else:
                opt = XGBRegressor(random_state=random_seed)

        case "XGB-C-D":
            if OMC:
                opt = GridSearchCV(
                    estimator=XGBClassifierOMC(random_state=random_seed, static_feature_position=static_feature_pos),
                    param_grid={'k_omc': list(range(2, round(nsample / 2)))},
                    cv=cv,
                    n_jobs=njob
                )
            else:
                opt = XGBClassifier(random_state=random_seed)

        case "LGBM-R-D":
            if OMC:
                opt = GridSearchCV(
                    estimator=LGBMRegressorOMC(random_state=random_seed, static_feature_position=static_feature_pos),
                    param_grid={'k_omc': list(range(2, round(nsample / 2)))},
                    cv=cv,
                    n_jobs=njob
                )
            else:
                opt = LGBMRegressor(random_state=random_seed)

        case "LGBM-C-D":
            if OMC:
                opt = GridSearchCV(
                    estimator=LGBMClassifierOMC(random_state=random_seed, static_feature_position=static_feature_pos),
                    param_grid={'k_omc': list(range(2, round(nsample / 2)))},
                    cv=cv,
                    n_jobs=njob
                )
            else:
                opt = LGBMClassifier(random_state=random_seed)

        case "LR-R-D":
            if OMC:
                opt = GridSearchCV(
                    estimator=LinearRegressionOMC(n_jobs=1, static_feature_position=static_feature_pos),
                    param_grid={'k_omc': list(range(2, round(nsample / 2)))},
                    cv=cv,
                    n_jobs=njob
                )
            else:
                opt = LinearRegression(n_jobs=1, static_feature_position=static_feature_pos)

        case "LR-C-D":
            if OMC:
                opt = GridSearchCV(
                    estimator=LogisticRegressionOMC(n_jobs=1, max_iter=10000, static_feature_position=static_feature_pos),
                    param_grid={'k_omc': list(range(2, round(nsample / 2)))},
                    cv=cv,
                    n_jobs=njob
                )
            else:
                opt = LogisticRegression(n_jobs=1, max_iter=10000)

        case "SVM-R-D":
            if OMC:
                opt = GridSearchCV(
                    estimator=SVROMC(static_feature_position=static_feature_pos),
                    param_grid={'k_omc': list(range(2, round(nsample / 2)))},
                    cv=cv,
                    n_jobs=njob
                )
            else:
                opt = SVR()

        case "SVM-C-D":
            if OMC:
                opt = GridSearchCV(
                    estimator=SVCOMC(probability=True, static_feature_position=static_feature_pos),
                    param_grid={'k_omc': list(range(2, round(nsample / 2)))},
                    cv=cv,
                    n_jobs=njob
                )
            else:
                opt = SVC(probability=True)

        ## with hyperparameter tuning
        case "CART-R":
            if OMC:
                opt = BayesSearchCV(
                    DecisionTreeRegressorOMC(random_state=random_seed, static_feature_position=static_feature_pos),
                    {
                        'criterion': Categorical(['squared_error', 'absolute_error', 'poisson']),
                        'max_depth': Integer(2, round(math.pow(nfeature,0.25))),
                        'min_samples_split': Integer(2, 10),
                        'k_omc': Integer(1, round(nsample/2))
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )
            else:
                opt = BayesSearchCV(
                    DecisionTreeRegressor(random_state=random_seed),
                    {
                        'criterion': Categorical(['squared_error', 'absolute_error', 'poisson']),
                        'max_depth': Integer(2, round(math.pow(nfeature,0.25))),
                        'min_samples_split': Integer(2, 10)
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )

        case "CART-C":
            if OMC:
                opt = BayesSearchCV(
                    DecisionTreeClassifierOMC(random_state=random_seed, static_feature_position=static_feature_pos),
                    {
                        'criterion': Categorical(['gini', 'entropy', 'log_loss']),
                        'max_depth': Integer(2, round(math.pow(nfeature,0.25))),
                        'min_samples_split': Integer(2, 10),
                        'k_omc': Integer(1, round(nsample/2))
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )
            else:
                opt = BayesSearchCV(
                    DecisionTreeClassifier(random_state=random_seed),
                    {
                        'criterion': Categorical(['gini', 'entropy', 'log_loss']),
                        'max_depth': Integer(2, round(math.pow(nfeature,0.25))),
                        'min_samples_split': Integer(2, 10)
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )

        case "RF-R":
            if OMC:
                opt = BayesSearchCV(
                    RandomForestRegressorOMC(n_estimators=1000, random_state=random_seed, static_feature_position=static_feature_pos),
                    {
                        'max_depth': Integer(2, round(math.log(nfeature))),
                        'min_samples_leaf': Integer(1, 5),
                        'criterion': Categorical(['squared_error', 'absolute_error', 'poisson']),
                        'max_features': Real(0.01, 1.0),
                        'k_omc': Integer(1, round(nsample/2))
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )
            else:
                opt = BayesSearchCV(
                    RandomForestRegressor(n_estimators=1000, random_state=random_seed),
                    {
                        'max_depth': Integer(2, round(math.log(nfeature))),
                        'min_samples_leaf': Integer(1, 5),
                        'criterion': Categorical(['squared_error', 'absolute_error', 'poisson']),
                        'max_features': Real(0.01, 1.0)
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )

        case "RF-C":
            if OMC:
                opt = BayesSearchCV(
                    RandomForestClassifierOMC(n_estimators=100, random_state=random_seed, static_feature_position=static_feature_pos),
                    {
                        'max_depth': Integer(2, round(math.log(nfeature))),
                        'min_samples_leaf': Integer(1, 5),
                        'criterion': Categorical(['gini', 'entropy', 'log_loss']),
                        'max_features': Real(0.01, 1.0),
                        'k_omc': Integer(1, round(nsample / 2))
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )
            else:
                opt = BayesSearchCV(
                    RandomForestClassifier(n_estimators=100, random_state=random_seed),
                    {
                        'max_depth': Integer(2, round(math.log(nfeature))),
                        'min_samples_leaf': Integer(1, 5),
                        'criterion': Categorical(['gini', 'entropy', 'log_loss']),
                        'max_features': Real(0.01, 1.0)
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )

        case "XGB-R":
            if OMC:
                opt = BayesSearchCV(
                    XGBRegressorOMC(random_state=random_seed, static_feature_position=static_feature_pos),
                    {
                        'max_depth': Integer(2, round(math.pow(nfeature,0.25))),
                        'max_leaves': Integer(3, round(math.pow(nfeature,0.25)*2)),
                        'grow_policy': Categorical(["depthwise", "lossguide"]),
                        'base_score': Real(0.1, 0.9),
                        'reg_alpha': Real(0.0, 3.0),
                        'reg_lambda': Real(0.0, 3.0),
                        'k_omc': Integer(1, round(nsample/2))
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )
            else:
                opt = BayesSearchCV(
                    XGBRegressor(random_state=random_seed),
                    {
                        'max_depth': Integer(2, round(math.pow(nfeature,0.25))),
                        'max_leaves': Integer(3, round(math.pow(nfeature,0.25)*2)),
                        'grow_policy': Categorical(["depthwise", "lossguide"]),
                        'base_score': Real(0.1, 0.9),
                        'reg_alpha': Real(0.0, 3.0),
                        'reg_lambda': Real(0.0, 3.0)
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )

        case "XGB-C":
            if OMC:
                opt = BayesSearchCV(
                    XGBClassifierOMC(random_state=random_seed, static_feature_position=static_feature_pos),
                    {
                        'max_depth': Integer(2, round(math.pow(nfeature, 0.25))),
                        'max_leaves': Integer(3, round(math.pow(nfeature, 0.25) * 2)),
                        'grow_policy': Categorical(["depthwise", "lossguide"]),
                        'base_score': Real(0.1, 0.9),
                        'reg_alpha': Real(0.0, 3.0),
                        'reg_lambda': Real(0.0, 3.0),
                        'k_omc': Integer(1, round(nsample / 2))
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )
            else:
                opt = BayesSearchCV(
                    XGBClassifier(random_state=random_seed),
                    {
                        'max_depth': Integer(2, round(math.pow(nfeature, 0.25))),
                        'max_leaves': Integer(3, round(math.pow(nfeature, 0.25) * 2)),
                        'grow_policy': Categorical(["depthwise", "lossguide"]),
                        'base_score': Real(0.1, 0.9),
                        'reg_alpha': Real(0.0, 3.0),
                        'reg_lambda': Real(0.0, 3.0)
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )

        case "LGBM-R":
            if OMC:
                opt = BayesSearchCV(
                    LGBMRegressorOMC(random_state=random_seed, static_feature_position=static_feature_pos),
                    {
                        'boosting_type': Categorical(['gbdt', 'dart', 'goss']),
                        'num_leaves': Integer(3, round(math.pow(nfeature,0.25)*2)),
                        'max_depth': Integer(2, round(math.pow(nfeature,0.25))),
                        'reg_lambda': Real(0.0, 3.0),
                        'min_child_samples': Integer(5, 40),
                        'min_child_weight': Real(0.0001, 0.01),
                        'reg_alpha': Real(0.0, 3.0),
                        'reg_lambda': Real(0.0, 3.0),
                        'k_omc': Integer(1, round(nsample/2))
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )
            else:
                opt = BayesSearchCV(
                    LGBMRegressor(random_state=random_seed),
                    {
                        'boosting_type': Categorical(['gbdt', 'dart', 'goss']),
                        'num_leaves': Integer(3, round(math.pow(nfeature,0.25)*2)),
                        'max_depth': Integer(2, round(math.pow(nfeature,0.25))),
                        'reg_lambda': Real(0.0, 3.0),
                        'min_child_samples': Integer(5, 40),
                        'min_child_weight': Real(0.0001, 0.01),
                        'reg_alpha': Real(0.0, 3.0),
                        'reg_lambda': Real(0.0, 3.0)
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )

        case "LGBM-C":
            if OMC:
                opt = BayesSearchCV(
                    LGBMClassifierOMC(random_state=random_seed, static_feature_position=static_feature_pos),
                    {
                        'num_leaves': Integer(3, round(math.pow(nfeature, 0.25) * 2)),
                        'max_depth': Integer(2, round(math.pow(nfeature, 0.25))),
                        'reg_lambda': Real(0.0, 3.0),
                        'min_child_samples': Integer(5, 40),
                        'min_child_weight': Real(0.0001, 0.01),
                        'reg_alpha': Real(0.0, 3.0),
                        'reg_lambda': Real(0.0, 3.0),
                        'k_omc': Integer(1, round(nsample/2))
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )
            else:
                opt = BayesSearchCV(
                    LGBMClassifier(random_state=random_seed),
                    {
                        'num_leaves': Integer(3, round(math.pow(nfeature, 0.25) * 2)),
                        'max_depth': Integer(2, round(math.pow(nfeature, 0.25))),
                        'reg_lambda': Real(0.0, 3.0),
                        'min_child_samples': Integer(5, 40),
                        'min_child_weight': Real(0.0001, 0.01),
                        'reg_alpha': Real(0.0, 3.0),
                        'reg_lambda': Real(0.0, 3.0)
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )

        case "LR-R":
            if OMC:
                opt = BayesSearchCV(
                    LinearRegressionOMC(n_jobs=1, static_feature_position=static_feature_pos),
                    {'k_omc': Integer(1, round(nsample/2))},
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )
            else:
                opt = LinearRegression(n_jobs=njob)

        case "LR-C":
            if OMC:
                opt = BayesSearchCV(
                    LogisticRegressionOMC(n_jobs=1, static_feature_position=static_feature_pos),
                    {
                        'C': Real(0.1, 10.0),
                        'k_omc': Integer(1, round(nsample / 2))
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )
            else:
                opt = LogisticRegression(random_state=random_seed, n_jobs=njob, max_iter=10000)

        case "SVM-R":
            if OMC:
                opt = BayesSearchCV(
                    SVROMC(static_feature_position=static_feature_pos),
                    {
                        'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
                        'tol': Real(0.0001, 0.1),
                        'epsilon': Real(0.5, 3.0),
                        'k_omc': Integer(1, round(nsample/2))
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )
            else:
                opt = BayesSearchCV(
                    SVR(),
                    {
                        'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
                        'tol': Real(0.0001, 0.1),
                        'epsilon': Real(0.5, 3.0)
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )

        case "SVM-C":
            if OMC:
                opt = BayesSearchCV(
                    SVCOMC(probability=True, static_feature_position=static_feature_pos),
                    {
                        'kernel': Categorical(['linear', 'rbf', 'sigmoid']),
                        'tol': Real(0.0001, 0.1),
                        'k_omc': Integer(1, round(nsample / 2))
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )
            else:
                opt = BayesSearchCV(
                    SVC(probability=True),
                    {
                        'kernel': Categorical(['linear', 'rbf', 'sigmoid']),
                        'tol': Real(0.0001, 0.1)
                    },
                    n_iter=niter,
                    n_jobs=njob,
                    n_points=npoint,
                    cv=cv
                )

        case "KNN-R":
            opt = BayesSearchCV(
                KNeighborsRegressor(),
                {
                    'n_neighbors': Integer(1, 10),
                    'weight': Categorical(['uniform', 'distance']),
                    'algorithm': Categorical(['ball_tree', 'kd_tree', 'brute']),
                    'metric ': Integer(1, 2)
                },
                n_iter=niter,
                n_jobs=njob,
                n_points=npoint,
                cv=cv
            )
        case "KNN-C":
            opt = BayesSearchCV(
                KNeighborsClassifier(),
                {
                    'n_neighbors': Integer(1, 10),
                    'weight': Categorical(['uniform', 'distance']),
                    'algorithm': Categorical(['ball_tree', 'kd_tree', 'brute']),
                    'metric ': Integer(1, 2)
                },
                n_iter=niter,
                n_jobs=njob,
                n_points=npoint,
                cv=cv
            )
        case _:
            print("Classifier " + classifier_name + " not supported.")
            opt = None

    return(opt)
