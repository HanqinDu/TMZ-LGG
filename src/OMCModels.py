#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
Version: 1.0

Author: Hanqin Du <hanqindu@outlook.com>

OMC version of scikit learn estimator, accept k_omc as param
"""

import sys
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from warnings import warn

from sklearn.feature_selection import SelectKBest

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from xgboost.sklearn import XGBClassifier, XGBRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor, LGBMModel
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from lightgbm.sklearn import LightGBMError

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

########################################################################
class DecisionTreeRegressorOMC(DecisionTreeRegressor):
    def __init__(
            self,
            *,
            criterion="squared_error",
            splitter="best",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            ccp_alpha=0.0,
            k_omc=10,
            static_feature_position=[],
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )
        self.k_omc = k_omc
        self.static_feature_position = static_feature_position

    
    def fit(self, X, y, sample_weight=None):
        self.selector = SelectKBest(k=self.k_omc).fit(X, y)
        self.omc_mask = self.selector.get_support()
        for i in self.static_feature_position:
            self.omc_mask[i] = True
        return (super().fit(X[:,self.omc_mask], y, sample_weight))

    def predict(self, X):
        return (super().predict(X[:,self.omc_mask]))


class DecisionTreeClassifierOMC(DecisionTreeClassifier):
    def __init__(
            self,
            *,
            criterion="gini",
            splitter="best",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
            k_omc=10,
            static_feature_position=[],
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )
        self.k_omc = k_omc
        self.static_feature_position = static_feature_position

    def fit(self, X, y, sample_weight=None):
        self.selector = SelectKBest(k=self.k_omc).fit(X, y)
        self.omc_mask = self.selector.get_support()
        for i in self.static_feature_position:
            self.omc_mask[i] = True
        return (super().fit(X[:,self.omc_mask], y, sample_weight))

    def predict(self, X):
        try:
            return (super().predict(X[:,self.omc_mask]))
        except:
            return (super().predict(X))

    def predict_proba(self, X):
        try:
            return (super().predict_proba(X[:,self.omc_mask]))
        except:
            return (super().predict_proba(X))


class LinearRegressionOMC(LinearRegression):

    def __init__(
            self,
            *,
            fit_intercept=True,
            copy_X=True,
            n_jobs=None,
            positive=False,
            k_omc=10,
            static_feature_position=[],
    ):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        self.k_omc = k_omc
        self.static_feature_position = static_feature_position

    def fit(self, X, y, sample_weight=None):
        self.selector = SelectKBest(k=self.k_omc).fit(X, y)
        self.omc_mask = self.selector.get_support()
        for i in self.static_feature_position:
            self.omc_mask[i] = True
        return (super().fit(X[:,self.omc_mask], y, sample_weight))

    def predict(self, X):
        return (super().predict(X[:,self.omc_mask]))


class LogisticRegressionOMC(LogisticRegression):

    def __init__(
            self,
            penalty="l2",
            *,
            dual=False,
            tol=1e-4,
            C=1.0,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            random_state=None,
            solver="lbfgs",
            max_iter=100,
            multi_class="auto",
            verbose=0,
            warm_start=False,
            n_jobs=None,
            l1_ratio=None,
            k_omc=10,
            static_feature_position=[],
    ):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.k_omc = k_omc
        self.static_feature_position = static_feature_position

    def fit(self, X, y, sample_weight=None):
        self.selector = SelectKBest(k=self.k_omc).fit(X, y)
        self.omc_mask = self.selector.get_support()
        for i in self.static_feature_position:
            self.omc_mask[i] = True
        return (super().fit(X[:,self.omc_mask], y, sample_weight))

    def predict(self, X):
        try:
            return (super().predict(X[:,self.omc_mask]))
        except:
            return (super().predict(X))

    def predict_proba(self, X):
        try:
            return (super().predict_proba(X[:,self.omc_mask]))
        except:
            return (super().predict_proba(X))



class RandomForestRegressorOMC(RandomForestRegressor):
    def __init__(
            self,
            n_estimators=100,
            *,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
            k_omc=10,
            static_feature_position=[],
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
        )
        self.k_omc = k_omc
        self.static_feature_position = static_feature_position

    def fit(self, X, y, sample_weight=None):
        self.selector = SelectKBest(k=self.k_omc).fit(X, y)
        self.omc_mask = self.selector.get_support()
        for i in self.static_feature_position:
            self.omc_mask[i] = True
        return (super().fit(X[:,self.omc_mask], y, sample_weight))

    def predict(self, X):
        return (super().predict(X[:,self.omc_mask]))




class RandomForestClassifierOMC(RandomForestClassifier):
    def __init__(
            self,
            n_estimators=100,
            *,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None,
            k_omc=10,
            static_feature_position=[],
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
        )
        self.k_omc = k_omc
        self.static_feature_position = static_feature_position

    def fit(self, X, y, sample_weight=None):
        self.selector = SelectKBest(k=self.k_omc).fit(X, y)
        self.omc_mask = self.selector.get_support()
        for i in self.static_feature_position:
            self.omc_mask[i] = True
        try:
            del i
        except:
            pass
        return (super().fit(X[:,self.omc_mask], y, sample_weight))

    def predict(self, X):
        try:
            return (super().predict(X[:,self.omc_mask]))
        except:
            return (super().predict(X))

    def predict_proba(self, X):
        try:
            return (super().predict_proba(X[:,self.omc_mask]))
        except:
            return (super().predict_proba(X))



class XGBRegressorOMC(XGBRegressor):

    def __init__(
            self,
            *,
            k_omc=10,
            static_feature_position=[],
            **kwargs: Any,
    ):
        self.k_omc = k_omc
        self.static_feature_position = static_feature_position
        super().__init__(
            **kwargs
        )

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params["num_parallel_tree"] = self.n_estimators
        return params

    def get_num_boosting_rounds(self) -> int:
        return 100

    def fit(self, X, y, sample_weight=None) -> "XGBRegressorOMC":
        self.selector = SelectKBest(k=self.k_omc).fit(X, y)
        self.omc_mask = self.selector.get_support()
        for i in self.static_feature_position:
            self.omc_mask[i] = True
        try:
            del i
        except:
            pass
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        args["X"] = X[:,self.omc_mask]
        super().fit(**args)
        return self

    def predict(self, X):
        return (super().predict(X[:,self.omc_mask]))



class XGBClassifierOMC(XGBClassifier):

    def __init__(
            self,
            *,
            k_omc=10,
            static_feature_position=[],
            **kwargs: Any,
    ):
        self.k_omc = k_omc
        self.static_feature_position = static_feature_position
        super().__init__(
            **kwargs
        )

    def get_xgb_params(self) -> Dict[str, Any]:
        params = super().get_xgb_params()
        params["num_parallel_tree"] = self.n_estimators
        return params

    def get_num_boosting_rounds(self) -> int:
        return 100
    

    def fit(self, X, y, sample_weight=None) -> "XGBClassifierOMC":
        self.selector = SelectKBest(k=self.k_omc).fit(X, y)
        self.omc_mask = self.selector.get_support()
        for i in self.static_feature_position:
            self.omc_mask[i] = True
        try:
            del i
        except:
            pass
        args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        args["X"] = X[:,self.omc_mask]
        super().fit(**args)
        return self

    def predict(self, X):
        try:
            return (super().predict(X[:,self.omc_mask]))
        except:
            return (super().predict(X))

    def predict_proba(self, X):
        try:
            return (super().predict_proba(X[:,self.omc_mask]))
        except:
            return (super().predict_proba(X))





class SVROMC(SVR):
    def __init__(
            self,
            *,
            kernel="rbf",
            degree=3,
            gamma="scale",
            coef0=0.0,
            tol=1e-3,
            C=1.0,
            epsilon=0.1,
            shrinking=True,
            cache_size=200,
            verbose=False,
            max_iter=-1,
            k_omc=10,
            static_feature_position=[],
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            epsilon=epsilon,
            verbose=verbose,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
        )
        self.k_omc = k_omc
        self.static_feature_position = static_feature_position

    def fit(self, X, y, sample_weight=None):
        self.selector = SelectKBest(k=self.k_omc).fit(X, y)
        self.omc_mask = self.selector.get_support()
        for i in self.static_feature_position:
            self.omc_mask[i] = True
        return (super().fit(X[:,self.omc_mask], y, sample_weight))

    def predict(self, X):
        return (super().predict(X[:,self.omc_mask]))



class SVCOMC(SVC):
    def __init__(
            self,
            *,
            C=1.0,
            kernel="rbf",
            degree=3,
            gamma="scale",
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape="ovr",
            break_ties=False,
            random_state=None,
            k_omc=10,
            static_feature_position=[],
    ):
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )
        self.k_omc = k_omc
        self.static_feature_position = static_feature_position

    def fit(self, X, y, sample_weight=None):
        self.selector = SelectKBest(k=self.k_omc).fit(X, y)
        self.omc_mask = self.selector.get_support()
        for i in self.static_feature_position:
            self.omc_mask[i] = True
        return (super().fit(X[:,self.omc_mask], y, sample_weight))

    def predict(self, X):
        try:
            return (super().predict(X[:,self.omc_mask]))
        except:
            return (super().predict(X))

    def predict_proba(self, X):
        try:
            return (super().predict_proba(X[:,self.omc_mask]))
        except:
            return (super().predict_proba(X))




class LGBMRegressorOMC(LGBMRegressor):
    def __init__(
            self,
            *,
            boosting_type: str = 'gbdt',
            num_leaves: int = 31, 
            max_depth: int = -1,
            learning_rate: float = 0.1,
            n_estimators: int = 100,
            subsample_for_bin: int = 200000,
            objective=None,
            class_weight=None,
            min_split_gain: float = 0.0,
            min_child_weight: float = 0.001,
            min_child_samples: int = 20,
            subsample: float = 1.0,
            subsample_freq: int = 0,
            colsample_bytree: float = 1.0,
            reg_alpha: float = 0.0,
            reg_lambda: float = 0.0,
            random_state=None,
            n_jobs: int = -1,
            silent='warn',
            importance_type: str = 'split',
            k_omc=10,
            static_feature_position=[],
    ):
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            silent=silent,
            importance_type=importance_type,
        )
        self.k_omc = k_omc
        self.static_feature_position = static_feature_position

    def fit(self, X, y, sample_weight=None):
        self.selector = SelectKBest(k=self.k_omc).fit(X, y)
        self.omc_mask = self.selector.get_support()
        for i in self.static_feature_position:
            self.omc_mask[i] = True
        return (super().fit(X[:,self.omc_mask], y, sample_weight))

    def predict(self, X):
        return (super().predict(X[:,self.omc_mask]))





class LGBMClassifierOMC(LGBMClassifier):
    def __init__(
            self,
            *,
            boosting_type: str = 'gbdt',
            num_leaves: int = 31,
            max_depth: int = -1,
            learning_rate: float = 0.1,
            n_estimators: int = 100,
            subsample_for_bin: int = 200000,
            objective=None,
            class_weight=None,
            min_split_gain: float = 0.0,
            min_child_weight: float = 0.001,
            min_child_samples: int = 20,
            subsample: float = 1.0,
            subsample_freq: int = 0,
            colsample_bytree: float = 1.0,
            reg_alpha: float = 0.0,
            reg_lambda: float = 0.0,
            random_state=None,
            n_jobs: int = -1,
            importance_type: str = 'split',
            k_omc=10,
            static_feature_position=[],
    ):
        super().__init__(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type,
        )
        self.k_omc = k_omc
        self.static_feature_position = static_feature_position

    def fit(self, X, y, sample_weight=None):
        self.selector = SelectKBest(k=self.k_omc).fit(X, y)
        self.omc_mask = self.selector.get_support()
        for i in self.static_feature_position:
            self.omc_mask[i] = True
        return (super().fit(X[:,self.omc_mask], y, sample_weight))

    def predict(
            self,
            X,
            raw_score=False,
            start_iteration=0,
            num_iteration=None,
            pred_leaf=False,
            pred_contrib=False,
            validate_features=False,
            **kwargs
    ):
        try:
            return (
                super().predict(
                    X=X[:,self.omc_mask],
                    raw_score=raw_score,
                    start_iteration=start_iteration,
                    num_iteration=num_iteration,
                    pred_leaf=pred_leaf,
                    pred_contrib=pred_contrib,
                    validate_features=validate_features,
                    **kwargs
                )
            )
        except:
            return (
                super().predict(
                    X=X,
                    raw_score=raw_score,
                    start_iteration=start_iteration,
                    num_iteration=num_iteration,
                    pred_leaf=pred_leaf,
                    pred_contrib=pred_contrib,
                    validate_features=validate_features,
                    **kwargs
                )
            )

    def predict_proba(
            self,
            X,
            raw_score=False,
            start_iteration=0,
            num_iteration=None,
            pred_leaf=False,
            pred_contrib=False,
            validate_features=False,
            **kwargs
    ):
        try:
            return (
                super().predict_proba(
                    X=X[:,self.omc_mask],
                    raw_score=raw_score,
                    start_iteration=start_iteration,
                    num_iteration=num_iteration,
                    pred_leaf=pred_leaf,
                    pred_contrib=pred_contrib,
                    validate_features=validate_features,
                    **kwargs
                )
            )
        except:
            return (
                super().predict_proba(
                    X=X,
                    raw_score=raw_score,
                    start_iteration=start_iteration,
                    num_iteration=num_iteration,
                    pred_leaf=pred_leaf,
                    pred_contrib=pred_contrib,
                    validate_features=validate_features,
                    **kwargs
                )
            )


