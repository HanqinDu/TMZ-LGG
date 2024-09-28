#!/usr/bin/python
#-*- coding:utf-8 -*-

import re
import pickle
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.feature_selection import SelectKBest
from lightgbm.sklearn import LightGBMError
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone
from ModelPreset import getModelByName
from PredictionSummary import *


# monkey fix
np.int = np.int64


def predictOutOfSample(model, X, y, do_sample_weight=True, kfold=10, random_seed=0):
    # set random seed and clone model
    np.random.seed(random_seed)
    model = clone(model)

    # prepare output container
    y_pred = np.zeros(len(y))
    y_proba = np.zeros((len(y), len(np.unique(y))))
    y_in_sample_pred = []
    y_in_sample_proba = []
    best_params = []

    # cross validation
    cv = StratifiedKFold(random_state=random_seed, n_splits=kfold, shuffle=True)
    cv_test_fold = []

    for train_index, test_index in cv.split(X, y):
        Xtrain = X[train_index]
        Xtest = X[test_index]
        ytrain = y[train_index]
        ytest = y[test_index]

        cv_test_fold.append(test_index)

        # weight y_train
        if do_sample_weight:
            sample_weight = compute_sample_weight("balanced", ytrain)
        else:
            sample_weight = None

        # fit model (handle the LGBM err due to too small min_child_samples)
        try:
            model.fit(Xtrain, ytrain, sample_weight=sample_weight)
        except LightGBMError as err:
            print(err)
            adjusted_min_child_samples = int(round(Xtrain.shape[0] / float(model.num_leaves)))
            model.min_child_samples = adjusted_min_child_samples
            model.fit(Xtrain, ytrain, sample_weight=sample_weight)

        # record params
        if isinstance(model, BayesSearchCV) or isinstance(model, GridSearchCV):
            best_params.append(model.best_params_)
        else:
            best_params.append(model.get_params())

        # get out-of-sample-prediction and in-sample-prediction
        y_pred[test_index] = model.predict(Xtest)
        y_in_sample_pred.append(model.predict(Xtrain))

        try:
            y_proba[test_index] = model.predict_proba(Xtest)
            y_in_sample_proba.append(model.predict_proba(Xtrain))
        except:
            y_proba = None

    # return the order of class
    try:
        y_class = model.classes_
    except:
        y_class = None

    return PredictionSummary(y, y_pred, y_proba, y_in_sample_pred, y_in_sample_proba, y_class, cv_test_fold, best_params, seed=random_seed)


if __name__=='__main__':
    ## Declaration of arguments to argparse
    parser = argparse.ArgumentParser(add_help = True)
    parser.add_argument('-i','--data', action = 'store' , dest = 'dataset', required = True, help = 'Drug-cancer type molecular profile dataset')
    parser.add_argument('-o', '--output', action='store', dest='output', required=True, help='prediction_output')
    parser.add_argument('-c','--classifier', action = 'store', dest = 'classifier', default = 'CART-C', help = 'Classifier used')
    parser.add_argument('-s','--seed', action = 'store', dest = 'seeds', type=int, default = [1, 2, 3, 4, 5], nargs='+', help = 'random seeds to be applied, devided by space. (e.g. -s 1 2 3 4)')

    # for Bayesopt
    parser.add_argument('-j','--njob', action = 'store', dest = 'njob', type=int, default = 12, help = 'number of cores to use for tunning')
    parser.add_argument('-p','--npoint', action = 'store', dest = 'npoint', type=int, default = 4, help = 'number of points to train in each iteration of tunning')
    parser.add_argument('-v','--outer', action = 'store', dest = 'outer', type=int, default = 10, help = 'number of outer-folds for validation')
    parser.add_argument('-f','--inner', action = 'store', dest = 'inner', type=int, default = 5, help = 'number of inner-folds for tunning')
    parser.add_argument('-r','--niter', action = 'store', dest = 'niter', type=int, default = 5, help = 'number of iteration to use for tunning')

    # for OMC
    parser.add_argument('-a','--allfeatures', action = 'store_true', dest = 'allfeatures', help = 'Whether to apply all features for validation')
    parser.add_argument('-e', '--static', action='store', dest='static_feature', default=[], nargs='*', help='feature name that statis to be include during OMC')

    # parse argument
    arguments = parser.parse_args()


    ## load data (feather or csv)
    # load as dataframe
    if (re.search("\.feather$", arguments.dataset)):
        data = pd.read_feather(arguments.dataset)
    else:
        data = pd.read_csv(arguments.dataset)

    # split data as design matrix and label
    X = data.drop(['bcr_patient_barcode', 'drug_name', 'measure_of_response', 'Patient'], axis=1)
    X_column = X.columns
    X = X.values
    y = data["measure_of_response"]


    ## preprocess
    # remove constant column
    X_column = X_column[~np.all(X[1:] == X[:-1], axis=0)]
    X = X[:, ~np.all(X[1:] == X[:-1], axis=0)]

    static_feature_pos = []
    if arguments.static_feature:
        for feature in arguments.static_feature:
            static_feature_pos.append(list(X_column).index(feature))


    ## loop random seeds
    prediction_list_collection = []
    k_list_collection = []

    for random_seed in arguments.seeds:
        print(random_seed)
        opt = getModelByName(
            classifier_name = arguments.classifier,
            OMC = (not arguments.allfeatures),
            nfeature = np.shape(X)[1]+100,
            nsample = len(y),
            random_seed = random_seed,
            njob = arguments.njob,
            npoint = arguments.npoint,
            cv = arguments.inner,
            niter = arguments.niter,
            static_feature_pos = static_feature_pos
        )

        prediction = predictOutOfSample(opt, X, y, do_sample_weight=True, kfold=arguments.outer, random_seed=random_seed)
        prediction_list_collection.append(prediction)


    ## save prediction and OMC-k list
    savePrediction(arguments.output, prediction_list_collection)

