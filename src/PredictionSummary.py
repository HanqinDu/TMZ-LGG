#!/usr/bin/python
#-*- coding:utf-8 -*-

########################################################################


import numpy as np
import sklearn.metrics
import pickle
from random import choices
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor



def strictMCC(y_truth, y_pred):
    if len(np.unique(y_truth)) < 2:
        return float("nan")
    else:
        return(sklearn.metrics.matthews_corrcoef(y_truth, y_pred))



def evaluateOptimalThreshold(y_truth, y_pred, larger=True):
    if max(y_pred) == min(y_pred):
        return (y_pred[0])

    candidate_threshold = np.arange(
        min(y_pred) - ((max(y_pred) - min(y_pred)) / 10),
        max(y_pred) + ((max(y_pred) - min(y_pred)) / 10),
        (max(y_pred) - min(y_pred)) / 20
    )

    if larger:
        mcc_list = [sklearn.metrics.matthews_corrcoef(y_truth, y_pred > t) for t in candidate_threshold]
    else:
        mcc_list = [sklearn.metrics.matthews_corrcoef(y_truth, y_pred < t) for t in candidate_threshold]

    index_best_mcc = int(np.median([i for i, k in enumerate(mcc_list) if k == max(mcc_list)]))

    return (candidate_threshold[index_best_mcc])



class PredictionSummary:
    def __init__(self, y_truth, y_pred, y_proba=None, y_in_sample_pred=None, y_in_sample_proba=None, y_class=None,
                 cv_fold=None, model_params=None, target_class=None, y_truth_binary=None, y_proba_binary=None,
                 y_in_sample_proba_binary=None, y_pred_binary=None, y_pred_binery_threshold_tuned=None, cv_notation=None,
                 notation_=None):
        self.y_truth = np.array(y_truth)
        self.y_pred = np.array(y_pred)
        self.y_proba = np.array(y_proba)
        self.y_in_sample_pred = y_in_sample_pred
        self.y_in_sample_proba = y_in_sample_proba
        self.y_class = y_class
        self.cv_fold = cv_fold
        self.model_params = model_params
        self.target_class = target_class
        self.y_truth_binary = y_truth_binary
        self.y_proba_binary = y_proba_binary
        self.y_in_sample_proba_binary = y_in_sample_proba_binary
        self.y_pred_binary = y_pred_binary
        self.y_pred_binery_threshold_tuned = y_pred_binery_threshold_tuned
        self.cv_notation = cv_notation
        self.notation_ = notation_


    def toDict(self):
        return ({"y_truth": self.y_truth,
                 "y_pred": self.y_pred,
                 "y_proba": self.y_proba,
                 "y_in_sample_pred": self.y_in_sample_pred,
                 "y_in_sample_proba": self.y_in_sample_proba,
                 "y_class": self.y_class,
                 "cv_fold": self.cv_fold,
                 "model_params": self.model_params,
                 "target_class": self.target_class,
                 "y_truth_binary": self.y_truth_binary,
                 "y_proba_binary": self.y_proba_binary,
                 "y_in_sample_proba_binary": self.y_in_sample_proba_binary,
                 "y_pred_binary": self.y_pred_binary,
                 "y_pred_binery_threshold_tuned": self.y_pred_binery_threshold_tuned,
                 "cv_notation": self.cv_notation,
                 "notation_": self.notation_})

    # need to binarize result first
    def thresholdTunning(self):
        if self.target_class is None:
            return (False)

        self.y_pred_binery_threshold_tuned = np.zeros(len(self.y_pred))

        if self.y_class is not None:
            for i, fold in enumerate(self.cv_fold):
                opt_threshold = evaluateOptimalThreshold(
                    [x for k, x in enumerate(self.y_truth_binary) if k not in fold], self.y_in_sample_proba_binary[i],
                    larger=True)
                self.y_pred_binery_threshold_tuned[fold] = self.y_proba_binary[fold] > opt_threshold
        #             else:
        #                 for i, fold in enumerate(self.cv_fold):
        #                     opt_threshold = evaluateOptimalThreshold([x for k,x in enumerate(self.y_truth_binary) if k not in fold], self.y_in_sample_proba_binary[i], larger=True)
        #                     self.y_pred_binery_threshold_tuned[fold] = self.y_proba_binary[fold] > opt_threshold

        else:
            if max(self.y_truth) in self.target_class:
                for i, fold in enumerate(self.cv_fold):
                    opt_threshold = evaluateOptimalThreshold(
                        [x for i, x in enumerate(self.y_truth_binary) if i not in fold], self.y_in_sample_pred[i],
                        larger=True)
                    self.y_pred_binery_threshold_tuned[fold] = self.y_pred[fold] > opt_threshold
            else:
                for i, fold in enumerate(self.cv_fold):
                    opt_threshold = evaluateOptimalThreshold(
                        [x for i, x in enumerate(self.y_truth_binary) if i not in fold], self.y_in_sample_pred[i],
                        larger=False)
                    self.y_pred_binery_threshold_tuned[fold] = self.y_pred[fold] < opt_threshold

        return (True)

    def binarizeResult(self, target_class):
        self.target_class = target_class
        self.y_truth_binary = np.array([i in target_class for i in self.y_truth])
        if self.y_class is not None:
            self.y_proba_binary = np.sum(self.y_proba[:, [i in target_class for i in self.y_class]], axis=1)
            self.y_in_sample_proba_binary = [np.sum(p[:, [i in target_class for i in self.y_class]], axis=1) for p
                                             in self.y_in_sample_proba]
            self.y_pred_binary = np.array([i in target_class for i in self.y_pred])
        else:
            if max(self.y_truth) in self.target_class:
                self.y_proba_binary = self.y_pred
            else:
                self.y_proba_binary = -self.y_pred

        self.thresholdTunning()

        return (True)

    def getMSE(self):
        return (sklearn.metrics.mean_squared_error(self.y_truth, self.y_pred))

    def getMAE(self):
        return (sklearn.metrics.mean_absolute_error(self.y_truth, self.y_pred))

    def getF1(self):
        return (sklearn.metrics.f1_score(self.y_truth, self.y_pred))

    def getR2(self):
        return (sklearn.metrics.r2_score(self.y_truth, self.y_pred))

    def getCM(self, use_tuned_result=True):
        if use_tuned_result:
            return (sklearn.metrics.confusion_matrix(self.y_truth_binary, self.y_pred_binery_threshold_tuned))
        else:
            return (sklearn.metrics.confusion_matrix(self.y_truth, self.y_pred))

    def getMCC(self, use_tuned_result=True, subset=None):
        if subset is not None:
            if use_tuned_result:
                return (strictMCC(self.y_truth_binary[subset], self.y_pred_binery_threshold_tuned[subset]))
            else:
                return (strictMCC(self.y_truth_binary[subset], self.y_pred_binary[subset]))
        if use_tuned_result:
            return (strictMCC(self.y_truth_binary, self.y_pred_binery_threshold_tuned))
        else:
            return (strictMCC(self.y_truth_binary, self.y_pred_binary))

    def getAUC(self, subset=None):
        if subset is not None:
            if len(np.unique(self.y_truth_binary[subset])) < 2:
                return float("nan")
            else:
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.y_truth_binary[subset], self.y_proba_binary[subset])
        else:
            if len(np.unique(self.y_truth_binary)) < 2:
                return float("nan")
            else:
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.y_truth_binary, self.y_proba_binary)

        return (sklearn.metrics.auc(fpr, tpr))

    def getPRAUC(self, subset=None):
        if subset is not None:
            if len(np.unique(self.y_truth_binary[subset])) < 2:
                return float("nan")
            else:
                precision, recall, aucpr_thresholds = sklearn.metrics.precision_recall_curve(self.y_truth_binary[subset], self.y_proba_binary[subset])
        else:
            if len(np.unique(self.y_truth_binary)) < 2:
                return float("nan")
            else:
                precision, recall, aucpr_thresholds = sklearn.metrics.precision_recall_curve(self.y_truth_binary, self.y_proba_binary)

        return (sklearn.metrics.auc(recall, precision))


    def getPRGAUC(self, subset=None):
        if subset is not None:
            if len(np.unique(self.y_truth_binary[subset])) < 2:
                return float("nan")
            else:
                precision, recall, aucpr_thresholds = sklearn.metrics.precision_recall_curve(self.y_truth_binary[subset], self.y_proba_binary[subset])
        else:
            if len(np.unique(self.y_truth_binary)) < 2:
                return float("nan")
            else:
                precision, recall, aucpr_thresholds = sklearn.metrics.precision_recall_curve(self.y_truth_binary, self.y_proba_binary)

        pi = sum(self.y_truth_binary) / len(self.y_truth_binary)
        recall_gain = np.array([(r - pi) / ((1 - pi) * r) for r in recall])
        precision_gain = np.array([(p - pi) / ((1 - pi) * p) for p in precision])
        recall_gain[recall_gain < 0] = 0
        precision_gain[precision_gain < 0] = 0

        return (sklearn.metrics.auc(recall_gain, precision_gain))


    def getPRAUCBaseline(self, subset=None):
        if subset is not None:
            return (np.mean(self.y_truth_binary[subset]))
        return (np.mean(self.y_truth_binary))


def loadPrediction(filename):
    def dictToObj(prediction):
        if type(prediction) == list:
            return([dictToObj(p) for p in prediction])
        else:
            return(PredictionSummary(**prediction))

    with open(filename, 'rb') as handle:
        performance_collection = pickle.load(handle)

    return(dictToObj(performance_collection))


def savePrediction(filename, prediction_list):
    def objToDict(prediction):
        if type(prediction) == list:
            return([objToDict(p) for p in prediction])
        else:
            return(prediction.toDict())

    with open(filename, 'wb') as handle:
        pickle.dump(objToDict(prediction_list), handle, protocol=pickle.HIGHEST_PROTOCOL)



def callBinarization(prediction, target_class):
    if type(prediction) == list:
        return([callBinarization(p, target_class) for p in prediction])
    else:
        prediction.binarizeResult(target_class)
        return(prediction)


def binarizePredictionList(prediction_list, target_class, ncpu=1):
    if(ncpu==1):
        callBinarization(prediction_list, target_class)
    else:
        with ProcessPoolExecutor(max_workers=ncpu) as pool:
            output = pool.map(callBinarization, prediction_list, [target_class]*len(prediction_list))
        return(output)



def flattenList(multi_level_list):
    output = list()
    def iteration(e):
        if type(e) == list:
            for i in e: iteration(i)
        else:
            output.append(e)
    iteration(multi_level_list)
    return(output)




def BBC_CV(prediction_list, use_tuned_result=True, niter=1000, robust=False):
    # get sample size and configuration number
    sample_size = len(prediction_list[0].y_truth)

    # get parameter of the best model
    index_best_performance = np.argmax([p.getMCC(use_tuned_result=use_tuned_result) for p in prediction_list])
    best_prediction = prediction_list[index_best_performance]

    # calculate L_bbc
    out_samples_performances = list()

    # get y_truth
    y_truth = prediction_list[0].y_truth

    for i in range(0, niter):

        # select subset
        if robust:
            index_in_samples = choices([i for i, x in enumerate(y_truth) if x == 1], k=sum(y_truth==1)-1) + \
                               choices([i for i, x in enumerate(y_truth) if x == 0], k=sum(y_truth==0)-1)
        else:
            index_in_samples = choices(range(0, sample_size), k=sample_size - 1)

        index_out_samples = list(set(range(0, sample_size)).difference(set(index_in_samples)))

        # get performance of in_sample
        performance_in_sample = [p.getMCC(use_tuned_result=use_tuned_result, subset=index_in_samples) for p in prediction_list]

        # evaluate the performance of the configuration with the best in sample performance
        index_best_in_sample = np.argmax(performance_in_sample)
        out_samples_performances.append(prediction_list[index_best_in_sample].getMCC(use_tuned_result=use_tuned_result, subset=index_out_samples))

    return (best_prediction, out_samples_performances)
