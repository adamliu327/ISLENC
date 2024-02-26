import numpy as np
import pandas as pd
import math
from enum import Enum
from collections import Counter
from skmultiflow.trees import HoeffdingTreeClassifier
from scipy.stats import poisson
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import sys

sys.path.append("../")

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from utils.data_structure import ClassificationModel
from ISLENCForest import ISLENCForest

__all__ = ["base_learner", "ECSMiner"]


class base_learner(Enum):
    KMEANS = 1
    KNN = 2
    DECISION_TREE = 3


class classification_ensemble_type(Enum):
    subset_data = 1
    subset_feature = 2


class ISLENCModel:
    NEW_CLASS = -1

    def __init__(self,
                 X_offline,
                 y_offline,
                 params):
        self.verbose = params['verbose'] if 'verbose' in params else True
        self.X_offline = X_offline
        self.y_offline = y_offline
        self.params = params
        self.rng = np.random.RandomState(params['random_state']) if 'random_state' in params else np.random.RandomState(
            1)
        self.random_seed_count = 0

        ## ---- data characteristics
        self.class_list = np.unique(y_offline)
        self.f = X_offline.shape[1]  ## --- number of features

        ## ---- mainflow configurations
        self.anomaly_label = params['anomaly_label'] if 'anomaly_label' in params else ISLENCModel.NEW_CLASS
        self.S_chunk_size = params['S_chunk_size'] if 'S_chunk_size' in params else 2000
        self.T_l = params['T_l'] if 'T_l' in params else 1000

        ## ---- classification model configurations
        self.classification_ensemble_type = params[
            'classification_ensemble_type'] if 'classification_ensemble_type' in params else classification_ensemble_type.subset_feature

        ## --- mu: the ratio of features selected from the original feature set
        self.mu = params['feature_ratio_mu'] if 'feature_ratio_mu' in params else 0.7
        self.m = params['M_clsfy_ensemble_size']  ## --- k: number of classifiers
        self.min_lambda = params['min_lambda']  ## --- min_lambda: minimum value of lambda
        self.max_lambda = params['max_lambda']

        self.sliding_window_per_class = {}  ## --- key: class, value: sliding window data
        self.random_subspace_size = {}  ## --- key: class, value: random subspace size $r$

        ## ---- model in memory
        self.ensemble_anomaly_detectors = []
        self.ensemble_classification_models = []

        self.recent_y_for_lambda_size = 1000
        self.recent_y_for_lambda = []

        ## ---- buffer in memory
        self._init_buffer()

        ## ---- for evaluation
        self.X_online = []
        self.y_online = []
        self.y_online_pred = []  ## 从buf中移出的时候，通过timestamp定位，更改predicted_label

        return

    def set_anomaly_forest_params(self, params):
        self.anomaly_forest_params = params
        self.is_per_class_anomaly_detector = params[
            'per_class_anomaly_detector'] if 'per_class_anomaly_detector' in params else False
        self.per_class_anomaly_train_buffer = {}  ## --- key: class, value: buffer for training anomaly detector
        logger.info("self.is_per_class_anomaly_detector: %s" % self.is_per_class_anomaly_detector)
        return

    def train_offline(self):
        self.train_anomaly_detection_offline()
        logger.info("offline learning anomaly detection model finished!")

        self.train_classification_offline()
        logger.info("offline learning classification model finished!")
        return

    def _init_buffer(self):
        self.buf = []  # -- also called short_mem
        self.U_unlabeled_buf = []  ## --- buffer for unlabeled data
        self.L_labeled_buf_X = []  ## --- buffer for labeled data
        self.L_labeled_buf_y = []
        logger.info("* Buffer initialized")
        return

    def train_anomaly_detection_offline(self):
        if self.is_per_class_anomaly_detector:
            self._train_anomaly_detector_per_class_offline()
        else:
            self._train_anomaly_detection_offline()
        return

    def _train_anomaly_detection_offline(self):
        """
        Do not split classes, train one anomaly detector for all classes
        """
        self.ensemble_anomaly_detectors.clear()
        self.anomaly_detector = ISLENCForest(feature_data=self.X_offline, params=self.anomaly_forest_params)
        self.anomaly_detector.build_forest()
        self.anomaly_detector.init_buffer()
        self.ensemble_anomaly_detectors.append(self.anomaly_detector)
        pass

    def _train_anomaly_detector_per_class_offline(self):
        """
        Split classes, train one anomaly detector for each class
        """
        self.ensemble_anomaly_detectors.clear()
        for class_label in self.class_list:
            class_X = self.X_offline[self.y_offline == class_label]
            class_anomaly_detector = ISLENCForest(feature_data=class_X, params=self.anomaly_forest_params)
            class_anomaly_detector.build_forest()
            class_anomaly_detector.init_buffer()
            self.ensemble_anomaly_detectors.append(class_anomaly_detector)
        pass

    def push_predict_label_for_online_instance(self, predicted_label):
        self.y_online_pred.append(predicted_label)
        return

    def _get_train_subset_feature_ids(self):
        """ Randomly choose a subset of features (like ROSE)"""
        r = round(self.mu * self.f + (1 - self.mu) * self.f * np.random.randn() / 2)
        r = min(r, self.f)
        ## randomly select r features from f features
        orig_feature_ids = np.arange(self.f)
        np.random.seed(self.random_seed_count)
        self.random_seed_count += 1
        selected_feature_ids = np.random.choice(orig_feature_ids, r, replace=False)
        selected_feature_ids = np.sort(selected_feature_ids)
        return selected_feature_ids

    def train_classification_offline(self):
        self.ensemble_classification_models = []
        if self.classification_ensemble_type == classification_ensemble_type.subset_feature:
            for j in range(self.m):
                selected_feature_ids = self._get_train_subset_feature_ids()
                logger.info("selected_feature_ids: {}".format(selected_feature_ids))
                train_X = self.X_offline
                train_y = self.y_offline.copy()
                clsfy_model = self._train_single_classification_model(train_X, train_y, selected_feature_ids)
                clsfy_model.set_id(j + 1)  ## --- model_id starts from 1
                self.ensemble_classification_models.append(clsfy_model)
        return self.ensemble_classification_models

    def update_classification_online(self, x, y):
        _lambda = self._compute_lambda(y)
        _lambda = min(_lambda, self.max_lambda)
        for j in range(self.m):
            sample_weight = poisson.rvs(_lambda, size=1)
            sample_weight = sample_weight * self.ensemble_classification_models[j].model_id / self.m
            self.ensemble_classification_models[j] = self._online_update_single_classification_model(
                self.ensemble_classification_models[j], x, y, sample_weight)
        return self.ensemble_classification_models

    def predict_online_instance(self, x):
        ## majority voting by ensemble
        y_pred_list = []
        for j in range(self.m):
            y_pred = self._predict_single_classification_model(self.ensemble_classification_models[j], x)
            y_pred_list.append(y_pred)
        y_pred_list = np.array(y_pred_list)
        y_pred = Counter(y_pred_list).most_common(1)[0][0]
        return y_pred

    def _train_single_classification_model(self, train_X, train_y, selected_feature_ids=None):
        ht = HoeffdingTreeClassifier()
        train_X = train_X[:, selected_feature_ids]
        ht.fit(train_X, train_y)
        return ClassificationModel(ht, selected_feature_ids)

    def _online_update_single_classification_model(self, classification_model, x, y, sample_weight):
        """x: 1d array"""
        x = x.reshape(1, -1)
        y = y.reshape(-1)
        sample_weight = sample_weight.reshape(-1)
        ## --- select features
        x = x[:, classification_model.feature_index_list]
        classification_model.model.partial_fit(x, y, sample_weight=sample_weight)
        return classification_model

    def _predict_single_classification_model(self, classification_model, x):
        x = x.reshape(1, -1)
        x = x[:, classification_model.feature_index_list]
        return classification_model.model.predict(x)[0]

    def _manage_recent_y_for_lambda(self, y):
        if len(self.recent_y_for_lambda) >= self.recent_y_for_lambda_size:
            self.recent_y_for_lambda.pop(0)
        self.recent_y_for_lambda.append(y)
        return

    def _compute_lambda(self, y):
        self._manage_recent_y_for_lambda(y)
        most_common_y = Counter(self.recent_y_for_lambda).most_common(1)[0][0]
        most_common_y_count = Counter(self.recent_y_for_lambda).most_common(1)[0][1]
        this_y_count = Counter(self.recent_y_for_lambda)[y]
        _lambda = self.min_lambda + math.log10(most_common_y_count / this_y_count) * self.min_lambda
        return _lambda

    ## ---- anomaly detection model train/update/predict 
    def anomaly_detection_update(self, x, y=None):
        if y not in self.class_list:
            logger.info("novel class %s found!" % y)
            self.class_list = np.concatenate([self.class_list, [y]])
        if self.is_per_class_anomaly_detector:
            _class_id = np.argwhere(self.class_list == y)[0][0]
            if len(self.ensemble_anomaly_detectors) > _class_id:
                self.ensemble_anomaly_detectors[_class_id].update(x)
            else:
                logger.debug("class %s anomaly detector not found" % y)
                if y in self.per_class_anomaly_train_buffer.keys():
                    self.per_class_anomaly_train_buffer[y].append(x)
                    logger.debug(
                        "class %s anomaly detector buffer size: %s" % (y, len(self.per_class_anomaly_train_buffer[y])))
                    if len(self.per_class_anomaly_train_buffer[y]) >= self.anomaly_forest_params['min_tree_size']:
                        logger.info("Begin to train class %s anomaly detector" % y)
                        class_X = np.array(self.per_class_anomaly_train_buffer[y])
                        class_anomaly_detector = ISLENCForest(feature_data=class_X, params=self.anomaly_forest_params)
                        class_anomaly_detector.build_forest()
                        class_anomaly_detector.init_buffer()
                        self.ensemble_anomaly_detectors.append(class_anomaly_detector)
                        self.per_class_anomaly_train_buffer.pop(y)
                else:
                    self.per_class_anomaly_train_buffer[y] = [x]
        else:
            self.ensemble_anomaly_detectors[0].update(x)
        return

    def anomaly_detection_predict(self, x):
        if self.is_per_class_anomaly_detector:
            _anomaly_flag_list = []
            _anomaly_score_list = []
            for class_anomaly_detector in self.ensemble_anomaly_detectors:
                _anomaly_flag, _anomaly_score = class_anomaly_detector.predict_anomaly_iForest(x)
                _anomaly_flag_list.append(_anomaly_flag)
                _anomaly_score_list.append(_anomaly_score)
            _anomaly_flag_list = np.array(_anomaly_flag_list)
            _anomaly_score_list = np.array(_anomaly_score_list)
            _anomaly_flag = np.all(_anomaly_flag_list)
            return _anomaly_flag
        else:
            _anomaly_flag, _anomaly_score = self.ensemble_anomaly_detectors[0].predict_anomaly_iForest(x)
            return _anomaly_flag

    ## ---- buffer management
    def push_U_unlabeled_buf(self, x, y):  ## --- one sample at a time
        self.U_unlabeled_buf.append((x, y))
        return

    def pop_U_unlabeled_buf(self):  ## --- pop the first sample
        return self.U_unlabeled_buf.pop(0)

    def get_U_unlabeled_buf_size(self):
        return len(self.U_unlabeled_buf)

    def push_L_labeled_buf(self, x, y):  ## --- one sample at a time
        if isinstance(x, dict):
            x = np.array(list(x.values()))
        self.L_labeled_buf_X.append(x)
        self.L_labeled_buf_y.append(y)
        return

    def gelabel_delay_T_labeled_buf_size(self):
        return len(self.L_labeled_buf_X)

    def clear_L_labeled_buf(self):
        self.L_labeled_buf_X.clear()
        self.L_labeled_buf_y.clear()
        return

    def _filter_buffer_label_delay_T(self):
        for instance in self.buf:
            if (self.online_sample_counter - instance.timestamp > self.T_l):
                self.buf.remove(instance)
                if self.verbose > 0: logger.debug(
                    "T_l filter buffer condition met: Sample " + str(instance.timestamp) + " removed from buffer")
        return

    ## ---- evaluation
    def push_online_instance_for_validation(self, x, y):
        self.X_online.append(x)
        self.y_online.append(y)
        return

    def compute_auc(self, y_true, y_scores):
        # Compute ROC curve and AUC-ROC for each class
        n_classes = len(np.unique(y_true))
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and AUC-ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        return roc_auc["micro"]

    def compute_kappa(self, y_true, y_pred):
        assert len(y_true) == len(y_pred)
        y_list = np.unique(y_true)
        y_list = np.sort(y_list)
        n = len(y_true)
        a = 0
        for y in y_list:
            a += np.sum(np.logical_and(y_true == y, y_pred == y))
        a = a / n
        b = 0
        for y in y_list:
            b += np.sum(y_true == y) * np.sum(y_pred == y)
        b = b / (n * n)
        _kappa = (a - b) / (1 - b)

        return _kappa

    def get_metrics(self, chunk_y_true, chunk_y_pred, prev_labels, _exist_metric):
        _metric_dict = {}
        cunrrenlabel_delay_Tabels = np.unique(chunk_y_true)
        new_labels = np.setdiff1d(cunrrenlabel_delay_Tabels, prev_labels)
        new_classes_true = np.isin(chunk_y_true, new_labels)
        exist_classes_true = np.isin(chunk_y_true, prev_labels)
        new_classes_pred = np.equal(chunk_y_pred, self.NEW_CLASS)
        # new_classes_pred = np.isin(chunk_y_pred,[self.NEW_CLASS,self.UNKNOWN_CLASS])  ## 忽略了 UNKNOWN_CLASS，认为也是novelty
        exist_classes_pred = np.isin(chunk_y_pred, prev_labels)

        _metric_dict['N'] = len(chunk_y_true)
        _metric_dict['N_newclass'] = np.sum(new_classes_true)
        _metric_dict['N_existclass'] = np.sum(exist_classes_true)
        ## f_p: existing classes predicted as new classes
        ## f_n: new classes predicted as existing classes
        ## f_e: existing classes wrongly predicted
        _metric_dict['f_total'] = np.sum(np.not_equal(chunk_y_true, chunk_y_pred))

        _metric_dict['f_p'] = np.sum(np.logical_and(exist_classes_true, new_classes_pred))
        _metric_dict['f_n'] = np.sum(np.logical_and(new_classes_true, exist_classes_pred))
        true_exist_predict_exist = np.logical_and(exist_classes_true, exist_classes_pred)
        _metric_dict['f_e'] = np.sum(np.logical_and(np.not_equal(chunk_y_true, chunk_y_pred), true_exist_predict_exist))

        _metric_dict['M_new'] = _metric_dict['f_n'] / (_metric_dict['N_newclass'])
        _metric_dict['F_new'] = _metric_dict['f_p'] / (_metric_dict['N_existclass'])
        _metric_dict['Err'] = (_metric_dict['f_p'] + _metric_dict['f_n'] + _metric_dict['f_e']) / (_metric_dict['N'])

        _exist_metric['N'] += _metric_dict['N']
        _exist_metric['N_newclass'] += _metric_dict['N_newclass']
        _exist_metric['N_existclass'] += _metric_dict['N_existclass']
        _exist_metric['f_total'] += _metric_dict['f_total']
        _exist_metric['f_p'] += _metric_dict['f_p']
        _exist_metric['f_n'] += _metric_dict['f_n']
        _exist_metric['f_e'] += _metric_dict['f_e']
        _exist_metric['M_new'] = _exist_metric['f_n'] / (_exist_metric['N_newclass'])
        _exist_metric['F_new'] = _exist_metric['f_p'] / (_exist_metric['N_existclass'])
        _exist_metric['Err'] = (_exist_metric['f_p'] + _exist_metric['f_n'] + _exist_metric['f_e']) / (
            _exist_metric['N'])
        _exist_metric['kappa'] = self.compute_kappa(chunk_y_true, chunk_y_pred)
        return _metric_dict, _exist_metric, cunrrenlabel_delay_Tabels

    def get_validation_metric_online(self):
        chunk_size = len(self.y_online) // self.S_chunk_size
        prev_labels = np.unique(self.y_offline)
        _chunk_metric_dict_list = []
        _exist_metric_dict_list = []
        _exist_metric = {'chunk_id': 0, 'N': 0, 'N_newclass': 0, 'N_existclass': 0, 'f_total': 0, 'f_p': 0, 'f_n': 0,
                         'f_e': 0, 'M_new': 0, 'F_new': 0, 'Err': 0}
        for i in range(chunk_size):
            _chunk_metric_dict = {'chunk_id': i}
            _exist_metric['chunk_id'] = i
            chunk_y_true = np.array(self.y_online[i * self.S_chunk_size:(i + 1) * self.S_chunk_size])
            chunk_y_pred = np.array(self.y_online_pred[i * self.S_chunk_size:(i + 1) * self.S_chunk_size])
            _metric, _exist_metric, cunrrent_labels = self.get_metrics(chunk_y_true, chunk_y_pred, prev_labels,
                                                                       _exist_metric)
            _chunk_metric_dict = {**_chunk_metric_dict, **_metric}
            if self.verbose: logger.info("chunk_metric_dict: %s", _chunk_metric_dict)
            if self.verbose: logger.info("current labels: %s", cunrrent_labels)
            for l in cunrrent_labels:
                if l not in prev_labels:
                    prev_labels = np.concatenate((prev_labels, [l]))
            _chunk_metric_dict_list.append(_chunk_metric_dict)
            _exist_metric_dict_list.append(_exist_metric.copy())
        self._metric_df = pd.DataFrame(_chunk_metric_dict_list)
        self._accumulated_metric_df = pd.DataFrame(_exist_metric_dict_list)
        return self._accumulated_metric_df
