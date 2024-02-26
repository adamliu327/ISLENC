import numpy as np
import math
from enum import Enum
from collections import Counter
from skmultiflow.trees import HoeffdingTreeClassifier
from scipy.stats import poisson

import sys

sys.path.append("../")

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import random

from utils.data_structure import ClassificationModel

from ISLENCForest import ISLENCForest


class base_learner(Enum):
    KMEANS = 1
    KNN = 2
    DECISION_TREE = 3


class classification_ensemble_type(Enum):
    subset_data = 1
    subset_feature = 2


class SemiWrapperISLENC:
    UNKNOWN_CLASS = -1
    NEW_CLASS = -2

    def __init__(self,
                 offline_data,
                 online_data,
                 params):
        self.verbose = params['verbose'] if 'verbose' in params else True

        ## ---- online/offline data characteristics
        self.offline_data = offline_data
        self.online_data = online_data
        if offline_data is not None:
            self.is_offline_training = True
            self.class_list = np.unique(offline_data['y'])
        else:
            self.is_offline_training = False
            self.class_list = np.array([])

        self.f = online_data['X'].shape[1]  ## --- number of features

        ## ---- model configurations
        self.anomaly_label = params['anomaly_label'] if 'anomaly_label' in params else SemiWrapperISLENC.NEW_CLASS

        self.main_model_params = params['main_model_params']
        self.ensemble_params = params['ensemble_params']
        self.anomaly_forest_param = params['iforest_params']
        if 'classification_ensemble_type' in self.ensemble_params:
            self.classification_ensemble_type = self.ensemble_params['classification_ensemble_type']
        else:
            self.classification_ensemble_type = classification_ensemble_type.subset_feature
        self.mu = self.ensemble_params['mu'] if 'mu' in self.ensemble_params else 1
        self.m = self.ensemble_params['M_clsfy_ensemble_size']  ## --- k: number of classifiers
        self.min_lambda = self.ensemble_params['min_lambda']  ## --- min_lambda: minimum value of lambda
        self.max_lambda = self.ensemble_params['max_lambda']
        self.margin_threshold = self.ensemble_params[
            'margin_threshold'] if 'margin_threshold' in self.ensemble_params else 0.7  ## --- margin_threshold: threshold for margin

        self.rng = np.random.RandomState(
            self.ensemble_params['random_state']) if 'random_state' in params else np.random.RandomState(1)
        self.random_seed_count = 0

        ## --- data set for model update
        self.label_data_collector = []  ## --- labeled data for model update
        self.label_data_x_collector = []  ## --- labeled data for model update
        self.label_data_y_collector = []  ## --- labeled data for model update
        self.label_data_collector_size = self.main_model_params[
            'label_data_collector_size'] if 'label_data_collector_size' in self.main_model_params else 1200

        self.buffer = []  ## --- buffer (unlabeled data) for model update
        self.buffer_size = self.main_model_params['buffer_size'] if 'buffer_size' in self.main_model_params else 50

        self.set_anomaly_forest_params(self.anomaly_forest_param)

        ## ---- model in memory
        self.anomaly_detector = None
        self.is_anomaly_checked = False  ## --- whether anomaly detector is checked (one new class is included in one chunk)

        self.ensemble_anomaly_detectors = []
        self.ensemble_classification_models = []

        self.y_pred = []
        self.init_label_data_collector()
        return

    def init_label_data_collector(self):
        """ if offline data is all labeled """
        for i in range(len(self.offline_data['X'])):
            self.label_data_collector.append((self.offline_data['X'][i], self.offline_data['y'][i]))
            self.label_data_x_collector.append(self.offline_data['X'][i])
            self.label_data_y_collector.append(self.offline_data['y'][i])
        return

    def set_anomaly_forest_params(self, params):
        self.anomaly_forest_params = params
        self.is_per_class_anomaly_detector = params[
            'per_class_anomaly_detector'] if 'per_class_anomaly_detector' in params else False
        self.per_class_anomaly_train_buffer = {}  ## --- key: class, value: buffer for training anomaly detector
        logger.info("self.is_per_class_anomaly_detector: %s" % self.is_per_class_anomaly_detector)
        return

    def update_unlabeled_buffer(self, x):
        self.buffer.append(x)
        if len(self.buffer) > self.buffer_size:
            self.update_model_by_buffer()
        return

    def select_subset_of_label_data_collector(self):
        keep_hist_ratio = 0.2
        self.label_data_collector = random.sample(self.label_data_collector,
                                                  int(keep_hist_ratio * len(self.label_data_collector)))
        return

    def update_model_by_buffer(self):
        if self.online_data['new_class'] == []:
            return
        else:
            _new_class = self.online_data['new_class'][0]
            for x in self.buffer:
                self.online_update_by_labeled_data(x, _new_class)
        self.buffer.clear()
        self.is_anomaly_checked = True
        return

    def train_offline(self):
        self.X_offline = self.offline_data['X']
        self.y_offline = self.offline_data['y']
        self.train_anomaly_detection_init(self.X_offline, self.y_offline)
        logger.info("offline learning anomaly detection model finished!")

        self.train_classification_init(self.X_offline, self.y_offline)
        logger.info("offline learning classification model finished!")

        self.select_subset_of_label_data_collector()
        return

    def train_anomaly_detection_init(self, X_train, y_train):
        if self.is_per_class_anomaly_detector:
            self._train_anomaly_detector_per_class_init(X_train, y_train)
        else:
            self._train_anomaly_detection_init(X_train)
        return

    def _train_anomaly_detection_init(self, X_train):
        """
        Do not split classes, train one anomaly detector for all classes
        """
        self.ensemble_anomaly_detectors.clear()
        self.anomaly_detector = ISLENCForest(feature_data=X_train, params=self.anomaly_forest_params)
        self.anomaly_detector.build_forest()
        self.anomaly_detector.init_buffer()
        self.ensemble_anomaly_detectors.append(self.anomaly_detector)
        return

    def _train_anomaly_detector_per_class_init(self, X_train, y_train):
        """
        Split classes, train one anomaly detector for each class
        """
        self.ensemble_anomaly_detectors.clear()
        class_list = np.unique(y_train)
        for class_label in class_list:
            class_X = X_train[y_train == class_label]
            class_anomaly_detector = ISLENCForest(feature_data=class_X, params=self.anomaly_forest_params,
                                                  verbose=False)
            class_anomaly_detector.build_forest()
            class_anomaly_detector.init_buffer()
            self.ensemble_anomaly_detectors.append(class_anomaly_detector)
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

    def train_classification_init(self, X_train, y_train):
        """
        """
        self.ensemble_classification_models = []
        if self.classification_ensemble_type == classification_ensemble_type.subset_feature:
            for j in range(self.m):
                selected_feature_ids = self._get_train_subset_feature_ids()
                logger.info("selected_feature_ids: {}".format(selected_feature_ids))
                # train_X = self.X_offline[:,selected_feature_ids].copy()
                train_X = X_train
                train_y = y_train.copy()
                clsfy_model = self._train_single_classification_model(train_X, train_y, selected_feature_ids)
                clsfy_model.set_id(j + 1)  ## --- model_id starts from 1
                self.ensemble_classification_models.append(clsfy_model)
        return self.ensemble_classification_models

    ## ---- classification model train/update/predict
    def _train_single_classification_model(self, train_X, train_y, selected_feature_ids=None):
        ht = HoeffdingTreeClassifier()
        train_X = train_X[:, selected_feature_ids]
        ht.fit(train_X, train_y)
        return ClassificationModel(ht, selected_feature_ids)

    def _select_label_data(self):
        ## select a subset of label data
        if len(self.label_data_collector) > self.label_data_collector_size:
            return np.array(self.label_data_x_collector[-self.label_data_collector_size:]), np.array(
                self.label_data_y_collector[-self.label_data_collector_size:])
        else:
            return np.array(self.label_data_x_collector), np.array(self.label_data_y_collector)

    def update_model_by_label_data(self):
        ## --- select a subset of label data
        X_train, y_train = self._select_label_data()
        ## --- update classification model
        self.train_anomaly_detection_init(X_train, y_train)
        logger.info("chunk update anomaly detection model finished!")

        self.train_classification_init(X_train, y_train)
        logger.info("chunk update classification model finished!")

        self.select_subset_of_label_data_collector()

        return

    def online_update_by_labeled_data(self, X, y):
        if y not in self.class_list:
            logger.info("novel class %s found!" % y)
            self.class_list = np.concatenate([self.class_list, [y]])
        self.label_data_collector.append((X, y))
        self.label_data_x_collector.append(X)
        self.label_data_y_collector.append(y)
        self.update_classification_online(X, y)  ## --- function from lyf_model.py
        self.anomaly_detection_update(X, y)  ## --- function from lyf_model.py
        if len(self.label_data_collector) >= self.label_data_collector_size:
            logger.info("Labeled data size exceeds %s, begin to update model!" % self.label_data_collector_size)
            self.update_model_by_label_data()
        return

    def _compute_lambda(self, y):
        most_common_y = Counter(self.label_data_y_collector).most_common(1)[0][0]
        most_common_y_count = Counter(self.label_data_y_collector).most_common(1)[0][1]
        this_y_count = Counter(self.label_data_y_collector)[y]
        try:
            _lambda = self.min_lambda + math.log10(most_common_y_count / this_y_count) * self.min_lambda
        except:
            logger.error("most_common_y_count: %s, y: %s this_y_count: %s" % (most_common_y_count, y, this_y_count))
            exit(-1)
        return _lambda

    def _online_update_single_classification_model(self, classification_model, x, y, sample_weight):
        """x: 1d array"""
        x = x.reshape(1, -1)
        y = np.array(y).reshape(-1)
        sample_weight = sample_weight.reshape(-1)
        ## --- select features
        x = x[:, classification_model.feature_index_list]
        classification_model.model.partial_fit(x, y, sample_weight=sample_weight)
        return classification_model

    def update_classification_online(self, x, y):
        _lambda = self._compute_lambda(y)
        for j in range(self.m):
            sample_weight = poisson.rvs(_lambda, size=1)
            sample_weight = sample_weight * self.ensemble_classification_models[j].model_id / self.m
            self.ensemble_classification_models[j] = self._online_update_single_classification_model(
                self.ensemble_classification_models[j], x, y, sample_weight)
        return self.ensemble_classification_models

    def _predict_single_classification_model(self, classification_model, x):
        x = x.reshape(1, -1)
        x = x[:, classification_model.feature_index_list]
        return classification_model.model.predict(x)[0]

    def predict_online_instance_with_selftraining(self, x):
        ## majority voting by ensemble
        y_pred_list = []
        for j in range(self.m):
            y_pred = self._predict_single_classification_model(self.ensemble_classification_models[j], x)
            if (y_pred) in self.class_list:
                y_pred_list.append(y_pred)
            # y_pred_list.append(y_pred)
        y_pred_list = np.array(y_pred_list)
        # logger.info("y_pred_list: %s"%y_pred_list)
        if len(y_pred_list) == 0:
            y_pred = SemiWrapperISLENC.UNKNOWN_CLASS
            return y_pred
        y_pred = Counter(y_pred_list).most_common(1)[0][0]
        if len(Counter(y_pred_list).most_common(1)) == 1:
            ensemble_margin = 1 * len(y_pred_list) / self.m
        else:
            ensemble_margin = (Counter(y_pred_list).most_common(1)[0][1] - Counter(y_pred_list).most_common(2)[1][
                1]) / self.m * len(y_pred_list) / self.m

        if y_pred not in self.label_data_y_collector:
            logger.error("y_pred_list: %s" % y_pred_list)
            logger.error("y_pred: %s not in label_date_y_collector.keys()" % y_pred)
            exit(-1)
        if ensemble_margin > self.margin_threshold:
            ## self training
            self.update_classification_online(x, y_pred)
        return y_pred

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
