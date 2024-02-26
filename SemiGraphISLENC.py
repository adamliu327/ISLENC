import numpy as np
import random

import sys
sys.path.append("../")

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from ISLENCForest import ISLENCForest

__all__ = ["SemiGraphISLENC"]

class SemiGraphISLENC:
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
        self.anomaly_label = params['anomaly_label'] if 'anomaly_label' in params else SemiGraphISLENC.NEW_CLASS

        self.main_model_params = params['main_model_params']
        self.anomaly_forest_param = params['iforest_params']
        
        self.random_seed_count = 0

        ## --- data set for model update
        self.label_data_collector = []  ## --- labeled data for model update
        self.label_data_x_collector = []  ## --- labeled data for model update
        self.label_data_y_collector = []  ## --- labeled data for model update
        self.label_data_collector_size = self.main_model_params['label_data_collector_size'] if 'label_data_collector_size' in self.main_model_params else 1200

        self.buffer = []  ## --- buffer (unlabeled data) for model update
        self.buffer_size = self.main_model_params['buffer_size'] if 'buffer_size' in self.main_model_params else 50
     
        self.set_anomaly_forest_params(self.anomaly_forest_param)

        ## ---- model in memory
        self.anomaly_detector = None
        self.is_anomaly_checked = False ## --- whether anomaly detector is checked (one new class is included in one chunk)

        self.ensemble_anomaly_detectors = []
        return
    
    def init_label_data_collector(self):
        """ if offline data is all labeled """
        for i in range(len(self.offline_data['X'])):
            self.label_data_collector.append((self.offline_data['X'][i],self.offline_data['y'][i]))
            self.label_data_x_collector.append(self.offline_data['X'][i])
            self.label_data_y_collector.append(self.offline_data['y'][i])
        return
    
    def set_anomaly_forest_params(self,params):
        self.anomaly_forest_params = params
        self.is_per_class_anomaly_detector = params['per_class_anomaly_detector'] if 'per_class_anomaly_detector' in params else False
        self.per_class_anomaly_train_buffer = {}  ## --- key: class, value: buffer for training anomaly detector
        return
    
    def train_offline(self):
        self.X_offline = self.offline_data['X']
        self.y_offline = self.offline_data['y']
        self.train_anomaly_detection_init(self.X_offline,self.y_offline)
        logger.info("offline learning anomaly detection model finished!")
        self.select_subset_of_label_data_collector()
        return
    
    def train_anomaly_detection_init(self,X_train, y_train):
        if self.is_per_class_anomaly_detector:
            self._train_anomaly_detector_per_class_init(X_train, y_train)
        else:
            self._train_anomaly_detection_init(X_train)
        return
    
    def _train_anomaly_detection_init(self,X_train):
        """
        Do not split classes, train one anomaly detector for all classes
        """
        self.ensemble_anomaly_detectors.clear()
        self.anomaly_detector = ISLENCForest(feature_data=X_train,params=self.anomaly_forest_params)
        self.anomaly_detector.build_forest()
        self.anomaly_detector.init_buffer()
        self.ensemble_anomaly_detectors.append(self.anomaly_detector)
        return

    def _train_anomaly_detector_per_class_init(self,X_train, y_train):
        """
        Split classes, train one anomaly detector for each class
        """
        self.ensemble_anomaly_detectors.clear()
        class_list = np.unique(y_train)
        for class_label in class_list:
            class_X = X_train[y_train==class_label]
            class_anomaly_detector = ISLENCForest(feature_data=class_X,params=self.anomaly_forest_params,verbose = False)
            class_anomaly_detector.build_forest()
            class_anomaly_detector.init_buffer()
            self.ensemble_anomaly_detectors.append(class_anomaly_detector)
        return
    
    def select_subset_of_label_data_collector(self):
        keep_hist_ratio = 0.2
        self.label_data_collector = random.sample(self.label_data_collector,int(keep_hist_ratio*len(self.label_data_collector)))
        return
    
    def online_update_by_labeled_data(self, X, y):
        if y not in self.class_list:
            logger.info("novel class %s found!" %y)
            self.class_list = np.concatenate([self.class_list,[y]])
        self.label_data_collector.append((X,y))
        self.label_data_x_collector.append(X)
        self.label_data_y_collector.append(y)
        self.anomaly_detection_update(X,y)  ## --- function from lyf_model.py
        if len(self.label_data_collector) >= self.label_data_collector_size:
            logger.info("Labeled data size exceeds %s, begin to update model!"%self.label_data_collector_size)
            self.update_model_by_label_data()
        return
    
    def _select_label_data(self):
        ## select a subset of label data
        if len(self.label_data_collector) > self.label_data_collector_size:
            return np.array(self.label_data_x_collector[-self.label_data_collector_size:]),np.array(self.label_data_y_collector[-self.label_data_collector_size:])
        else:
            return np.array(self.label_data_x_collector),np.array(self.label_data_y_collector)

    def update_model_by_label_data(self):
        ## --- select a subset of label data
        X_train,y_train = self._select_label_data()
        ## --- update classification model
        self.train_anomaly_detection_init(X_train,y_train)
        logger.info("chunk update anomaly detection model finished!")
        
        self.select_subset_of_label_data_collector()
        return
    
    ## ---- anomaly detection model train/update/predict 
    def anomaly_detection_update(self,x,y=None):
        if y not in self.class_list:
            logger.info("novel class %s found!" %y)
            self.class_list = np.concatenate([self.class_list,[y]])
        if self.is_per_class_anomaly_detector:
            _class_id = np.argwhere(self.class_list==y)[0][0]
            if len(self.ensemble_anomaly_detectors) > _class_id:
                self.ensemble_anomaly_detectors[_class_id].update(x)
            else:
                logger.debug("class %s anomaly detector not found"%y)
                if y in self.per_class_anomaly_train_buffer.keys():
                    self.per_class_anomaly_train_buffer[y].append(x)
                    logger.debug("class %s anomaly detector buffer size: %s"%(y,len(self.per_class_anomaly_train_buffer[y])))
                    if len(self.per_class_anomaly_train_buffer[y]) >= self.anomaly_forest_params['min_tree_size']:
                        logger.info("Begin to train class %s anomaly detector"%y )
                        class_X = np.array(self.per_class_anomaly_train_buffer[y])
                        class_anomaly_detector = ISLENCForest(feature_data=class_X,params=self.anomaly_forest_params)
                        class_anomaly_detector.build_forest()
                        class_anomaly_detector.init_buffer()
                        self.ensemble_anomaly_detectors.append(class_anomaly_detector)
                        self.per_class_anomaly_train_buffer.pop(y)
                else:
                    self.per_class_anomaly_train_buffer[y] = [x]
        else:
            self.ensemble_anomaly_detectors[0].update(x)
        return
    
    def anomaly_detection_predict(self,x):
        if self.is_per_class_anomaly_detector:
            _anomaly_flag_list = []
            _anomaly_score_list = []
            for class_anomaly_detector in self.ensemble_anomaly_detectors:
                _anomaly_flag,_anomaly_score = class_anomaly_detector.predict_anomaly_iForest(x)
                _anomaly_flag_list.append(_anomaly_flag)
                _anomaly_score_list.append(_anomaly_score)
            _anomaly_flag_list = np.array(_anomaly_flag_list)
            _anomaly_score_list = np.array(_anomaly_score_list)
            _anomaly_flag = np.all(_anomaly_flag_list)
            return _anomaly_flag
        else:
            _anomaly_flag,_anomaly_score = self.ensemble_anomaly_detectors[0].predict_anomaly_iForest(x)
            return _anomaly_flag
    
    def update_unlabeled_buffer(self,x):
        self.buffer.append(x)
        if len(self.buffer) > self.buffer_size:
            self.update_model_by_buffer()
        return

    def update_model_by_buffer(self):
        if self.online_data['new_class'] == []:
            return
        else:
            _new_class = self.online_data['new_class'][0]
            for x in self.buffer:
                self.online_update_by_labeled_data(x,_new_class)
        self.buffer.clear()
        self.is_anomaly_checked = True
        return

    