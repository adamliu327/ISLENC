import numpy as np
import pandas as pd
import json
import os
import pickle
from LGC import SemiLGCGraph
from SemiGraphISLENC import SemiGraphISLENC

import time
import sklearn.preprocessing

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

total_config_json_fname = "./semi_experiment_config.json"
with open(total_config_json_fname) as f:
    total_config_param = json.load(f)

default_model_param = total_config_param['graph_semi_parameter_setting']['default_setting']
default_main_model_param = default_model_param['main_model_params']

case_list = total_config_param['graph_semi_parameter_setting']['case_list']
logger.info("case_list:{}".format(case_list))

total_data_json_fname = "./computational_result_presentation/data_config.json"
with open(total_data_json_fname) as f:
    total_data_config = json.load(f)
semi_supervised_data_dir = total_data_config['main_dataset_dir']['semi_data_dir']
semi_graph_output_dir = total_data_config['main_dataset_dir']['Semi_Graph_supervised_output_dir']


def get_date_dict(input_dir):
    data_list = os.listdir(input_dir)
    data_list = [data for data in data_list if data.endswith(".data")]

    total_data_dict = {}
    for index in range(len(data_list)):
        data = str(index) + ".data"
        # for data in data_list:
        with open(input_dir + data, "rb") as f:
            dataset = pickle.load(f)

        dataset['labeled_X'] = dataset['X'][dataset['label_state']]
        dataset['labeled_y'] = dataset['y'][dataset['label_state']]
        total_data_dict[index] = dataset

    return total_data_dict


def get_sample_offline_data(offline_data, sample_size=10):
    _dict = {}
    X = offline_data['X']
    y = offline_data['y']
    X_y = np.column_stack((X, y))
    offline_classes = np.unique(y)

    ## select 10 samples from each class
    selected_samples = []
    for c in offline_classes:
        X_c = X[X_y[:, -1] == c]
        select_num = min(sample_size, X_c.shape[0])
        selected_samples.append(X_c[:select_num, :])
        _dict[c] = X[:select_num, :]
    return _dict


def get_sample_labeled_data(data_dict, sample_size=10):
    classes = len(data_dict)
    per_class_sample = int(sample_size / classes)
    sample_dict = {}
    for c in data_dict:
        ## ramdomly select 10 samples from each class
        data_c = data_dict[c]
        np.random.shuffle(data_c)
        sample_dict[c] = data_c[:per_class_sample, :]
    return sample_dict


def init_graph_with_sample_label_data(graph_model, sample_data_dict):
    for c in sample_data_dict.keys():
        X_sample = sample_data_dict[c]
        for i in range(X_sample.shape[0]):
            X_i, y_i = X_sample[i], c
            graph_model.grow_graph_labeled(X_i, y_i)
    return graph_model


def run_Graph_model(total_data_dict, model_param):
    verbose = False
    graph_param = model_param['graph_params']

    y_true = []
    y_pred = []

    labeled_data_dict = {}

    _trunk_update_count = 0
    trunk_update_threshold = 512

    offline_X = total_data_dict[0]['X']
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(offline_X)
    for chunk_id in total_data_dict.keys():
        if chunk_id == 0:
            continue
        print("--------- begin predict chunk_id", chunk_id, " --------------")

        _last_data_set = total_data_dict[chunk_id - 1]
        _offline_data = _last_data_set

        data_set = total_data_dict[chunk_id]
        X, y, new_class, label_state = data_set['X'], data_set['y'], data_set['new_class'], data_set['label_state']
        X = scaler.transform(X)
        _online_data = data_set

        main_model_semi_graph = SemiGraphISLENC(_offline_data, _online_data, model_param)
        main_model_semi_graph.train_offline()

        graph_model = SemiLGCGraph(graph_param)
        print("graph_model's gamma is_", graph_model.gamma)
        print(graph_model.ratio)
        offline_data = total_data_dict[0]
        labeled_data_dict = get_sample_offline_data(offline_data, sample_size=50)
        graph_model = init_graph_with_sample_label_data(graph_model, labeled_data_dict)
        print("offline graph model training finished")

        n = X.shape[0]
        for i in range(n):
            X_i, y_i = X[i], y[i]
            is_labeled = label_state[i]
            if verbose: logger.info("i: {} is_labeled: {}".format(i, is_labeled))
            if is_labeled:
                main_model_semi_graph.online_update_by_labeled_data(X_i, y_i)
                graph_model.grow_graph_labeled(X_i, y_i)
                if y_i in labeled_data_dict.keys():
                    labeled_data_dict[y_i] = np.concatenate([labeled_data_dict[y_i], np.array([X_i])], axis=0)
                else:
                    labeled_data_dict[y_i] = np.array([X_i])
            else:
                _trunk_update_count += 1
                _anomaly_flag = main_model_semi_graph.anomaly_detection_predict(X_i)
                if verbose: logger.info("_anomaly_flag: {}".format(_anomaly_flag))
                if _anomaly_flag:
                    _predicted_label = main_model_semi_graph.anomaly_label
                    main_model_semi_graph.update_unlabeled_buffer(X_i)
                else:
                    _predicted_label, _ = graph_model.predict(X_i, y_i)

                y_true.append(y_i)
                y_pred.append(_predicted_label)

                if _trunk_update_count > trunk_update_threshold:
                    print("---------- update trunk model ----------, i: {}".format(i))
                    graph_model = SemiLGCGraph(graph_param)
                    offline_data = total_data_dict[0]
                    labeled_data_dict = get_sample_labeled_data(labeled_data_dict, sample_size=50)
                    graph_model = init_graph_with_sample_label_data(graph_model, labeled_data_dict)
                    _trunk_update_count = 0
    return y_true, y_pred


def run_and_save_Graph_model(total_data_dict, model_param, output_fname, output_time_fname, case_name):
    start = time.time()
    y_true, y_pred = run_Graph_model(total_data_dict, model_param)
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        print(y_true.shape)
        print(y_pred.shape)
    except:
        return y_true, y_pred
    pd.DataFrame({0: y_true, 1: y_pred})
    end = time.time()
    total_seconds = end - start
    pd.DataFrame({0: y_true, 1: y_pred}).to_csv(output_fname, index=False, header=False)
    ## output the total time to txt file
    with open(output_time_fname + "run_time.txt", "a") as f:
        now = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        f.write(now + "," + output_fname + "_pred," + str(total_seconds) + '\n')
    print("write to csv file and time file", output_fname)


def main():
    case_name = 'HTRU'
    case_config_param = total_config_param['semi_graph_config'][case_name]
    data_name = case_config_param['data_name']
    input_dir = semi_supervised_data_dir + data_name + '/'
    total_data_dict = get_date_dict(input_dir)
    graph_params = default_model_param['graph_params'].copy()
    forest_params = default_model_param['iforest_params'].copy()
    output_fname = case_name + "_semi_graph_result.csv"
    output_time_fname = case_name + "_semi_graph_result"
    model_param = {'main_model_params': default_main_model_param, 'graph_params': graph_params,
                   'iforest_params': forest_params}
    run_and_save_Graph_model(total_data_dict, model_param, output_fname, output_time_fname, case_name)

    return 0


if __name__ == '__main__':
    main()
