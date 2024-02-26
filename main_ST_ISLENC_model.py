import numpy as np
import pandas as pd
import json
import os
import pickle

import time
from SemiWrapperISLENC import SemiWrapperISLENC

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

total_config_json_fname = "./semi_experiment_config.json"
with open(total_config_json_fname) as f:
    total_config_param = json.load(f)

default_model_param = total_config_param['wrapper_semi_parameter_setting']['default_setting']
default_main_model_param = default_model_param['main_model_params']
default_model_iforest_param = default_model_param['iforest_params']
default_model_classify_param = default_model_param['ensemble_params']

total_data_json_fname = "./computational_result_presentation/data_config.json"
with open(total_data_json_fname) as f:
    total_data_config = json.load(f)
semi_supervised_data_dir = total_data_config['main_dataset_dir']['semi_data_dir']
semi_wrapper_output_dir = total_data_config['main_dataset_dir']['Semi_Wrapper_supervised_output_dir']


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


def run_ST_semi_model(total_data_dict, model_param, verbose=False):
    y_true = []
    y_pred = []

    for chunk_id in total_data_dict.keys():
        if chunk_id == 0:
            continue
        print("chunk_id", chunk_id)

        data_set = total_data_dict[chunk_id]
        X, y, new_class, label_state = data_set['X'], data_set['y'], data_set['new_class'], data_set['label_state']

        _last_data_set = total_data_dict[chunk_id - 1]
        _offline_data = _last_data_set

        _online_data = data_set
        model_semi_ST = SemiWrapperISLENC(_offline_data, _online_data, model_param)
        model_semi_ST.train_offline()

        n = X.shape[0]
        for i in range(n):
            X_i, y_i = X[i], y[i]
            is_labeled = label_state[i]
            if verbose: logger.info("i: {} is_labeled: {}".format(i, is_labeled))
            if is_labeled:
                model_semi_ST.y_pred.append(y_i)
                model_semi_ST.online_update_by_labeled_data(X_i, y_i)
            else:
                _anomaly_flag = model_semi_ST.anomaly_detection_predict(X_i)
                if _anomaly_flag:
                    _predicted_label = model_semi_ST.anomaly_label
                    model_semi_ST.update_unlabeled_buffer(X_i)
                else:
                    _predicted_label = model_semi_ST.predict_online_instance_with_selftraining(X_i)
                model_semi_ST.y_pred.append(_predicted_label)

                y_true.append(y_i)
                y_pred.append(_predicted_label)
    return y_true, y_pred


def run_and_save_ST_model(total_data_dict, model_param, output_fname, output_time_fname):
    start = time.time()
    y_true, y_pred = run_ST_semi_model(total_data_dict, model_param)
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
    case_config_param = total_config_param['semi_wrapper_config'][case_name]
    data_name = case_config_param['data_name']
    input_dir = semi_supervised_data_dir + data_name + '/'
    total_data_dict = get_date_dict(input_dir)

    ensemble_params = default_model_classify_param.copy()
    forest_params = default_model_iforest_param.copy()
    model_param = {'main_model_params': default_main_model_param, 'ensemble_params': ensemble_params,
                   'iforest_params': forest_params}
    output_fname = case_name + "_semi_ST_result.csv"
    output_time_fname = case_name + "_semi_ST_result"
    run_and_save_ST_model(total_data_dict, model_param, output_fname, output_time_fname)
    return 0


if __name__ == '__main__':
    main()
