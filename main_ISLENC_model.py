import numpy as np
import pandas as pd
import json

from ISLENCModel import ISLENCModel
from computational_result_presentation.get_data import get_data

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

total_config_json_fname = "./experiment_config.json"
with open(total_config_json_fname) as f:
    total_config_param = json.load(f)
params = total_config_param['supervised_parameter_setting']['params']
param_setting_list = total_config_param['supervised_parameter_setting']['param_setting_list']
print(params, param_setting_list)

total_data_json_fname = "./computational_result_presentation/data_config.json"
with open(total_data_json_fname) as f:
    total_data_config = json.load(f)
supervised_data_dir = total_data_config['main_dataset_dir']['supervised_data_dir']
supervised_output_dir = total_data_config['main_dataset_dir']['supervised_output_dir']
default_params = total_config_param['supervised_parameter_setting']['default_setting']
print(default_params)

case_list = total_config_param['case_list']
print(case_list)


def run_supervised_model(case_name, ensemble_params, forest_params, output_fname):
    logger.info(case_name)
    case_config_param = total_config_param['case_config'][case_name]
    logger.info("case_config:", case_config_param)
    data_name = case_config_param['data_name']
    data_config_param = total_data_config[data_name]
    data_dir = supervised_data_dir + data_config_param['supervised_data_inputdir']

    general_params = default_params['default_params'].copy()
    init_chunk_size = general_params['init_chunk_size']  ## paper: init_number = 3*S
    S_chunk_size = general_params['S_chunk_size']
    train_size = init_chunk_size * S_chunk_size
    X_train, y_train, X_test, y_test = get_data(data_name, data_dir=data_dir, \
                                                offline_size=train_size, norm=False)
    logger.info("X_train: %s", X_train.shape)
    logger.info("X_test: %s", X_test.shape)

    params = ensemble_params.copy()
    params['X_offline'] = X_train
    params['y_offline'] = y_train
    params['verbose'] = True
    model = ISLENCModel(X_train, y_train, params)

    model.set_anomaly_forest_params(forest_params)

    ## --- offline ---
    model.train_offline()
    logger.info("offline learning classification model finished!")

    _count = 0
    _anomaly_count = 0
    for x, y_label in zip(X_test, y_test):
        x_j, y_label_j = x, y_label
        model.push_U_unlabeled_buf(x_j, y_label_j)
        model.push_online_instance_for_validation(x_j, y_label_j)
        _anomaly_flag = model.anomaly_detection_predict(x_j)
        if _anomaly_flag:
            _anomaly_count += 1
            _predicted_label = model.anomaly_label
            # break
        else:
            _predicted_label = model.predict_online_instance(x_j)
        model.push_predict_label_for_online_instance(_predicted_label)
        if model.get_U_unlabeled_buf_size() > model.T_l:
            (x_k, y_k) = model.pop_U_unlabeled_buf()
            model.push_L_labeled_buf(x_k, y_k)
            model.update_classification_online(x_k, y_k)
            model.anomaly_detection_update(x_k, y_k)
        _count += 1

    logger.info("online learning finished!")
    ## save true_pred_online.csv
    true_online = np.array(model.y_online)
    pred_online = np.array(model.y_online_pred)
    merged = np.concatenate([true_online.reshape(-1, 1), pred_online.reshape(-1, 1)], axis=1)
    pd.DataFrame(merged).to_csv(output_fname, index=False)
    logger.info("save true_pred_online.csv to %s", output_fname)

    logger.info("experiment end normally!")


def main():
    case_name = 'HTRU'
    ensemble_params = default_params['ensemble_default_params'].copy()
    forest_params = default_params['Forest_default_params'].copy()

    output_fname = case_name + "_true_pred_online.csv"
    print(output_fname)
    run_supervised_model(case_name, ensemble_params, forest_params, output_fname)
    return 0


if __name__ == '__main__':
    main()
