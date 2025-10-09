import os
import logging
import datetime
from config.configurator import configs

def get_local_time():
    return datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

class Logger(object):
    def __init__(self, log_configs=True):
        model_name = configs['model']['name']
        dataset_name = configs['data']['name']
        log_dir_path = './results/log/{}'.format(model_name)

        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.INFO)
        # Dynamically construct sub-directory from tuning params (read from environment variables)
        tuning_param = os.environ.get('RLMREC_TUNING_PARAM')
        tuning_value = os.environ.get('RLMREC_TUNING_VALUE')
        # Debug print to confirm the env variables are read correctly
        if tuning_param is not None and tuning_value is not None:
            if tuning_param == "ablation":
                subdir = f"{tuning_param}/{tuning_value}"
            else:
                subdir = f"{tuning_param}/{dataset_name}"
        else:
            subdir = f""
        full_dir = f"{log_dir_path}/{subdir}"
        print(f"[logger] log dir resolved to: {full_dir}")
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
        log_file = logging.FileHandler('{}/{}_{}.log'.format(full_dir, dataset_name, get_local_time()), 'a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        log_file.setFormatter(formatter)
        self.logger.addHandler(log_file)
        if log_configs:
            tmp_configs = {}
            tmp_configs['optimizer'] = configs['optimizer']
            tmp_configs['train'] = configs['train']
            tmp_configs['test'] = configs['test']
            tmp_configs['data'] = configs['data']
            tmp_configs['model'] = configs['model']
            self.log(tmp_configs)

    def log(self, message, save_to_log=True, print_to_console=True):
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_loss(self, epoch_idx, loss_log_dict, save_to_log=True, print_to_console=True):
        epoch = configs['train']['epoch']
        message = '[Epoch {:3d} / {:3d}] '.format(epoch_idx, epoch)
        for loss_name in loss_log_dict:
            message += '{}: {:.4f} '.format(loss_name, loss_log_dict[loss_name])
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_eval(self, eval_result, k, data_type, save_to_log=True, print_to_console=True, epoch_idx=None):
        if epoch_idx is not None:
            message = 'Epoch {:3d} {} '.format(epoch_idx, data_type)
        else:
            message = ''

        for metric in eval_result:
            message += '['
            for i in range(len(k)):
                message += '{}@{}: {:.4f} '.format(metric, k[i], eval_result[metric][i])
            message += '] '
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)