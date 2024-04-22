import os
import yaml

_WORK_PATH = os.environ["BMOCA_HOME"]


def load_config(config_path=f'{_WORK_PATH}/config/model_config.yaml'):
    configs = dict(os.environ)
    with open(config_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    configs.update(yaml_data)
    return configs