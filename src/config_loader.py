import os
import yaml


def load_config(config_path=None):
    if config_path is None:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..')
        )
        config_path = os.path.join(project_root, 'config', 'config.yaml')

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
