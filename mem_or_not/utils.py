import yaml


def load_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
