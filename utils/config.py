import yaml

def load_config(file):
    with open(file, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    return cfg
