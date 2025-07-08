import yaml
import os


class Parameters():
    def __init__(self, yaml_file='parameters.yaml'):
        # print(__file__)
        global_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_file_global = global_dir + '/' + yaml_file
        with open(yaml_file_global) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

PARAMETERS = Parameters()