from .base_options import BaseOptions
import argparse
import os

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=600, help='how many test images to run')
        self.parser.add_argument('--video_folder', type=str, default='bear', help='folder name ..')
        self.parser.add_argument('--ab_bin', type=int, default=64, help='ab_bin')
        self.parser.add_argument('--l_bin', type=int, default=8, help='l_bin')
        self.isTrain = False



# import yaml

# par = os.getcwd()
# file_path = os.path.join(par, 'HairStepInfer', 'lib', 'hair_config.yml')
# def load_config_from_yaml(file_path):
#     with open(file_path, 'r') as file:
#         config = yaml.safe_load(file)
#     return config
# # hair_options = BaseOptions().parse()
# def dict_to_namespace(config_dict):
#     return argparse.Namespace(**config_dict)
# config_dict = load_config_from_yaml(file_path)
# # Convert and print the Namespace object
# namespace_config = dict_to_namespace(config_dict)