import argparse
from pathlib import Path
from druglikeness.trainer import TrainRunner

usage = 'python train.py -c <config_path> [-g <gpu_index> -o]'

parser = argparse.ArgumentParser(usage=usage)
parser.add_argument('-c', '--config_path', help='Path of config yaml file', type=Path, required=True)
parser.add_argument('-g', '--gpu', help='Index of GPU to use.')
parser.add_argument('-o', '--override', help='Override existing output files', action='store_true')
args = parser.parse_args()

runner = TrainRunner(args)
runner.run_cv_train()
