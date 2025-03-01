import argparse
from druglikeness.predictor import TestRunner

usage = 'python predict.py -i <input_path> -m <model_dir> -o <output_path> [-g <gpu_index> -p]'
parser = argparse.ArgumentParser(usage=usage)

parser.add_argument('-i', '--input_path', help='Path of input file. Should be smi or csv file.', required=True)
parser.add_argument('-m', '--model_dir', help='Path of directory containing saved models and the config file', required=True)
parser.add_argument('-o', '--output_path', help='Path of output file. Is a csv file.', required=True)
parser.add_argument('-g', '--gpu', help='Index of GPU to use.', default='0')
parser.add_argument('-p', '--is_pair_data', help='Use this if the input file contains pairs of molecules as data.', action='store_true')
args = parser.parse_args()

runner = TestRunner(args)
runner.predict_ensemble()
