#!/public/home/caiyi/software/miniconda3/bin/python
import os
import argparse
from data_preprocess.mol_collection import MolCollection, MolPairCollection
from data_preprocess.data_preprocess import DataProcessor, PairDataProcessor

usage = 'python preprocess_data.py <input_path1> ... -o <output_path> [-l <default_label> ' \
        '-sc <smiles_col> -lc <label_col> -np -ra -p -ah -os] \n\n' \
        '-ah: keep_first, keep_all, mean, remove_positive, remove_negative, remove_all'
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument('inputs', nargs='+', help='Positional arguments')
parser.add_argument('-np', '--no_preprocess', action='store_true')
parser.add_argument('-ra', '--remove_ambiguity', action='store_true')
parser.add_argument('-sc', '--smiles_col', default='smiles')
parser.add_argument('-lc', '--label_col', default='label')
parser.add_argument('-os', '--output_scaffold', action='store_true')
parser.add_argument('-l', '--default_label', type=int)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('-p', '--is_pair', action='store_true')
parser.add_argument('-ah', '--ambiguity_handling')
args = parser.parse_args()


input_list = args.inputs

if not args.is_pair:
    store_mol = False if args.no_preprocess else True
    dataset = MolCollection()
    for input_file in input_list:
        input_name = os.path.basename(input_file).split('.')[0]
        dataset = dataset + MolCollection(data_source=input_file, 
                                    name=input_name, 
                                    smiles_col=args.smiles_col, 
                                    label_col=args.label_col,
                                    label=args.default_label,
                                    store_mol=store_mol)
    processor = DataProcessor()
    if not args.no_preprocess:
        processor.preprocess(dataset=dataset)

    if args.remove_ambiguity:
        processor.remove_ambiguity(dataset=dataset, removed_class=1, threshold=0.8)

    # processor.deduplicate(dataset=dataset, ambiguity_handling='remove_positive')
    processor.deduplicate(dataset=dataset, ambiguity_handling=args.ambiguity_handling)

    print("Output file:", args.output)
    # dataset.to_csv(args.output, include_scaffold=args.output_scaffold)
    dataset.to_smi(args.output)
else:
    dataset = MolPairCollection()
    for input_file in input_list:
        input_name = os.path.basename(input_file).split('.')[0]
        dataset = dataset + MolPairCollection(data_source=input_file, 
                                              name=input_name,
                                              smiles_col_0=f'{args.smiles_col}_i', 
                                              smiles_col_1=f'{args.smiles_col}_j', 
                                              label_col=args.label_col,
                                              store_mol=True)
    processor = PairDataProcessor()
    if not args.no_preprocess:
        processor.preprocess(dataset=dataset)

    print("Output file:", args.output)
    dataset.to_csv(args.output)
    out_prefix = args.output[:-4]
    dataset.to_separate_csvs(file_path_prefix=out_prefix)
