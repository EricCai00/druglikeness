# Comprehensive drug-likeness prediction using pre-trained models and multi-task learning

We here introduce a comprehensive framework for predicting drug-likeness, a crucial factor in drug discovery, by leveraging pre-trained molecular representation models combined with multi-task learning. Traditional drug-likeness prediction methods often rely on handcrafted features or predefined molecular fingerprints, which can limit their ability to generalize across diverse chemical spaces. The proposed framework addresses these limitations by employing pre-trained models to capture complex molecular features, enabling a broader and more accurate assessment of drug-likeness.

The framework consists of two parts: **SpecDL** models, which is tailored for specialized drug-likeness tasks, and **GeneralDL**, which synthesizes predictions across multiple datasets to offer a more generalized evaluation. Through evaluation on the test sets, the framework demonstrated superior performance, with all four SpecDL models achieving an ROC-AUC higher than 0.7 on their respective test sets, and GeneralDL achieving a 0.781 ROC-AUC across six internal and external test sets.

## Requirements
python\==3.7.16

numpy\==1.21.5

pandas\==1.3.5

rdkit\==2023.3.2

pytorch\==1.12.1

transformers\==4.24.0

datasets\==2.6.1

scikit-learn\==0.23.2

yaml\==0.2.5

## Preparation

Download model weights from https://huggingface.co/ericsai/generaldl, and place them in the `weights` folder.

## Predict using GeneralDL or SpecDL

Use preprocess_data.py to pre-process your data. The workflow includes:
- removing isotopic labels
- converting salts to their corresponding acids or bases
- eliminating inorganic fragments
- removing mixtures
- excluding molecules with molecular weights exceeding 1000 Da or fewer than six heavy atoms
- neutralizing molecular charges
- standardizing molecules to their canonical tautomeric forms
- identifying and removing duplicate molecular entries

```python
python preprocess_data.py example/example_ori.smi -o example/example.smi
```

To predict the SMILES file using GeneralDL:

```python
python predict.py -i example/example.smi -m weights/generaldl -o example/example_generaldl.csv
```


You can also using the models to predict a CSV file with `smiles` and `label` as columns:
```python
python predict.py -i data/test_set/approved+ftt_test.csv -m weights/generaldl -o example/approved+ftt_test_generaldl.csv
```


Replace `-m weights/generaldl` with `-m weights/specdl-ftt` or others to use SpecDL models.

## Train custom models

To train models using custom data, first preprocess your datasets:
```python
python preprocess_data.py <your_csv_file> -o <preprocessed_csv_file>
```


Then provide a YAML config file to `train.py` to train the model:
```python
python train.py -c <your_config_file>
```
Refer to the `default_config` folder to understand the required arguments for the config file.
