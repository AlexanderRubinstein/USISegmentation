import sys
import os

import numpy as np
import pandas as pd

from configs.parsing import cmd_args_parsing, args_parsing
from patient_data import Patient

DATASET_TABLE_PATH = './dataset.csv'

def make_csv_table(dataset, csv_file_path):
    pd.DataFrame(dataset, columns=['image', 'mask', 'frame']).to_csv(csv_file_path, index=False)

def main(argv):
    params = args_parsing(cmd_args_parsing(argv))
    root, raw_data_path, preprocessed_data_path = params['root'], params['raw_data_path'], params['preprocessed_data_path']
    
    patients_paths = [os.path.join(raw_data_path, patient_name) for patient_name in sorted(os.listdir(raw_data_path))]
    
    print('preprocessing data for {} patients'.format(len(patients_paths)))
    print()

    dataset = []
    for patient_path in patients_paths:
        patient = Patient(patient_path)
        patient_name = patient.get_patient_name()
    
        print('{} data reading ...'.format(patient_name)) 
        patient_data = patient.get_patient_data()
    
        print('{} data preprocessing ...'.format(patient_name))
        patient.data_preprocessing(patient_data)
    
        print('{} preprocessed data saving ...'.format(patient_name))
        patient.save_tif_images(patient_data, preprocessed_data_path)
    
        print()
        
        dataset.append(patient.make_dataset_table(patient_data, preprocessed_data_path))
    
    print('dataset csv table creating...')
    make_csv_table(np.vstack(dataset), os.path.join(root, DATASET_TABLE_PATH))

if __name__ == "__main__":
    main(sys.argv[1:])
