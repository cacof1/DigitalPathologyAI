import matplotlib.pyplot as plt
from Dataloader.Dataloader import LoadFileParameter
import pandas as pd

def AppendSarcomaLabel(ids, SVS_Folder, Patch_Folder):

    # Add a code that makes sure we have the sarcoma label for each
    wsi_files = LoadFileParameter(ids, SVS_Folder, Patch_Folder)[0]

    sarcoma_diagnoses_path = '../SarcomaClassification/data/'
    df_sarcoma_diagnoses = pd.read_csv(sarcoma_diagnoses_path + "sarcoma_diagnoses.csv")
    df_sarcoma_mapping = pd.read_csv(sarcoma_diagnoses_path + "mapping.csv")

    for wsi_file in wsi_files:
        cur_coords_file = pd.read_csv(Patch_Folder + wsi_file + '.csv', index_col=0)
        if not "sarcoma_label" in cur_coords_file:

            #print('Appending sarcoma label to {}.csv...'.format(wsi_file))

            cur_wsi_mask = int(wsi_file) == df_sarcoma_diagnoses['id']

            diag = df_sarcoma_diagnoses['diagnosis'][cur_wsi_mask].values[0]
            grade = df_sarcoma_diagnoses['grade'][cur_wsi_mask].values[0]
            current_diagnosis = diag + '_' + grade if isinstance(grade, str) else diag

            diag_mask = current_diagnosis == df_sarcoma_mapping['sarcoma_type']
            current_sarcoma_label = df_sarcoma_mapping['label'][diag_mask].values[0]

            csv_path = Patch_Folder + wsi_file + '.csv'

            cur_coords_file["sarcoma_label"]  =  pd.Series([current_sarcoma_label]*len(cur_coords_file))
            cur_coords_file.to_csv(csv_path)