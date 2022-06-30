"""
Set of tools to manage .npy creation to save patch data.
"""
import numpy as np
import pandas as pd
import os
# import xlsxwriter

def basemodel_to_str(config):
    """Converts config['BASEMODEL'] dict to a string, using valid_keys only."""

    bs = ''
    d = config['BASEMODEL']
    valid_keys = ['Patch_Size', 'Vis']

    for nk, k in enumerate(valid_keys):
        bs += k + '_' + str(d[k]) + ('_' if k != valid_keys[-1] else '')

    return bs

def adjust_df_to_excel_columns(writer, df, sheet):
    # TODO: work in progress. Does not work well with xlsxwriter yet.

    # ws = writer.sheets[sheet]
    # format1 = xlsxwriter.workbook.add_format({'bold': True})
    # format1.set_align('left')

    for column in df:
        column_length = max(df[column].astype(str).map(len).max(), len(column))
        col_idx = df.columns.get_loc(column)
        # print(column, col_idx+1, column_length)  # debugging purposes
        writer.sheets[sheet].set_column(col_idx + 1, col_idx + 1, column_length)  # +1 to account for index

    # dims = {}
    # for row in ws.rows:
    #     for cell in row:
    #         cell.alignment = Alignment(horizontal="left")
    #         if cell.value:
    #             dims[cell.column_letter] = max((dims.get(cell.column_letter, 0), len(str(cell.value))))
    # for col, value in dims.items():
    #     ws.column_dimensions[col].width = value


def decode_npy(patch_npy_export_filename):
    #  Creates a multi-sheet excel file from numpy file patch_npy_export_filename, for visualisation and debugging.

    datasets = list(np.load(patch_npy_export_filename, allow_pickle=True))

    folder = str.split(patch_npy_export_filename, 'patches')
    filename = str.split(folder[1], '.npy')[0][1:]
    excel_file_folder = os.path.join(folder[0][:-1], 'patches', 'excel')
    os.makedirs(excel_file_folder, exist_ok=True)
    excel_file = os.path.join(excel_file_folder, filename + '.xlsx')

    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:

        for dataset in datasets:

            sheet_name_df = dataset['header']['INTERNAL']['timestamp']
            sheet_name_h = 'header_' + dataset['header']['INTERNAL']['timestamp']

            # Write the header
            h_df = pd.DataFrame({"Category": [], "Parameter": [], "Value": []})
            for k, v in dataset['header'].items():

                if isinstance(v, dict):
                    for kk, vv in dataset['header'][k].items():
                        h_df = pd.concat([h_df, pd.DataFrame({"Category": [k], "Parameter": [kk], "Value": [vv]})])

                elif k == 'PREPROCESSING_MAPPING':
                    c = ["MAPPING" for _ in range(len(v))]
                    p = [row['contour_name'] + ' -> ' + row['mapped_contour_name'] for index, row in v.iterrows()]
                    val = list(v['contour_id'])
                    h_df = pd.concat([h_df, pd.DataFrame({"Category": c, "Parameter": p, "Value": val})])
                else:
                    h_df = pd.concat([h_df, pd.DataFrame({"Category": [k], "Parameter": ['-'], "Value": [v]})])

            h_df.to_excel(writer, sheet_name=sheet_name_h)

            # Write the data
            dataset['dataframe'].to_excel(writer, sheet_name=sheet_name_df)

            # Adjust columns to match text
            adjust_df_to_excel_columns(writer, dataset['dataframe'], sheet_name_df)
            adjust_df_to_excel_columns(writer, h_df, sheet_name_h)
