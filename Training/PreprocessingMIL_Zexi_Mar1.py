from typing import Union
import shutil
import openslide
import numpy as np
import os
import pandas as pd


def tumour_tile(file_path, percentage):  # path = .npy file path; percentage = the percentage of tumour
    s: Union[int, float, complex] = np.load(file_path, allow_pickle=True).item()
    firstkey = list(s.keys())[0]
    t = s[firstkey][1]
    t.columns = map(str.lower, t.columns)
    if 'prob_tissue_type_tumour' not in t.columns:
        t = pd.DataFrame()
        return t
    else:
        return t[t['prob_tissue_type_tumour'] >= percentage]  # return df of quantified tiles


def tiles_loc(numoftiles, dataframe):
    df = dataframe.sample(n=numoftiles)
    l = []
    for num in range(0, numoftiles):
        pair = (df['coords_x'].values[num], df['coords_y'].values[num])
        l.append(pair)
    return l


def load(currentdir, filename, locations, slide):
    pathh = os.path.join(currentdir, filename)
    #os.chdir(currentdir)
    os.mkdir(pathh)
    #folder_path = './' + filename
    for i in range(0, len(locations)):
        image = slide.read_region(location=locations[i], level=0, size=(256, 256))
        image_name = str(locations[i][1]) + '_' + str(locations[i][0]) + '.png'  # row = y, column = x
        os.chdir(pathh)
        image.save(image_name)
        os.chdir(path)


'''
path = '/home/dgs/data/DigitalPathologyAI/patches/484757.npy'

tem = np.load(path, allow_pickle=True).item()
firstkey = list(tem.keys())[0]
t = tem[firstkey][0]
print(t)
'''
path = "/home/dgs/data/DigitalPathologyAI"
des_path = '/media/dgs/lacie1/Dataset'

# 15 types
'''
#diagnosis_list = ['angioleiomyoma', 'de-differentiated_liposarcoma', 'dermatofibrosarcoma_protuberans_(dfsp)',
                  'desmoid_fibromatosis', 'extraskeletal_myxoid_chondrosarcomas',
                  'gastrointestinal_stromal_tumor_(gist)',
                  'low_grade_fibromyxoid_sarcoma', 'myxoid_liposarcoma', 'myxoma', 'neurofibroma', 'nodular_fasciitis',
                  'schwannoma',
                  'solitary_fibrous_tumor_(sft)', 'superficial_fibromatosis', 'synovial_sarcoma']
'''
csv_path = os.path.join(path, 'id_diagnosis.csv')
csv = pd.read_csv(csv_path)
# print(list(csv[csv['diagnosis'] == 'angioleiomyoma']['id']))

diagnosis_list_5types = ['angioleiomyoma', 'de_differentiated_liposarcoma',
                         'desmoid_fibromatosis', 'myxoma', 'synovial_sarcoma']

'''
for item in diagnosis_list_5types:
    current = os.path.join(des_path, item)  # enter the type folder
    patches = os.path.join(current, 'patches')
    os.mkdir(current)
'''

one = list(csv[csv['diagnosis'] == 'angioleiomyoma']['id'])
two = list(csv[csv['diagnosis'] == 'de_differentiated_liposarcoma']['id'])
three = list(csv[csv['diagnosis'] == 'desmoid_fibromatosis']['id'])
four = list(csv[csv['diagnosis'] == 'myxoma']['id'])
five = list(csv[csv['diagnosis'] == 'synovial_sarcoma']['id'])

'''
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)):
        filename = file.replace('.svs', '')
        if filename in one:



'''


def checkdiag(file, filename):
    if (file in one) or (filename in one):
        return 'angioleiomyoma'
    if (file in two) or (filename in two):
        return 'de_differentiated_liposarcoma'
    if (file in three) or (filename in three):
        return 'desmoid_fibromatosis'
    if (file in four) or (filename in four):
        return 'myxoma'
    if (file in five) or (filename in five):
        return 'synovial_sarcoma'
    else:
        return 'n'


# loop .svs files

os.chdir(path)
for file in os.listdir(path):
    os.chdir(path)
    if os.path.isfile(os.path.join(path, file)):
        filename = file.replace('.svs', '')
        diatype = checkdiag(file, filename)
        if diatype == 'n':
            continue
        npy_path = path + '/patches/' + filename + '.npy'
        if not os.path.exists(npy_path):
            continue
        tumour_df = tumour_tile(npy_path, 0.94)
        if len(tumour_df) < 350:
            os.remove(os.path.join(path, file))
            os.remove(npy_path)
            continue

        slide = openslide.open_slide(file)
        locations = tiles_loc(350, tumour_df)  # 100 tiles per slide
        load_path = os.path.join(des_path, diatype)
        load(load_path, filename, locations, slide)
        # os.remove(os.path.join(current, file))
        # os.remove(npy_path)

# os.rmdir(current + '/patches')
# os.chdir(path)
