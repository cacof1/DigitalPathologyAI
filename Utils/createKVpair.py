# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:12:02 2022

@author: zhuoy
"""

import omero
from omero.gateway import BlitzGateway
from omero.rtypes import rstring, rlong
import omero.scripts as scripts
from omero.cmd import Delete2
import os
import sys
import csv
import copy
import re
import pandas as pd
import numpy as np

from omero.util.populate_roi import DownloadingOriginalFileProvider
from collections import OrderedDict
from OmeroTools import *


## CONNECTION
HOST = '128.16.11.124'
USER =  input("Enter User Name:\n")
PASS = input("Enter User Password:\n")

conn = connect(HOST, USER, PASS)
my_exp_id = conn.getUser().getId()
conn.SERVICE_OPTS.setOmeroGroup('-1')

##Switch Group
image = conn.getObject("Image",4842)
group_id = image.getDetails().getGroup().getId()
conn.SERVICE_OPTS.setOmeroGroup(group_id)
print("Current group: ", group_id)

##Read the csv file
#Encoded diagnosis:
#'Solitary Fibrous Tumor (SFT)': 'SFT',
#'Desmoid Fibromatosis':'DF',
#'Nodular Fasciitis':'NF',
#'Superficial Fibromatosis':'SF',
#'De-differentiated Liposarcoma':'DDL',
#'Synovial Sarcoma':'SS',
#'Dermatofibrosarcoma protuberans (DFSP)':'DFSP',
#'Low grade fibromyxoid sarcoma':'FS',
#'Neurofibroma':'NF'

csv_file = pd.read_csv('Sarcoma_LowGrade_test.csv')
csv_file.drop('omero_status',axis=1,inplace=True)
csv_file = csv_file.astype({'leeds_id': 'int32'})
keys = ['diagnosis', 'tumour_grade','type'] #Set up the  accroding to the csv file


for project in conn.listProjects():
    #original_file = get_original_file(project)
    #print("Original File", original_file.id.val, original_file.name.val)
    #provider = DownloadingOriginalFileProvider(conn)
    #temp_file = provider.get_original_file_data(original_file)

    print_obj(project)
   
    for dataset in project.listChildren():
        print_obj(dataset, 2)
        dataset_id = dataset.getId()
        #images_by_name = get_children_by_name(dataset)
        
        for image in conn.getObjects('Image', opts={'dataset': dataset_id}):   

            if re.findall(r'\[(.*?)\]',image.getName())[0] == '0':
                image_name = os.path.splitext(image.getName())[0]
                image_id = image.getId()
                index = csv_file[csv_file.leeds_id==int(image_name)].index[0]
                print('Slide:{}, ID:{}'.format(image_name,image_id))
                
                existing_kv = get_existing_map_annotations(image)
                updated_kv = copy.deepcopy(existing_kv)
                print("Existing kv:")
                for k, vset in existing_kv.items():
                    for v in vset:
                        print("   ", k, v)
                    
                print("Adding kv:")
                for key in keys:
                    val = csv_file.loc[index, key]
                    if key not in updated_kv:
                        updated_kv[key] = set()
                        print("   ", key, val)
                        updated_kv[key].add(val)
                        
                if existing_kv != updated_kv:
                    obj_updated = True
                    print("The key-values pairs are different")
                    remove_map_annotations(conn, image)
                    map_ann = omero.gateway.MapAnnotationWrapper(conn)
                    namespace = omero.constants.metadata.NSCLIENTMAPANNOTATION
                    map_ann.setNs(namespace)
                    # convert the ordered dict to a list of lists
                    kv_list = []
                    for k, vset in updated_kv.items():
                        for v in vset:
                            kv_list.append([k, v])
                    map_ann.setValue(kv_list)
                    map_ann.save()
                    print("Map Annotation created", map_ann.id)
                    image.linkAnnotation(map_ann)
                else:
                    print("No change change in kv")

conn.close()
   
