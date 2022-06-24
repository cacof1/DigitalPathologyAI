# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:12:02 2022

@author: zhuoy

# Work in progress - will be removed/largely modified. Works as a temporary solution towards a more
# complete data model.
"""

import omero
from omero.gateway import BlitzGateway
import copy
import re
from OmeroTools import *

# Connection
HOST = '128.16.11.124'
USER = input("Enter User Name:\n")
PASS = input("Enter User Password:\n")

conn = connect(HOST, USER, PASS)
my_exp_id = conn.getUser().getId()
conn.SERVICE_OPTS.setOmeroGroup('-1')

# Switch Group
image = conn.getObject("Image", 4842)  # my group
#image = conn.getObject("Image", 2173)  # for zhuoyan
group_id = image.getDetails().getGroup().getId()
conn.SERVICE_OPTS.setOmeroGroup(group_id)
print("Current group: ", group_id)

missing_value = ''  # fills empty fields using this value.

# for MS
csv_file = pd.read_excel('/Users/mikael/Dropbox/M/PostDoc/UCL/datasets/Digital_Pathology/SARCOMA/_Res_SpindleCellSarcoma_LowGrade_anonymised_MS_21_june_22.xlsx', keep_default_na=False).fillna(missing_value)
# for Zhuoyan
#csv_file = pd.read_csv('mitotic_counts_master.csv', keep_default_na=False).fillna(missing_value)


# for MS
keys = ['id_internal', 'diagnosis', 'tumour_grade', 'type']  # set up according to csv file for all but id (multiple possible sources)

# for Zhuoyan
#keys = ['id_internal', 'diagnosis', 'type']


for project in conn.listProjects():
    # original_file = get_original_file(project)
    # print("Original File", original_file.id.val, original_file.name.val)
    # provider = DownloadingOriginalFileProvider(conn)
    # temp_file = provider.get_original_file_data(original_file)

    print_obj(project)

    for dataset in project.listChildren():
        print_obj(dataset, 2)
        dataset_id = dataset.getId()
        # images_by_name = get_children_by_name(dataset)

        for image in conn.getObjects('Image', opts={'dataset': dataset_id}):

            if re.findall(r'\[(.*?)\]', image.getName())[0] == '0':
                image_name = os.path.splitext(image.getName())[0]
                image_id = image.getId()

                # This is not 100% robust, but the ID can come from either leeds or rnoh_leica fields. Take this into
                # consideration:
                if any(csv_file.leeds_id.astype(str) == image_name):
                    svs_index = csv_file[csv_file.leeds_id.astype(str) == image_name].index[0]
                    id_key = 'leeds_id'
                elif any(csv_file.rnoh_leica_id.astype(str) == image_name):
                    svs_index = csv_file[csv_file.rnoh_leica_id.astype(str) == image_name].index[0]
                    id_key = 'rnoh_leica_id'

                print('Slide:{}, ID:{}'.format(image_name, image_id))

                existing_kv = get_existing_map_annotations(image)
                updated_kv = copy.deepcopy(existing_kv)
                print("Existing key/value pairs:")
                for k, vset in existing_kv.items():
                    for v in vset:
                        print("   ", k, v)

                print("Adding key/value pair:")
                for key in keys:

                    if (key in csv_file.columns) or (key == 'id_internal'):  # otherwise do nothing with it.

                        if key == 'id_internal':
                            val = str(csv_file.loc[svs_index, id_key])
                        else:
                            val = str(csv_file.loc[svs_index, key])

                        # if key not in updated_kv:  not useful, because we might want to update keys
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


def remove_all_annotations(conn):

    # quick function if you want to remove all annotations and start from scratch.

    for project in conn.listProjects():

        print_obj(project)

        for dataset in project.listChildren():
            print_obj(dataset, 2)
            dataset_id = dataset.getId()

            for image in conn.getObjects('Image', opts={'dataset': dataset_id}):
                remove_map_annotations(conn, image)

    conn.close()
