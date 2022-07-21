from omero.api import RoiOptions
from omero.rtypes import rstring, rlong, unwrap, rdouble, rint
from omero.gateway import BlitzGateway, Delete2
from omero.cli import cli_login, CLI
import omero
import numpy as np
from omero.cmd import DiskUsage2
from omero.cli import CmdControl
import sys
from omero.cmd import Duplicate
from collections import defaultdict
import pandas as pd
def FindIDfromName(name):
    params = omero.sys.ParametersI()
    query ="""                                                                                                                                                                                           
        select image.id, image.name from
        ImageAnnotationLink ial
        join ial.child a
        join ial.parent image        
        where image.name like :key
        """
    
    params.addString('key' , "%"+name+".svs%")
    result  = conn.getQueryService().projection(query, params, {"omero.group": "-1"})
    df_criteria = pd.DataFrame()
    for row in result: 
        temp = pd.DataFrame([[row[0].val, row[1].val,]],columns=["id_omero", "Name"])                            
        df_criteria = pd.concat([df_criteria, temp])

    print(name, df_criteria)
    return df_criteria


def duplicate(conn, data_type, IDs):
    dtype = data_type
    ids = IDs
    targets = defaultdict(list)
    for obj in conn.getObjects(dtype, ids):
        print(obj)
        targets[dtype].append(obj.id)
    print(targets)
    
    cmd = Duplicate()
    cmd.targetObjects = targets
    cb = conn.c.submit(cmd)
    return cb

def remove_map_annotations(conn, object):
    """Remove ALL Map Annotations on the object"""
    anns = list(object.listAnnotations())
    mapann_ids = [ann.id for ann in anns
                  if isinstance(ann, omero.gateway.MapAnnotationWrapper)]

    try:
        delete = Delete2(targetObjects={'MapAnnotation': mapann_ids})
        handle = conn.c.sf.submit(delete)
        conn.c.waitOnCmd(handle, loops=10, ms=500, failonerror=True,
                         failontimeout=False, closehandle=False)

    except Exception as ex:
        print("Failed to delete links: {}".format(ex.message))
    return

conn = BlitzGateway('msimard', 'msimard', host='128.16.11.124', secure=True)
conn.connect()
conn.SERVICE_OPTS.setOmeroGroup(-1)
imagenamelist = np.loadtxt(sys.argv[1], comments="#", delimiter=",", unpack=False,dtype='str')
print(len(imagenamelist))
for nb1, imagename in enumerate(imagenamelist):

    df = FindIDfromName(imagename)

    omero_image = conn.getObject("Image",df.iloc[0].id_omero)
    print(dir(omero_image))
    fileset = omero_image.getFileset().getId()
    print(type(fileset))
    cb = duplicate(conn, 'Fileset',[fileset])
    #cb = duplicate(conn, 'Image',[df.iloc[0].id_omero])

    duplicate_ids = cb.getResponse().duplicates['ome.model.core.Image']

    for nb2,duplicate_id in enumerate(duplicate_ids):
        to_delete = []
        image = conn.getObject("Image", duplicate_id)
        image.setName(str(nb1)+"-"+str(nb2))
        image.save()
        remove_map_annotations(conn,image)
        for ann in image.listAnnotations():
            if isinstance(ann, omero.gateway.FileAnnotationWrapper): to_delete.append(ann.id)
        try:conn.deleteObjects('Annotation', to_delete, wait=True)  
        except: print('ok')


conn.close()
