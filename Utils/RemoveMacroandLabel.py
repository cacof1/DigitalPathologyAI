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
import struct
import tifffile
"""
with tifffile.TiffFile('Data/484757.svs', mode='r+b') as svs:
    assert svs.is_svs
    fh = svs.filehandle
    print(dir(svs.pages[0]))

    tiff = svs.tiff
    for page in svs.pages[::-1]:
        print(page,page.treeindex)
        if page.subfiletype not in (1, 9):
            print(page.subfiletype)
            continue  # not a label or macro image
        # zero image data in page
        print('next')

        for offset, bytecount in zip(page.dataoffsets, page.databytecounts):
            fh.seek(offset)
            fh.write(b'\0' * bytecount)
        # seek to position where offset to label/macro page is stored
        previous_page = svs.pages[page.index - 1]  # previous page
        fh.seek(previous_page.offset)
        tagno = struct.unpack(tiff.tagnoformat, fh.read(tiff.tagnosize))[0]
        offset = previous_page.offset + tiff.tagnosize + tagno * tiff.tagsize
        fh.seek(offset)
        # terminate IFD chain
        fh.write(struct.pack(tiff.offsetformat, 0))

        print(f'wiped {page}')

"""
def FindIDfromName():
    params = omero.sys.ParametersI()
    query ="""
    select p from Image p
    """
    params = omero.sys.Parameters()
    queryService = conn.getQueryService()
    result = queryService.findAllByQuery(query, None)#{"omero.group": "-1"}) 
    #result = queryService.findAllByString("Image","Name","*",True,filter=None)
    df_criteria = pd.DataFrame()
    for row in result:
        temp = pd.DataFrame([[row.getId()._val, row.getName()._val,]],columns=["id_omero", "Name"])
        df_criteria = pd.concat([df_criteria, temp])

    return df_criteria


conn = BlitzGateway('root', 'mortavar1988', host='128.16.11.124',port=4064)#, secure=True)
conn.connect()

#df = FindIDfromName()

conn.close()
