from omero.gateway import BlitzGateway
import omero
import numpy as np
from omero.api import RoiOptions
import os, sys
from omero.rtypes import rstring, rlong, unwrap, rdouble, rint
import numpy as np
import pandas as pd
import cv2
import re
from Dataloader.Dataloader import *


def print_obj(obj, indent=0):
    print("""%s%s:%s  Name:"%s" (owner=%s)""" % (
        " " * indent,
        obj.OMERO_CLASS,
        obj.getId(),
        obj.getName(),
        obj.getName()))  # obj.getOwnerOmeName()))


def connect(hostname, username, password):
    conn = BlitzGateway(username, password,
                        host=hostname, secure=True)
    conn.connect()
    conn.c.enableKeepAlive(60)
    return conn


def disconnect(conn):
    conn.close()


def create_roi(img, shapes):
    roi = omero.model.RoiI()
    roi.setImage(img._obj)
    for shape in shapes:
        print(shape)
        roi.addShape(shape)

    return updateService.saveAndReturnObject(roi)


def rgba_to_int(red, green, blue, alpha=255):
    r = red << 24
    g = green << 16
    b = blue << 8
    a = alpha
    rgba_int = r + g + b + a
    if (rgba_int > (2 ** 31 - 1)):  # convert to signed 32-bit int
        rgba_int = rgba_int - 2 ** 32
    return rgba_int


def Send_Mitotic_Figures(conn, SVS_ID, ImageID, mitosis_df, cls_threshold=0.9, detect_threshold=0.5):
    image = conn.getObject('Image', ImageID)
    ImageName = os.path.splitext(image.getName())[0]
    assert ImageName == SVS_ID

    print('Start Sending Mitotic Figures for Slide {}'.format(SVS_ID))
    mitosis_df = mitosis_df[mitosis_df['prob_1'] > cls_threshold].reset_index(drop=True)
    mitosis_df = mitosis_df[mitosis_df['scores'] > detect_threshold].reset_index(drop=True)

    z = image.getSizeZ() / 2
    t = 0

    for i in range(mitosis_df.shape[0]):
        x = int((mitosis_df['xmax'][i] + mitosis_df['xmin'][i]) / 2) + mitosis_df['coords_x'][i]
        y = int((mitosis_df['ymax'][i] + mitosis_df['ymin'][i]) / 2) + mitosis_df['coords_y'][i]

        width = int(mitosis_df['xmax'][i] - mitosis_df['xmin'][i])
        height = int(mitosis_df['ymax'][i] - mitosis_df['ymin'][i])

        ellipse = omero.model.EllipseI()
        ellipse.x = rdouble(x)
        ellipse.y = rdouble(y)
        ellipse.radiusX = rdouble(width)
        ellipse.radiusY = rdouble(height)
        ellipse.theZ = rint(z)
        ellipse.theT = rint(t)
        ellipse.textValue = rstring(
            "MF{}-{}-{}".format(i, round(mitosis_df['prob_1'][i], 2), round(mitosis_df['scores'][i], 2)))
        # ellipse.textValue = rstring("MF{}".format(i))
        create_roi(image, [ellipse])

    print('Mitotic Figures for Slide {} Added'.format(ImageName))


def DeleteROIs(datasetId):
    for image in conn.getObjects('Image', opts={'dataset': datasetId}):
        ImageID = image.getId()
        roi_count = image.getROICount()
        print("{} ROI Count:{}".format(ImageID, roi_count))
        result = roi_service.findByImage(ImageID, None)
        for roi in result.rois:
            for s in roi.copyShapes():
                if type(s) == omero.model.EllipseI:
                    comment = s.getTextValue().getValue()
                    if (comment.split('-')[0] == '10HPFs') | (comment.split('-')[0][:4] == 'MFin') | (comment.split('-')[0][:2] == 'MF'):
                        conn.deleteObjects("Roi", [roi.id.val])
                        print('{} Removed'.format(s.getTextValue().getValue()))

def RenameROIs(ImageID):
    image = conn.getObject('Image', ImageID)
    roi_count = image.getROICount()
    print("{} ROI Count:{}".format(ImageID, roi_count))
    result = roi_service.findByImage(ImageID, None)

    z = image.getSizeZ() / 2
    t = 0

    for i, roi in enumerate(result.rois):
        for s in roi.copyShapes():
            if type(s) == omero.model.EllipseI:
                comment = s.getTextValue().getValue()
                if (comment.split('-')[0][:2] == 'MF'):

                    ellipse = omero.model.EllipseI()
                    ellipse.x = rdouble(s.getX().getValue())
                    ellipse.y = rdouble(s.getY().getValue())
                    ellipse.radiusX = rdouble(s.getRadiusX().getValue())
                    ellipse.radiusY = rdouble(s.getRadiusY().getValue())
                    ellipse.theZ = rint(z)
                    ellipse.theT = rint(t)
                    ellipse.textValue = rstring("MF{}".format(i))

                    #roi.removeShape(s)
                    #roi.addShape(ellipse)
                    #roi = updateService.saveAndReturnObject(roi)
                    conn.deleteObjects("Roi", [roi.id.val])
                    create_roi(image, [ellipse])
                    print('{} Renamed as MF{}'.format(s.getTextValue().getValue(), i))
#%%
HE_Path = '/home/dgs2/data/DigitalPathologyAI/'
Detection_Path = '/home/dgs2/data/DigitalPathologyAI/MitoticDetection/DetectionResults/'
diagnosis = 'SFT_high'
version = 0
SVS_dataset = pd.read_csv('/home/dgs2/data/DigitalPathologyAI/MitoticDetection/Inference_{}.csv'.format(diagnosis))
df = pd.read_csv(Detection_Path + 'classification_coords_{}{}.csv'.format(diagnosis,version))
df.rename(columns={'prob_yes':'prob_1','prob_no':'prob_0'},inplace=True)
df['SVS_ID'] = df['SVS_ID'].astype('str')
df = df[df['prob_tissue_type_Tumour']>0.94].reset_index(drop=True)

SVS_dataset['id_internal'] = SVS_dataset['id_internal'].astype('str')
print(df.SVS_ID.unique())
SVS_dataset = SVS_dataset[SVS_dataset['id_internal'].isin(df.SVS_ID.unique())].reset_index(drop=True)
#%%
HOST = '128.16.11.124'
USER = 'msimard'
PASS = 'msimard'
conn = connect(HOST, USER, PASS)

roi_service = conn.getRoiService()
updateService = conn.getUpdateService()

my_exp_id = conn.getUser().getId()
conn.SERVICE_OPTS.setOmeroGroup('55') ## cross-group querying

roi_options = RoiOptions()
roi_options.userId = rlong(conn.getUserId())

for project in conn.getObjects("Project", opts={'owner': my_exp_id,
                                            'order_by': 'lower(obj.name)',
                                            'limit': 5, 'offset': 0}):
    print_obj(project)

    for dataset in project.listChildren():
        print_obj(dataset, 2)

#DeleteROIs(input('Enter Dataset ID:\n'))
RenameROIs(12309)#(input('Enter Image ID:\n'))

'''for SVS_ID in ['485295','485333','485349','485395','485512','485515','484959','484945',]:#df.SVS_ID.unique():
    mitosis_df = df[df['SVS_ID'] == SVS_ID].reset_index(drop=True)
    ImageID = SVS_dataset[SVS_dataset['id_internal'] == SVS_ID]['id_omero'].unique()[0]
    Send_Mitotic_Figures(conn, SVS_ID, ImageID, mitosis_df, cls_threshold=0.5, detect_threshold=0.8)'''

conn.close()


