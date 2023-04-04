from omero.gateway import BlitzGateway
import omero
import numpy as np
from omero.api import RoiOptions
import os,sys
from omero.rtypes import rstring, rlong, unwrap, rdouble, rint
import pandas as pd
from scipy import stats

def print_obj(obj, indent=0):
    print("""%s%s:%s  Name:"%s" (owner=%s)""" % (
        " " * indent,
        obj.OMERO_CLASS,
        obj.getId(),
        obj.getName(),
        obj.getName()))#obj.getOwnerOmeName()))

def connect(hostname, username, password):
    conn = BlitzGateway(username, password,
                        host=hostname, secure=True)
    conn.connect()
    conn.c.enableKeepAlive(60)
    return conn

def disconnect(conn):
    conn.close()

def create_roi(img, shapes):
    # create an ROI, link it to Image
    roi = omero.model.RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(img._obj)
    for shape in shapes:
        print(shape)
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    return updateService.saveAndReturnObject(roi)

def rgba_to_int(red, green, blue, alpha=255):
    """ Return the color as an Integer in RGBA encoding """
    r = red << 24
    g = green << 16
    b = blue << 8
    a = alpha
    rgba_int = r+g+b+a
    if (rgba_int > (2**31-1)):       # convert to signed 32-bit int
        rgba_int = rgba_int - 2**32
    return rgba_int

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
                    if (comment.split('-')[0] == '10HPFs')|(comment.split('-')[0][:4] == 'MFin')|(comment.split('-')[0] == 'MF'):
                        roi.removeShape(s)
                        roi = updateService.saveAndReturnObject(roi)
                        print('{} Removed'.format(s.getTextValue().getValue()))

def Generate10HPFs(image, r=3750):
    ImageID = image.getId()
    ImageName = os.path.splitext(image.getName())[0]
    roi_count = image.getROICount()
    print("{} ROI Count:{}".format(ImageName, roi_count))
    result = roi_service.findByImage(ImageID, None)
    mitosis_df = pd.DataFrame()
    x_centers = []
    y_centers = []
    texts = []
    cls_scores = []
    detect_scores = []

    for roi in result.rois:
        for s in roi.copyShapes():
            if type(s) == omero.model.EllipseI:
                comment = s.getTextValue().getValue()
                if len(comment.split('-')) == 3:
                    if (comment.split('-')[0][:2] == 'MF') & (comment.split('-')[1][0] == '0'):
                        x_centers.append(s.getX().getValue())
                        y_centers.append(s.getY().getValue())
                        texts.append(comment.split('-')[0])
                        cls_scores.append(float(comment.split('-')[1]))
                        detect_scores.append(float(comment.split('-')[2]))

    mitosis_df['x_center'] = np.array(x_centers)
    mitosis_df['y_center'] = np.array(y_centers)
    mitosis_df['cls_score'] = np.array(cls_scores)
    mitosis_df['detect_score'] = np.array(detect_scores)
    mitosis_df['text'] = np.array(texts)

    coord, data_in = CreateDensityMap(mitosis_df, r=r)
    ellipse = omero.model.EllipseI()
    ellipse.x = rdouble(coord[0]+r)
    ellipse.y = rdouble(coord[1]+r)
    ellipse.radiusX = rdouble(r)
    ellipse.radiusY = rdouble(r)
    ellipse.strokeColor = rint(rgba_to_int(0, 0, 255))
    z = image.getSizeZ() / 2
    t = 0
    ellipse.theZ = rint(z)
    ellipse.theT = rint(t)
    ellipse.textValue = rstring("10HPFs-with-{}MFs".format(data_in.shape[0]))
    create_roi(image, [ellipse])

    for i in range(data_in.shape[0]):
        ellipse = omero.model.EllipseI()
        ellipse.x = rdouble(data_in.x_center[i])
        ellipse.y = rdouble(data_in.y_center[i])
        ellipse.radiusX = rdouble(40)
        ellipse.radiusY = rdouble(40)
        ellipse.strokeColor = rint(rgba_to_int(255, 0, 0))
        ellipse.theZ = rint(z)
        ellipse.theT = rint(t)
        ellipse.textValue = rstring("MFin{}-{}-{}".format(i, data_in['cls_score'][i], data_in['detect_score'][i]))
        create_roi(image, [ellipse])

    print('10HPFs with densest mitotic activity for Slide {} generated'.format(ImageName))

def CreateDensityMap(data, r=3750):

    if data.shape[0] == 0:
        coord = (-9999, -9999)
        data_in = data
    elif data.shape[0] < 3:
        coord = (int(data.x_center[0] - r), int(data.y_center[0] - r))
        data_in = data
    else:
        x = data.x_center
        y = data.y_center
        values = np.vstack([x, y])
        kde = stats.gaussian_kde(values)
        density = kde(values)
        xy_max = values.T[np.argmax(density)]

        region_size = (2 * r, 2 * r)
        center = (xy_max[0], xy_max[1])
        coord_x = int(center[0] - r)
        coord_y = int(center[1] - r)
        coord = (coord_x, coord_y)

        data['x_center_in'] = x - coord_x
        data['y_center_in'] = y - coord_y
        data_in = data[data.x_center_in > 0]
        data_in = data_in[data_in.y_center_in > 0]
        data_in = data_in[data_in.x_center_in < region_size[0]]
        data_in = data_in[data_in.y_center_in < region_size[1]]
        data_in['distance^2'] = (data_in.x_center_in - r) ** 2 + (data_in.y_center_in - r) ** 2
        data_in = data_in[data_in['distance^2'] < r ** 2].reset_index(drop=True)

    return coord, data_in

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

Detection_Path = 'E:/Projects/DigitalPathologyAI/dgs1/'
diagnosis = 'SFT_high'
#desmoid_fibromatosis
#superficial_fibromatosis
SVS_dataset = pd.read_csv(Detection_Path + 'Inference_{}.csv'.format(diagnosis))
#SVS_dataset = SVS_dataset.iloc[4:10, :].reset_index(drop=True)

for project in conn.getObjects("Project", opts={'owner': my_exp_id,
                                                'order_by': 'lower(obj.name)',
                                                'limit': 5, 'offset': 0}):
    print_obj(project)
    for dataset in project.listChildren():
        print_obj(dataset, 2)

DeleteROIs(input('Enter Dataset ID:\n'))

for ImageID in SVS_dataset.id_omero.unique():
    image = conn.getObject('Image', ImageID)
    Generate10HPFs(image, r=3750)

conn.close()





