import numpy as np
import pandas as pd
import os, sys
from collections import OrderedDict
from pathlib import Path

try:
    from omero.api import RoiOptions
    from omero.rtypes import rstring, rlong, unwrap, rdouble, rint
    from omero.gateway import BlitzGateway, Delete2
    from omero.cli import cli_login, CLI
    import omero
    from omero.cmd import DiskUsage2
    from omero.cli import CmdControl    
except ImportError:
    print('Unable to load omero modules. Make sure they are installed, otherwise you will not be able to use omero'
          'tools to load data.')

from scipy.io import loadmat
import matplotlib.pyplot as plt

cmap = plt.get_cmap('Set1')
rgb_cm = cmap.colors  # returns array-like color


# Functions to upload/download contours from Omero.

def connect(hostname, username, password, **kwargs):
    """
    Connect to an OMERO server
    :param hostname: Host name
    :param username: User
    :param password: Password
    :return: Connected BlitzGateway
    """
    conn = BlitzGateway(username, password, host=hostname, secure=True, **kwargs)
    conn.connect()
    conn.SERVICE_OPTS.setOmeroGroup('-1')
    conn.c.enableKeepAlive(60)
    return conn


def print_obj(obj, indent=0):
    """
    Helper method to display info about OMERO objects.
    Not all objects will have a "name" or owner field.
    """
    print("""%s%s:%s  Name:"%s" (owner=%s)""" % (
        " " * indent,
        obj.OMERO_CLASS,
        obj.getId(),
        obj.getName(),
        obj.getName()))  # obj.getOwnerOmeName()))


def get_existing_map_annotations(obj):
    """Get all Map Annotations linked to the object"""
    ord_dict = OrderedDict()
    for ann in obj.listAnnotations():
        if isinstance(ann, omero.gateway.MapAnnotationWrapper):
            kvs = ann.getValue()
            for k, v in kvs:
                if k not in ord_dict:
                    ord_dict[k] = set()
                ord_dict[k].add(v)
    return ord_dict


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


# We have a helper function for creating an ROI and linking it to new shapes
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


def connect_to_member(host, user, pw, target_group, target_member):
    conn = connect(host, user, pw, group=target_group)
    conn.SERVICE_OPTS.setOmeroGroup('-1')
    group = conn.getGroupFromContext()  # get current group
    owners, members = group.groupSummary()  # List the group owners and other members
    target_member_ID = None
    for m in members:
        if m.getOmeName() == target_member:
            target_member_ID = m.getId()

    return conn, target_member_ID


# Another helper for generating the color integers for shapes
def rgba_to_int(red, green, blue, alpha=255):
    """ Return the color as an Integer in RGBA encoding """
    r = red << 24
    g = green << 16
    b = blue << 8
    a = alpha
    rgba_int = r + g + b + a
    if rgba_int > (2 ** 31 - 1):  # convert to signed 32-bit int
        rgba_int = rgba_int - 2 ** 32
    return rgba_int


def download_annotation(image, image_dir):
    print("\nAnnotations on Dataset:", image.getName())
    for ann in image.listAnnotations():                
        if isinstance(ann, omero.gateway.FileAnnotationWrapper):
            print("File ID:", ann.getFile().getId(), ann.getFile().getName(), "Size:", ann.getFile().getSize())
            file_path = os.path.join(image_dir, ann.getFile().getName())
            with open(str(file_path), 'wb') as f:
                print("\nDownloading file to", file_path, "...")
                for chunk in ann.getFileInChunks():
                    f.write(chunk)
            print("File downloaded!")


def download_image(imageid, image_dir, user, host, pw):
    login_cmd = user + "@" + host
    with cli_login(login_cmd, "-w", pw) as cli:
        cli.invoke(["download", f'Image:{imageid}', image_dir])
        
def download_omero_ROIs(config, dataset, download_path=None):
    # Connection to the correct group and identify the correct ID. Use user as target member.
    conn = connect(config['OMERO']['Host'], config['OMERO']['User'], config['OMERO']['Pw'])  ## Group not implemented yet
    conn.SERVICE_OPTS.setOmeroGroup('-1')

    # Set ROI options and load projects of target member
    roi_service = conn.getRoiService()
    roi_options = RoiOptions()
    # roi_options.userId = rlong(conn.getUserId())  # if you wanted to specify one user

    # Loop over all images associated with the owner and find the files of interest. Score the ROIs in csv files.
    # Very loosely based on https://docs.openmicroscopy.org/omero/5.6.1/developers/Python.html.
    for image_id, image_name in zip(dataset['id_omero'],dataset['id_external']):
        
        image = conn.getObject('Image', image_id)
        print('OMERO: located {} ROIs from file "{}".'.format(image.getROICount(),image_name ))
        found_rois = roi_service.findByImage(image_id, roi_options)
        
        # Loop over each ROI:
        ROI_points = []
        ROI_name = []
        ROI_id = []
        image_name_list = []
        ROI_type = []
        for roi in found_rois.rois:
            for s in roi.copyShapes():
                
                if s.__class__.__name__ == 'PolygonI':
                    ROI_points.append(s.getPoints().getValue())
                elif s.__class__.__name__ == 'RectangleI':
                    xmin = str(round(s.getX().getValue(), 1))
                    xmax = str(round(s.getX().getValue(), 1) + round(s.getWidth().getValue(), 1))
                    ymin = str(round(s.getY().getValue(), 1))
                    ymax = str(round(s.getY().getValue(), 1) + round(s.getHeight().getValue(), 1))
                    fullstr = xmin + ',' + ymin + ' ' + xmax + ',' + ymin + ' ' + xmax + ',' + ymax + \
                        ' ' + xmin + ',' + ymax
                    ROI_points.append(fullstr)
                else:
                    RuntimeError('Shape " ' + s.__class__.__name__ + '" unsupported yet.')
                    
                ROI_type.append('polygon')
                ROI_name.append(s.getTextValue().getValue())
                ROI_id.append(s.getId().getValue())
                image_name_list.append(image_name)
        
        os.makedirs(download_path, exist_ok=True)
        df = pd.DataFrame({'image_name': image_name_list, 'type': ROI_type, 'roi_id': ROI_id, 'Text': ROI_name, 'Points': ROI_points})
        export_file = Path(download_path, image_name + '_roi_measurements.csv')
        df.to_csv(export_file)
        print('OMERO: {}/{} ROIs exported to location: {}'.format(len(ROI_name), str(image.getROICount()), export_file))
    conn.close()


def list_project_files(host=None, user=None, pw=None, target_group=None, target_member=None):
    # Connection to the correct group and identify the correct ID.
    conn, target_member_ID = connect_to_member(host, user, pw, target_group, target_member)
    projects = conn.getObjects("Project", opts={'owner': target_member_ID,
                                                'order_by': 'lower(obj.name)',
                                                'limit': 5, 'offset': 0})

    project_files = list()
    for project in projects:
        cp = project.getName()
        for dataset in project.listChildren():
            cd = dataset.getName()
            for image in dataset.listChildren():
                pth = cp + '/' + cd + '/' + image.getName()
                project_files.append(pth)  # TODO: maybe cat the project / dataset / image name also.

    return project_files


if __name__ == '__main__':
    # Example of how to use the download_omero_ROIs function:
    download_path = '/media/mikael/LaCie/sarcoma/contours/test/'
    host = '128.16.11.124'
    user = 'msimard'
    pw = 'msimard'
    target_member = 'msimard'
    target_group = 'Sarcoma Classification'
    ids = ['484759']  # ids should be a list
    download_image('484759', './Data', user, host, pw)
