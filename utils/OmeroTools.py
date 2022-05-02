import numpy as np
import pandas as pd
import os, sys
try:
    from omero.api import RoiOptions
    from omero.rtypes import rstring, rlong, unwrap, rdouble, rint
    from omero.gateway import BlitzGateway
    from omero.cli import cli_login, CLI
    import omero
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
    conn = BlitzGateway(username, password,
                        host=hostname, secure=True, **kwargs)
    conn.connect()
    conn.c.enableKeepAlive(60)
    return conn


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



def download_image(imageid, image_dir, user, host, pw):

    with cli_login("{user}@{host}", "-w", "{pw}") as cli:
        cli.invoke(["download", f'Image:{imageid}',image_dir])


def download_omero_ROIs(host=None, user=None, pw=None, target_group=None, target_member=None, ids=None,
                        download_path=None):
    # Connection to the correct group and identify the correct ID.
    conn, target_member_ID = connect_to_member(host, user, pw, target_group, target_member)

    # Set ROI options and load projects of target member
    roi_service = conn.getRoiService()
    roi_options = RoiOptions()
    #roi_options.userId = rlong(conn.getUserId())  # if you wanted to specify one user
    projects = conn.getObjects("Project", opts={'owner': target_member_ID,
                                                'order_by': 'lower(obj.name)',
                                                'limit': 5, 'offset': 0})

    # Loop over all images associated with the owner and find the files of interest. Score the ROIs in csv files.
    # Very loosely based on https://docs.openmicroscopy.org/omero/5.6.1/developers/Python.html.
    for project in projects:
        for dataset in project.listChildren():
            for image in dataset.listChildren():

                cur_img_name = image.getName()

                for id in ids:

                    id_string_omero = id + '.svs [0]'

                    if id_string_omero in cur_img_name:

                        print('OMERO: located {} ROIs from file "{}".'.format(image.getROICount(), id_string_omero))
                        found_rois = roi_service.findByImage(image.getId(), roi_options)

                        # Loop over each ROI:
                        ROI_points = []
                        ROI_name = []
                        ROI_id = []
                        image_name = []
                        ROI_type =  []
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

                                ROI_type.append('polygon')
                                ROI_name.append(s.getTextValue().getValue())
                                ROI_id.append(s.getId().getValue())
                                image_name.append(id_string_omero)

                        os.makedirs(download_path, exist_ok=True)
                        df = pd.DataFrame(
                            {'image_name': image_name, 'type': ROI_type, 'roi_id': ROI_id, 'Text': ROI_name,
                             'Points': ROI_points})
                        export_file = download_path + id_string_omero + '_roi_measurements.csv'
                        df.to_csv(export_file)
                        print('OMERO: {}/{} ROIs exported to location: {}'.format(len(ROI_name),str(image.getROICount()),export_file))


def upload_omero_polygon_ROIs(host=None, user=None, pw=None, n_contours_to_draw=-1, export_ROI=True,
                              contour_folder=None, target_IDs=None, target_contours=None, target_member=None,
                              target_group=None, contour_prefix_on_omero=None):
    # Upload currently works with Matlab-drawn contours only.

    # Connection to the correct group and identify the correct ID.
    conn, target_member_ID = connect_to_member(host, user, pw, target_group, target_member)

    # Set ROI options and load projects of target member
    roi_service = conn.getRoiService()
    roi_options = RoiOptions()
    roi_options.userId = rlong(conn.getUserId())
    projects = conn.getObjects("Project", opts={'owner': target_member_ID,
                                                'order_by': 'lower(obj.name)',
                                                'limit': 5, 'offset': 0})

    # Gather all image names and IDs
    img_names = list()
    img_IDs = list()
    for project in projects:
        for dataset in project.listChildren():
            for image in dataset.listChildren():
                img_names.append(image.getName())
                img_IDs.append(image.getId())

    # Loop over target contours and upload to Omero.
    for target_ID, target_contour in zip(target_IDs, target_contours):

        formatted_target_ID = str(target_ID) + '.svs [0]'

        omero_index = [index for index, value in enumerate(img_names) if formatted_target_ID in value]
        omero_ID = img_IDs[omero_index[0]]

        image = conn.getObject("Image", omero_ID)

        z = image.getSizeZ() / 2
        t = 0

        tmp = loadmat(os.path.join(contour_folder, target_contour), matlab_compatible=True, mat_dtype=True)

        contours = [key for key in tmp.keys() if key.lower().startswith('contour')]

        # Add contours one by one. The way I have encoded the contours (which can come from multiple class) is
        # contour_N_class_M  (where N is the contour #, and M is the class #). First, list the number of classes:

        contour_number = [int(contour.split('_')[1]) for contour in contours]
        class_number = [int(contour.split('_')[-1]) for contour in contours]
        N_classes = len(np.unique(class_number))

        # Either draw all contours or the specified number
        if n_contours_to_draw == -1:
            nc = len(contours)
        else:
            nc = n_contours_to_draw

        # Draw each contour now.
        for i in range(nc):
            for j in range(N_classes):
                cur_cont = tmp['contour_' + str(i + 1) + '_class_' + str(j + 1)]
                coords_str = ''
                for pt in np.arange(cur_cont.shape[0]):
                    coords_str = coords_str + str(cur_cont[pt, 0]) + ',' + str(cur_cont[pt, 1]) + ' '

                ## Export a ROI!
                polygon = omero.model.PolygonI()
                polygon.theZ = rint(z)
                polygon.theT = rint(t)
                R, G, B = (np.array(255) * rgb_cm[j]).astype(int)
                polygon.fillColor = rint(rgba_to_int(R, G, B, 50))
                polygon_name = contour_prefix_on_omero + ' class ' + str(j + 1) + ' contour ' + str(i + 1)
                polygon.textValue = rstring(polygon_name)
                polygon.strokeColor = rint(rgba_to_int(R, G, B))
                polygon.points = rstring(coords_str)
                if export_ROI:
                    create_roi(image, [polygon])
                print('ROI "{}" generated for {} ...'.format(polygon_name, formatted_target_ID))

    conn.close()


if __name__ == '__main__':

    # Example of how to use the download_omero_ROIs function:
    download_path = '/media/mikael/LaCie/sarcoma/contours/test/'
    host = '128.16.11.124'
    user = 'msimard'
    pw = 'msimard'
    target_member = 'msimard'
    target_group = 'Sarcoma Classification'
    ids = ['484759']  # should be a list!

    ids = ['500161']

    download_omero_ROIs(host=host, user=user, pw=pw, target_group=target_group, target_member=target_member, ids=ids,
                        download_path=download_path)

    # Example of how to use the upload_omero_polygon_ROIs function:

    # Input parameters
    host = '128.16.11.124'
    user = 'msimard'
    pw = 'msimard'
    n_contours_to_draw = 5
    export_ROI = False
    contour_folder = '/home/mikael/Dropbox/M/PostDoc/UCL/datasets/Digital_Pathology/zhuoyan/to_omero/contours/matlab_coordinates/'
    target_IDs = ['210000003', '210002933']
    target_contours = ['AEDClustering_210000003_C4_tumour_binary_matlab_contours.mat',
                       'AEDClustering_210002933_C4_tumour_binary_matlab_contours.mat']
    target_member = 'zhuoyanshen'
    target_group = 'Mitosis Detection'
    contour_prefix_on_omero = "Autoencoder"

    upload_omero_polygon_ROIs(host=host, user=user, pw=pw, n_contours_to_draw=n_contours_to_draw, export_ROI=export_ROI,
                              contour_folder=contour_folder, target_IDs=target_IDs, target_contours=target_contours,
                              target_member=target_member, target_group=target_group,
                              contour_prefix_on_omero=contour_prefix_on_omero)
