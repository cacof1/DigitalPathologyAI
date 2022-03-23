import utils.PreProcessingTools
import toml

config = toml.load('/Users/mikael/Dropbox/M/PostDoc/UCL/Code/Python/DigitalPathologyAI/PreProcessing/config_files/tumour_identification_training_mac.ini')
vis = config['DATA']['Vis']
patch_size = config['DATA']['Patch_Size']
contour_type = config['CONTOURS']['Type']
patch_path = config['PATHS']['Patch']
contour_path = config['PATHS']['Contour']
QA_path = config['PATHS']['QA']
svs_path = config['PATHS']['WSI']
ids = config['DATA']['Ids']
specific_contours = config['CONTOURS']['Specific_Contours']
remove_outliers = config['CONTOURS']['Remove_BW']

omero_login = {'host': config['OMERO']['Host'], 'user': config['OMERO']['User'], 'pw': config['OMERO']['Pw'],
               'target_member': config['OMERO']['Target_Member'], 'target_group': config['OMERO']['Target_Group'],
               'ids': config['DATA']['Ids']}


csv_save_dirs = utils.PreProcessingTools.preprocess_WSI(vis=vis,
                                                        patch_size=patch_size,
                                                        patch_path=patch_path,
                                                        svs_path=svs_path,
                                                        QA_path=QA_path,
                                                        ids=ids,
                                                        contour_path=contour_path,
                                                        contour_type=contour_type,
                                                        specific_contours=specific_contours,
                                                        omero_login=omero_login,
                                                        remove_outliers=remove_outliers)
