# Introduction to the preprocessing config file

Last updated: 25 March 2022.

The configuration file for preprocessing must be defined by the user, and its path is the only argument to be passed to the preprocessing script, located at `./PreProcessing/preprocessing.py`. An example can be found in the `./PreProcessing/config_files/` folder.

As of now, the preprocessing code is used to create .csv files containing the coordinates and labels of relevant patches in the whole slide images. Labels can be assigned by using contour files, which can be local or downloaded from an OMERO server.

The configuration file contains 4 subsections, which are
listed below, with a general description.

| Subsection | Description |
| :---       |   :---: |
| CONTOURS | Contour-specific parameters to generate patches |
| DATA | List of ids and patch dimensions |
| OMERO | server credentials if accessing contour from OMERO |
| PATHS | relevant paths for patch generation  |

# Detailed parameters

Each of the 4 subsections contain one or more parameters whose values must be
set. Details on the parameters can be found in the comprehensive tables below.

The **Options/restrictions** column, if not empty, provides the currently available values that can be used for the parameter.

## CONTOURS parameters

| Parameter      | Description | Options/restrictions     | Example |
| :---        |    :----:   |          ---: | ---:|
| Type        |  String defining the type of contour to be used.      | `'local'` for a local contours, `'omero'` to download contours on an OMERO server. If not using contours, leave empty `''` or `false`.          | `'local'`|
| Specific_Contours        |   A list of strings providing contours that should be processed; unspecified contours will not be processed.    |  list of contours, or `false` to process all available contours.  | `['Tumour', 'Fat']` |
| Remove_BG        | Float used to remove background voxels using a hard thresholding method. For each tile that belongs to a contour listed in `Remove_BG_Contours`, the tile is first colour-normalised using Macenko's approach, and the number of pixels whose gray-scale colour is > `Remove_BG` * 255 is calculated. If more than 50% of pixels in the tile meet the condition, the tile is recognised as background and is not processed.  | `float` between 0 and 1 to use, or set to `false`.           | `0` |
| Remove_BG_Contours        |   A list of strings specifying the contours on which the above procedure (`Remove_BG`) is applied.     | list of contours, or `false` to process all available contours. The contour names should be the original ones from Omero, not the proposed contours in `Contour_Mapping` below. | `['Tumour']` |
| Contour_Mapping        |   A list of strings to specify how to map each existing contour to a new name.    | list of mappings (see below), or `false` to keep the contours as they are.  | see below. |

For the `Contour_Mapping` parameter, we use the following nomenclature:

`['key1:value1', 'key2:value2', 'key3:value3']`

The `keys` should be contour names that are found in the local contour files, or OMERO contours. The corresponding `value` should be the name of the label that you want to assign to this specific contour. **By convention**, use the `remaining` key to assign remaining contours to a specific value, if you do not want to create a mapping for each possible label. **By default**, all `values` will be transformed to lower case and spaces will be removed.

For instance, consider the case where you have contours named "Tumour", "Tumor" and about 20 other contours, and you want to train an algorithm to differentiate tumours from all remaining datasets. The `Contour_Mapping` parameter would then be defined as:

`['Tumour:tumour', 'Tumor:tumour', 'remaining:not_tumour']`

Also, re-consider the above case, where you also want to differentiate fat tissue, and the annotator has created labels 'Fat [in]', 'Fat [out]'. You could write the `Contour_Mapping` parameter as:

`['Tumour:tumour', 'Tumor:tumour', 'Fat [in]:fat', 'Fat [out]:fat', 'remaining:not_tumour']`

Note that you could also write the above using the `Fat*` key, which would assign all contours begining with 'Fat' to the key 'fat':

`['Tumour:tumour', 'Tumor:tumour', 'Fat*:fat', 'remaining:not_tumour']`

## DATA parameters

| Parameter      | Description | Options/restrictions     | Example |
| :---        |    :----:   |          ---: | ---:|
| Label_Name        |  Name of the created label in csv file      | String  | Defaults to `'label'`. |
| Patch_Size        |  Size of each patch.      | Integer  | `256` |
| Vis        |  Visibility level to analyse. Leave to `0` for optimal results.      | Typically `0` to `3`.  | `0` |
| Ids        |  List of strings with the WSI ids to be processed.      |  | `['123','456']` |

## OMERO parameters

Will only be used if the `Type` parameter of `CONTOURS` is set to `omero`. Otherwise, leave all fields empty or `false`.

| Parameter      | Description | Options/restrictions     | Example |
| :---        |    :----:   |          ---: | ---:|
| Host        |  Address of the server.      | string  | `'128.0.0.0'` |
| User        |  Username to log into the server.      | string  | `'user'` |
| Pw        |  Password of the above user.      | string  | `'password'` |
| Target_Member        |  Name of the member owning the OMERO group from which you will download contours.      | string  | `'user'` |
| Target_Group        |  Name of the OMERO group from which you will download contours.      | string  | `'sarcoma study'` |

## PATHS parameters

| Parameter      | Description | Options/restrictions     | Example |
| :---        |    :----:   |          ---: | ---:|
| Patch        |  Path where patches (.csv files) will be created, for each file specified in the `'Ids'` parameters of DATA.      | string  | `'./patches/'` |
| Contour        |  If the `Type` parameter of CONTOURS is `'local'`, this is the path where contours (csv files) should be located, with their names including the whole slide images IDs that were specified in the `'Ids'` parameters of DATA. Also see below. Otherwise, if the `Type` parameter of CONTOURS is `'omero'`, this is the path where contours will be downloaded.  | string  | `'./contours/'` |
| QA        |  Path where images showing the result of preprocessing will be saved.      | string  | `'./QA/'` |
| WSI        |  Path where the whole slide images defined in the `Ids` parameter of DATA are located. Whole slide images can be in subfolders of the path specified in the `WSI` parameter.      | string  | `'./svs/'` |

Concerning the `Contour` parameter of PATHS: OMERO typically downloads the contours of a slide with id `1234` as `1234.svs [0]_roi_measurements.csv`. For simplicity, it is assumed that local contours also follow the same nomenclature. The current code may therefore only work with contour files that were originally pulled from Omero.
