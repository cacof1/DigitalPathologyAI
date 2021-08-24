# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 09:28:19 2021

@author: zhuoy
"""

import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px
import re
import time
from skimage import io
import os
import cv2
import sys
import h5py
import numpy as np
from wsi_core.WholeSlideImage import WholeSlideImage

DEBUG = True

NUM_ATYPES = 2
DEFAULT_FIG_MODE = "layout"
annotation_colormap = px.colors.qualitative.Light24

annotation_types = [
    "mitosis",
    "mitosis_like",
    ]

DEFAULT_ATYPE = annotation_types[0]

typ_col_pairs = [
    (t, annotation_colormap[n % len(annotation_colormap)])
    for n, t in enumerate(annotation_types)
]

color_dict = {}

type_dict = {}
for typ, col in typ_col_pairs:
    color_dict[typ] = col
    type_dict[col] = typ

options = list(color_dict.keys())
columns = ["Type", "X0", "Y0", "X1", "Y1"]


with open("assets/Howto.md", "r") as f:

    howto = f.read()


def debug_print(*args):
    if DEBUG:
        print(*args)


def coord_to_tab_column(coord):
    return coord.upper()


def time_passed(start=0):
    return round(time.mktime(time.localtime())) - start


def format_float(f):
    return "%.2f" % (float(f),)


def shape_to_table_row(sh):
    return {
        "Type": type_dict[sh["line"]["color"]],
        "X0": format_float(sh["x0"]),
        "Y0": format_float(sh["y0"]),
        "X1": format_float(sh["x1"]),
        "Y1": format_float(sh["y1"]),
    }


def default_table_row():
    return {
        "Type": DEFAULT_ATYPE,
        "X0": format_float(10),
        "Y0": format_float(10),
        "X1": format_float(20),
        "Y1": format_float(20),
    }


def table_row_to_shape(tr):
    return {
        "editable": True,
        "xref": "x",
        "yref": "y",
        "layer": "above",
        "opacity": 1,
        "line": {"color": color_dict[tr["Type"]], "width": 1, "dash": "solid"},
        "fillcolor": "rgba(0, 0, 0, 0)",
        "fillrule": "evenodd",
        "type": "rect",
        "x0": tr["X0"],
        "y0": tr["Y0"],
        "x1": tr["X1"],
        "y1": tr["Y1"],
    }


def shape_cmp(s0, s1):
    
    return (
        (s0["x0"] == s1["x0"])
        and (s0["x1"] == s1["x1"])
        and (s0["y0"] == s1["y0"])
        and (s0["y1"] == s1["y1"])
        and (s0["line"]["color"] == s1["line"]["color"])
    )


def shape_in(se):
    """ check if a shape is in list (done this way to use custom compare) """
    return lambda s: any(shape_cmp(s, s_) for s_ in se)


def index_of_shape(shapes, shape):
    for i, shapes_item in enumerate(shapes):
        if shape_cmp(shapes_item, shape):
            return i
    raise ValueError  


def annotations_table_shape_resize(annotations_table_data, fig_data):

    debug_print("fig_data", fig_data)
    debug_print("table_data", annotations_table_data)
    for key, val in fig_data.items():
        shape_nb, coord = key.split(".")
        shape_nb = shape_nb.split(".")[0].split("[")[-1].split("]")[0]

        annotations_table_data[int(shape_nb)][
            coord_to_tab_column(coord)
        ] = format_float(fig_data[key])

    return annotations_table_data


def shape_data_remove_timestamp(shape):

    new_shape = dict()
    for k in shape.keys() - set(["timestamp"]):
        new_shape[k] = shape[k]
    return new_shape


external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/image_annotation_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


filename = sys.argv[1]
basepath = 'C:/Users/zhuoy/Note/PathAI/'
coords_file = h5py.File(basepath + 'data/wsi/{}.h5'.format(filename),'r')
wsi_object = WholeSlideImage(basepath + 'data/wsi/{}.svs'.format(filename))
coords = coords_file['coords'] 
vis_level = 0
dim = (256,256)

dirs = os.listdir(basepath + 'annotation_app/{}/'.format(filename))
ids = []
for i in dirs:
    if os.path.splitext(i)[1] == '.npz':
        ids.append(i)

def visualize_prediction(file):
    data = np.load(basepath + 'annotation_app/{}/{}'.format(filename,file))
    top_left = data['top_left']
    boxes = data['boxes']
    scores = data['scores']
    img = np.array(wsi_object.wsi.read_region(top_left, vis_level, dim).convert("RGB"))
    count = 0
    for i in range(scores.shape[0]):
        box = boxes[i]
        score = scores[i]
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        if score > 0.7:
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255*score,255,0),thickness=1)
                        
    return img, top_left

filelist = []

for i in range(len(ids)):
    img, top_left = visualize_prediction(ids[i])
    fn = 'images/_{}_{}_{}.png'.format(i,top_left[0],top_left[1])
    io.imsave(fn,img)
    filelist.append(fn)
    
    if i%100 == 0:
        print('{}/{} patches loaded'.format(i,len(ids)))

fig = px.imshow(img=io.imread(filelist[0]), 
                title='Patch No.{}/{}; Top left coordinate:({},{}) '.format(filelist[0].split('_')[1],
                                                                            len(ids),
                                                                            filelist[0].split('_')[2],
                                                                            filelist[0].split('_')[3][:-4]),
                binary_backend="png")

fig.update_layout(
    newshape_line_color=color_dict[DEFAULT_ATYPE],
    margin=dict(l=0, r=0, b=0, t=30, pad=0),
    dragmode="drawrect",
)


button_gh = dbc.Button(
    "Learn more",
    id="howto-open",
    outline=True,
    color="secondary",

    style={"textTransform": "none"},
)

modal_overlay = dbc.Modal(
    [
        dbc.ModalBody(html.Div([dcc.Markdown(howto, id="howto-md")])),
        dbc.ModalFooter(dbc.Button("Close", id="howto-close", className="howto-bn",)),
    ],
    id="modal",
    size="lg",
    style={"font-size": "small"},
)

image_annotation_card = dbc.Card(
    id="imagebox",
    children=[
        dbc.CardHeader(html.H2("Annotation area")),
        dbc.Row(
            dbc.Col(html.H5('Slide ID: {}'.format(filename)))
            ),
        dbc.CardBody(
            [   
                dcc.Graph(
                    id="graph",
                    figure=fig,
                    config={"modeBarButtonsToAdd": ["drawrect", "eraseshape"]},
                )
            ]
        ),
        dbc.CardFooter(
            [
                dcc.Markdown(
                    "To annotate the above image, select an appropriate label on the right and then draw a "
                    "rectangle with your cursor around the area of the image you wish to annotate.\n\n"
                ),
                dbc.ButtonGroup(
                    [
                        dbc.Button("Previous image", id="previous", outline=True),
                        dbc.Button("Next image", id="next", outline=True),
                    ],
                    size="lg",
                    style={"width": "100%"},
                ),
            ]
        ),
    ],
)

annotated_data_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Annotated data")),
        dbc.CardBody(
            [
                dbc.Row(dbc.Col(html.H3("Coordinates of annotations"))),
                dbc.Row(
                    dbc.Col(
                        [
                            dash_table.DataTable(
                                id="annotations-table",
                                columns=[
                                    dict(
                                        name=n,
                                        id=n,
                                        presentation=(
                                            "dropdown" if n == "Type" else "input"
                                        ),
                                    )
                                    for n in columns
                                ],
                                editable=True,
                                style_data={"height": 40},
                                style_cell={
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                    "maxWidth": 0,
                                },
                                dropdown={
                                    "Type": {
                                        "options": [
                                            {"label": o, "value": o}
                                            for o in annotation_types
                                        ],
                                        "clearable": False,
                                    }
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": "Type"}, "textAlign": "left",}
                                ],
                                fill_width=True,
                            ),
                            dcc.Store(id="graph-copy", data=fig),
                            dcc.Store(
                                id="annotations-store",
                                data=dict(
                                    **{ 
                                        filename.split('_')[2]+'_'+filename.split('_')[3][:-4]: {"shapes": []}
                                        for filename in filelist
                                    },
                                    **{"starttime": time_passed()}
                                ),
                            ),
                            dcc.Store(
                                id="image_files",
                                data={"files": filelist, "current": 0},
                            ),
                            
                        ],
                    ),
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            html.H3("Create new annotation for"),
                            dcc.Dropdown(
                                id="annotation-type-dropdown",
                                options=[
                                    {"label": t, "value": t} for t in annotation_types
                                ],
                                value=DEFAULT_ATYPE,
                                clearable=False,
                            ),
                        ],
                        align="center",
                    )
                ),
            ]
        ),
        dbc.CardFooter(
            [
                html.Div(
                    [

                        html.A(
                            id="download",
                            download="{}mitosis.json".format(filename),

                            style={"display": "none"},
                        ),
                        dbc.Button(
                            "Download annotations", id="download-button", outline=True,
                        ),
                        html.Div(id="dummy", style={"display": "none"}),
                        dbc.Tooltip(
                            "You can download the annotated data in a .json format by clicking this button",
                            target="download-button",
                        ),
                    ],
                ),
                dcc.Markdown(
                    "Save the annotations as {}mitosis.json".format(filename)
                ),
            ]
        ),
    ],
)


app.layout = html.Div(
    [

        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(image_annotation_card, md=7),
                        dbc.Col(annotated_data_card, md=5),
                    ],
                ),
            ],
            fluid=True,
        ),
    ]
)


@app.callback(
    [Output("annotations-table", "data"), Output("image_files", "data")],
    [
        Input("previous", "n_clicks"),
        Input("next", "n_clicks"),
        Input("graph", "relayoutData"),
    ],
    [
        State("annotations-table", "data"),
        State("image_files", "data"),
        State("annotations-store", "data"),
        State("annotation-type-dropdown", "value"),
    ],
)
def modify_table_entries(
    previous_n_clicks,
    next_n_clicks,
    graph_relayoutData,
    annotations_table_data,
    image_files_data,
    annotations_store_data,
    annotation_type,
):
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if cbcontext == "graph.relayoutData":
        debug_print("graph_relayoutData:", graph_relayoutData)
        debug_print("annotations_table_data before:", annotations_table_data)
        if "shapes" in graph_relayoutData.keys():

            annotations_table_data = [
                shape_to_table_row(sh) for sh in graph_relayoutData["shapes"]
            ]
        elif re.match("shapes\[[0-9]+\].x0", list(graph_relayoutData.keys())[0]):

            annotations_table_data = annotations_table_shape_resize(
                annotations_table_data, graph_relayoutData
            )
        if annotations_table_data is None:
            return dash.no_update
        else:
            debug_print("annotations_table_data after:", annotations_table_data)
            return (annotations_table_data, image_files_data)
    image_index_change = 0
    if cbcontext == "previous.n_clicks":
        image_index_change = -1
    if cbcontext == "next.n_clicks":
        image_index_change = 1
    image_files_data["current"] += image_index_change
    image_files_data["current"] %= len(image_files_data["files"])
    if image_index_change != 0:

        annotations_table_data = []
        filename = image_files_data["files"][image_files_data["current"]]
        index = filename.split('_')[2]+'_'+filename.split('_')[3][:-4]
        debug_print(annotations_store_data[index])
        for sh in annotations_store_data[index]["shapes"]:
            annotations_table_data.append(shape_to_table_row(sh))
        return (annotations_table_data, image_files_data)
    else:
        return dash.no_update

@app.callback(
    [Output("graph", "figure"), Output("annotations-store", "data"),],
    [Input("annotations-table", "data"), Input("annotation-type-dropdown", "value")],
    [State("image_files", "data"), State("annotations-store", "data")],
)
def send_figure_to_graph(
    annotations_table_data, annotation_type, image_files_data, annotations_store
):
    if annotations_table_data is not None:
        filename = image_files_data["files"][image_files_data["current"]]
        index = filename.split('_')[2]+'_'+filename.split('_')[3][:-4]

        fig_shapes = [table_row_to_shape(sh) for sh in annotations_table_data]
        debug_print("fig_shapes:", fig_shapes)
        debug_print(
            "annotations_store[%s]['shapes']:" % (filename,),
            annotations_store[index]["shapes"],
        )

        new_shapes_i = []
        old_shapes_i = []
        for i, sh in enumerate(fig_shapes):
            if not shape_in(annotations_store[index]["shapes"])(sh):
                new_shapes_i.append(i)
            else:
                old_shapes_i.append(i)

        for i in new_shapes_i:
            fig_shapes[i]["timestamp"] = time_passed(annotations_store["starttime"])

        for i in old_shapes_i:
            old_shape_i = index_of_shape(
                annotations_store[index]["shapes"], fig_shapes[i]
            )
            fig_shapes[i]["timestamp"] = annotations_store[index]["shapes"][
                old_shape_i
            ]["timestamp"]
        shapes = fig_shapes
        debug_print("shapes:", shapes)
        fig = px.imshow(io.imread(filename),
                        title='Patch No.{}/{}; Top left coordinate:({},{}) '.format(filename.split('_')[1],
                                                                                    len(ids),
                                                                                    filename.split('_')[2],
                                                                                    filename.split('_')[3][:-4]),
                        binary_backend="png")
        fig.update_layout(
            shapes=[shape_data_remove_timestamp(sh) for sh in shapes],
            newshape_line_color=color_dict[annotation_type],
            margin=dict(l=0, r=0, b=0, t=30, pad=0),
            dragmode="drawrect",
        )
        
        annotations_store[index]["shapes"] = shapes
        return (fig, annotations_store)
    return dash.no_update

@app.callback(
    Output("modal", "is_open"),
    [Input("howto-open", "n_clicks"), Input("howto-close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


app.clientside_callback(
    """
function(the_store_data) {
    let s = JSON.stringify(the_store_data);
    let b = new Blob([s],{type: 'text/plain'});
    let url = URL.createObjectURL(b);
    return url;
}
""",
    Output("download", "href"),
    [Input("annotations-store", "data")],
)

# click on download link via button
app.clientside_callback(
    """
function(download_button_n_clicks)
{
    let download_a=document.getElementById("download");
    download_a.click();
    return '';
}
""",
    Output("dummy", "children"),
    [Input("download-button", "n_clicks")],
)


@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == "__main__":
    app.run_server()
