
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:17:40 2021

@author: zhuoy
"""
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import pandas as pd
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

patient_list = sys.argv[1]

#%%
app.layout = html.Div([
        
    html.H4(children='3D Visualization of the Feature Embeddings'),
    
    html.Div([
        dcc.Graph(id='3d-graph')
        ], style={'width': '89%','height': '89%', 'display': 'inline-block', 'padding': '0 20'}),
    
    html.Div([
                dcc.RadioItems(
                        id='displayed-proportion',
                        options=[
                                {'label': '25%', 'value': 0.25},
                                {'label': '50%', 'value': 0.5},
                                {'label': '75%', 'value': 0.75},
                                {'label': '100%', 'value': 1.0},
                                ],
                        value=1.0,
                        )
                
                ], style={'width': '9%', 'display': 'inline-block'}),
])

@app.callback(
        Output('3d-graph', 'figure'),
        [Input('displayed-proportion', 'value')]
        )

def upgrade_graph(proportion):
    
    COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
            'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
            'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
            'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
            'rgb(188, 189, 34)', 'rgb(23, 190, 207)']
    
    
    groups = []
    
    for i ,patient in enumerate(patient_list):
        df = pd.read_csv('feature_{}'.format(patient))
        df = df.sample(frac=proportion)
        
        trace = go.Scatter3d(x = df['c1'],
                             y = df['c2'], 
                             z = df['c3'], 
                             name = patient,
                             mode = 'markers',
                             marker=dict(
                                     size=2,
                                     color=COLORS[i],
                                     opacity=0.8
                                     )
                             )
    
        groups.append(trace)

    data = go.Data(groups)
    
    layout = {
            "scene": {
                    "xaxis": {
                            "title": {"text": "Component 1"}, 
                            "showline": False
                            }, 
                            "yaxis": {
                                    "title": {"text": "Component 2"}, 
                                    "showline": False
                                    }, 
                                    "zaxis": {
                                            "title": {"text": "Component 3"}, 
                                            "showline": False
                                            }
                                    }, 
                                    "title": {"text": "Feature Embeddings (3D)"}
                                    }

    fig = go.Figure(data=data, layout=layout)
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
