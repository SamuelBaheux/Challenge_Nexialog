import sys
sys.path.append("./script/")

from dash import dcc, html, dash_table
import dash_daq as daq
from data_preparation import ConstantFeatures
from vars import df, dataprep
from plot_utils import *


graph_list = []

def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab1",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Specs-tab",
                        label="Paramètres",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=create_layout(),
                    )
                ],
            )
        ],
    )

def render_this(render_list):
    def decorator(function):
        render_list.append(function)

        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    return decorator

@render_this(graph_list)
def title_layout():
    return(html.Div(className= 'results-title',
                    style={'textAlign': 'center'},
                    children=[html.H2("Résultats")]))

@render_this(graph_list)
def title_layout():
    return(html.Div(className= 'results-title',
                    children=[html.H3("1.Vérification des hypothèses")]))

@render_this(graph_list)
def stability_plot():
    return html.Div(className='graphpart',
             children=[
                 dcc.Dropdown(
                     id='stability-dropdown',
                     options=[{'label': label, 'value': col} for col, label in zip(*dataprep.get_explicative_features())],
                     value=dataprep.get_explicative_features()[0][0],
                     style={'marginBottom': '20px'}
                 ),
                 dcc.Graph(id='stability-graph')
             ]
    )

@render_this(graph_list)
def title_layout():
    return(html.Div(className= 'results-title',
                    children=[html.H3("2.Performances du modèle")]))


@render_this(graph_list)
def stability_plot():
    return html.Div(className='graphpart',
             children=[
                 dcc.Graph(figure = roc_curve())
             ]
    )

@render_this(graph_list)
def title_layout():
    return(html.Div(className= 'results-title',
                    children=[html.H3("3.Grille de Score")]))
@render_this(graph_list)
def table():
    grid_score = model.get_grid_score(dataprep.train)

    return dash_table.DataTable(grid_score.to_dict('records'), [{"name": i, "id": i} for i in grid_score.columns])

@render_this(graph_list)
def title_layout():
    return(html.Div(className= 'results-title',
                    children=[html.H3("4.Segmentation")]))

@render_this(graph_list)
def title_layout():
    return(html.Div(className= 'results-title',
                    children=[html.H3("5. MOC")]))
@render_this(graph_list)
def AUC_Metric():
    return html.Div(className='metricspart',
             children=[
                 daq.Gauge(
                     id="score-gauge",
                     max=1,
                     min=0,
                     size=150,
                     color={
                         "gradient": True,
                         "ranges": {
                             "red": [0, 0.5],
                             "yellow": [0.5, 0.7],
                             "green": [0.7, 1],
                         },
                     },
                     value=model.get_metrics()["roc_auc"],
                     showCurrentValue=True,
                 )
             ]
        )

def build_all_panels():
    # Séparez AUC_Metric des autres panels
    auc_metric_panel = [panel() for panel in graph_list if panel.__name__ == "AUC_Metric"]
    other_panels = [panel() for panel in graph_list if panel.__name__ != "AUC_Metric"]

    # Construisez une disposition avec AUC_Metric à gauche et les autres panels à droite
    layout = html.Div(
        className='panels-container',
        children=[
            html.Div(className='left-panel', children=auc_metric_panel),
            html.Div(className='right-panel', children=other_panels),
        ]
    )

    return layout

def create_layout():
    return html.Div(className='hub', children=[

        html.Div(className='header', children=[
            html.H3(children='Challenge Nexialog x MoSEF - Modélisation de la Probabilité de Défaut')]),


        html.Div(className='container', children=[

            html.Div(id='md_title_1',children=[
                dcc.Markdown(id='markdown_title', className='md_title', children='##### 1. Choisir le type de modèle :')
            ]),

            html.Div(className='form-input row', children=[
                html.Div(className='form-label col', children=[
                    html.Label('Type de modèle', className='label-inline'),
                ]),
                html.Div(className='form-dropdown col', children=[
                    dcc.Dropdown(
                        id='model-choice',
                        options=[
                            {'label': 'Logit', 'value': 'Logit'},
                            {'label': 'XGBoost', 'value': 'XGBoost'}
                        ],
                        value='Logit',
                        className='dropdown-inline'
                    ),
                ]),
            ]),

            html.Br(),
            html.Div(id='md_title_2', children=[
                dcc.Markdown(id='markdown_title2', className='md_title', children='##### 2. Choisir les variables explicatives :')
            ]),

            html.Div(className='form-input row', children=[
                html.Div(className='form-label col', children=[
                    html.Label('Choix des Variables', className='label-inline'),
                    ]),
                html.Div(className='form-dropdown col', children=[
                    dcc.Dropdown(id='variables-dropdown',
                                 options=ConstantFeatures().all_features,
                                 multi=True,
                                 placeholder="Choisir des variables",
                                 className='dropdown-inline',),
                ])
            ]),

            html.Div(className='form-input row', children=[
            html.Div(className='predefined-vars', children=[
                html.Button('Interprétabilité', id='interpretabilite-button', n_clicks=0, className='button-inline'),
                html.Button('Performance', id='performance-button', n_clicks=0, className='button-inline'),
            ])]),

            html.Div(id='variables-info', className='variables-info', children=[dcc.Markdown(id='variables-info-markdown', children='')]),

            html.Br(),
            html.Div(id='md_title_3', children=[
                dcc.Markdown(id='markdown_title4', className='md_title',
                             children='##### 3. Choisir la Discrétisation des variables numériques :')
            ]),

            html.Div(className='form-input row', children=[
                html.Div(className='form-label col', children=[
                    html.Label('Choix de Discrétisation', className='label-inline'),
                ]),
                html.Div(className='form-dropdown col', children=[
                    dcc.Dropdown(
                        id='discret-choice',
                        options=[
                            {'label': 'Discrétiser', 'value': True},
                            {'label': 'Ne pas discrétiser', 'value': False}
                        ],
                        value='Discrétiser',
                        className='dropdown-inline'
                    ),
                ]),
            ]),

            html.Button('Générer', id='launch-button', n_clicks=0, className='button'),
        ])
    ])
