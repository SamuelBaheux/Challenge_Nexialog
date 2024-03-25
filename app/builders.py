import sys

sys.path.append("./script/")

from dash import dcc, html, dash_table
import dash_daq as daq
from data_preparation import ConstantFeatures
from plot_utils import *

graph_left = []
graph_right = []


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
                    ),
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="Resultats",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected"
                    ),
                ],
            )
        ],
    )


################################################ ONGLET 1 : PARAMÈTRES #################################################

def create_layout():
    return html.Div(className='hub', children=[

        html.Div(className='header', children=[
            html.H3(children='Challenge Nexialog x MoSEF - Modélisation de la Probabilité de Défaut')]),

        html.Div(className='container', children=[

            html.Div(id='md_title_1', children=[
                dcc.Markdown(id='markdown_title', className='md_title', children='#### 1. Choisir le type de modèle :')
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
                dcc.Markdown(id='markdown_title2', className='md_title',
                             children='#### 2. Choisir les variables explicatives :')
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
                                 className='dropdown-inline', ),
                ])
            ]),

            html.Div(className='form-input row', children=[
                html.Div(className='predefined-vars', children=[
                    html.Button('Interprétabilité', id='interpretabilite-button', n_clicks=0,
                                className='button-inline'),
                    html.Button('Performance', id='performance-button', n_clicks=0, className='button-inline'),
                ])]),

            html.Div(id='variables-info', className='variables-info',
                     children=[dcc.Markdown(id='variables-info-markdown', children='')]),

            html.Br(),

            html.Div(id="loading-div",
                     children=[
                         dcc.Loading(
                             id="loading",
                             children=[html.Div(id="loading-output",
                                                className="loading-page")],
                             type="default",
                             fullscreen=True,
                         ),
                     ],
                     ),
            html.Button('Lancer la Modélisation', id='launch-button', n_clicks=0, className='button'),
        ])
    ])


################################################ ONGLET 2 : RÉSULTATS #################################################

def render_this(render_list):
    def decorator(function):
        render_list.append(function)

        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    return decorator


@render_this(graph_right)
def title_layout():
    return (html.Div(className='results-title',
                     children=[html.Br(), html.H3("1.Vérification des hypothèses"), html.Br()]))


@render_this(graph_right)
def stability_plot():
    return html.Div(className='graphpart',
                    children=[
                        dcc.Dropdown(
                            id='stability-dropdown',
                            className='dropdown-results',
                            options=[{'label': label, 'value': col} for col, label in
                                     zip(*dataprep.get_explicative_features())],
                            value=dataprep.get_explicative_features()[0][0],
                            style={'marginBottom': '20px'}
                        ),
                        dcc.Graph(id='stability-graph')
                    ]
                    )


@render_this(graph_right)
def title_layout():
    return (html.Div(className='results-title',
                     children=[html.Br(), html.H3("2.Performances du modèle"), html.Br()]))


@render_this(graph_right)
def stability_plot():
    return html.Div(className='graphpart',
                    children=[
                        dcc.Graph(figure=roc_curve())
                    ]
                    )

@render_this(graph_right)
def shap_values():
    if model.model_name == 'xgb' :
        return html.Div(className='graphpart',
                        children=[
                            dcc.Graph(figure=plot_shap_values())
                        ]
                        )
    else :
        return html.Div()


@render_this(graph_right)
def title_layout():
    return (html.Div(className='results-title',
                     children=[html.Br(), html.H3("3.Grille de Score"), html.Br()]))


@render_this(graph_right)
def table():
    grid_score = model.get_grid_score(dataprep.train)

    return dash_table.DataTable(round(grid_score, 2).to_dict('records'),
                                [{"name": i, "id": i} for i in grid_score.columns],
                                style_header={
                                    'backgroundColor': 'rgb(76, 82, 94)',
                                    'color': 'white',
                                    'fontSize': '20px',
                                    'height': '50px',
                                    'whiteSpace': 'normal',
                                    'padding': '15px',
                                    'fontWeight': 'bold'
                                },
                                style_data={
                                    'backgroundColor': 'rgb(78, 85, 103)',
                                    'color': 'white',
                                    'fontSize': '17px',
                                    'height': '40px',
                                    'whiteSpace': 'normal',
                                    'padding': '15px',
                                    'fontWeight': 'normal'
                                },
                                )


@render_this(graph_right)
def title_layout():
    return (html.Div(className='results-title',
                     children=[html.Br(), html.H3("4.Segmentation"), html.Br()]))


@render_this(graph_right)
def table():
    segments = model.get_segmentation()
    return dash_table.DataTable(round(segments, 2).to_dict('records'), [{"name": i, "id": i} for i in segments.columns],
                                style_header={
                                    'backgroundColor': 'rgb(76, 82, 94)',
                                    'color': 'white',
                                    'fontSize': '20px',
                                    'height': '50px',
                                    'whiteSpace': 'normal',
                                    'padding': '15px',
                                    'fontWeight': 'bold'
                                },
                                style_data={
                                    'backgroundColor': 'rgb(78, 85, 103)',
                                    'color': 'white',
                                    'fontSize': '16px',
                                    'height': '40px',
                                    'whiteSpace': 'normal',
                                    'padding': '15px',
                                    'fontWeight': 'normal'
                                },
                                )


@render_this(graph_right)
def classes_plot():
    return html.Div(children=[html.Br(),
                              html.Div([
                                  dcc.Dropdown(
                                      id='graph-type-selector',
                                      className='graphpart',
                                      options=[
                                          {'label': 'Gini', 'value': 'gini'},
                                          {'label': 'Taux', 'value': 'taux'}
                                      ],
                                      value='gini'
                                  ),
                                  dcc.Graph(id='class-display', className='graphpart')
                              ])])


@render_this(graph_right)
def title_layout():
    return (html.Div(className='results-title',
                     children=[html.H3("5. MOC")]))


@render_this(graph_left)
def title_layout():
    return (html.Div(className='results-title',
                     children=[html.H5("Données", style={"text-align": 'center'}),
                               html.Br(),
                               html.Div(children=[dcc.Markdown(
                                   f'''
                                  - {model.df_score.shape[0]} observations.
                                  - {model.df_score["TARGET"].sum()} défauts.
                                  - {round(model.df_score["TARGET"].mean(), 2) * 100} % de taux de défaut.
                                  '''
                               )]),
                               html.Br()
                               ]))


@render_this(graph_left)
def title_layout():
    return (html.Div(className='results-title',
                     children=[html.H5("Métriques", style={"text-align": 'center'})]))


@render_this(graph_left)
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


@render_this(graph_left)
def download_df_score():
    return html.Div(className='results-title',
                    children=[
                        html.Br(),
                        html.H5("Artefacts", style={"text-align": 'center'}),
                        html.Br(),
                        html.Button(["Données"], id="btn_df_score", className='download-button'),
                        dcc.Download(id="download-df-score"),
                    ], style={'textAlign': 'center'}
                    )


@render_this(graph_left)
def download_grille_score():
    return html.Div(className='results-title',
                    children=[
                        html.Br(),
                        html.Button("Grille de score", id="btn_grille_score", className='download-button'),
                        dcc.Download(id="download-grille-score"),
                    ], style={'textAlign': 'center'}
                    )


@render_this(graph_left)
def download_model():
    return html.Div(className='results-title',
                    children=[
                        html.Br(),
                        html.Button("Modèle", id="btn_model", className='download-button'),
                        dcc.Download(id="download-model"),
                    ], style={'textAlign': 'center'}
                    )


def build_all_panels():
    other_panels = [panel() for panel in graph_right]
    auc_metric_panel = [panel() for panel in graph_left]
    layout = html.Div(children=[
        html.Div(className="tab2-title",
                 children=[html.H2("Panneau des résultats")]),
        html.Div(
            className='panels-container',
            children=[
                html.Div(className='left-panel', children=auc_metric_panel),
                html.Div(className='right-panel', children=other_panels),
            ]
        )])

    return layout