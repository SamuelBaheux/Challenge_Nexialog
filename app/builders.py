import sys

sys.path.append("./script/")

from dash import dcc, html, dash_table
import dash_daq as daq
from plot_utils import *

graph_left = []
graph_right = []


def build_tabs():
    return html.Div([html.Div(className='header', children=[
            html.Img(src='./assets/images/logo.png', className='logo-title')]),
                            html.Div(
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
                    ])


################################################ ONGLET 1 : PARAMÈTRES #################################################

def create_layout():
    return html.Div(className='hub', children=[

        html.Div(className='container', children=[

            html.Div(id='md_title_0', children=[
                html.Label(className='md_title', children='1. Importer les données :')
            ]),
            html.Br(),

            html.Div([dcc.Upload(id='upload-data', className = "uploader", children=html.Div(
                    ['Glisser et déposer ou ', html.A('Sélectionner le fichier')]
                )), html.Div(id='output-data-upload', style={"color":"#ffffff", "textAlign":"center"}),
            ]),

            html.Br(),
            html.Br(),

            html.Div(id='md_title_1', children=[
               html.Label(className='md_title', children='2. Paramétrer la modélisation :')
            ]),

            html.Br(),

            html.Div(className='form-input row', children=[
                html.Div(className='logo-and-label col', children=[
                    html.Img(src='./assets/images/model.png', className='logo-inline', style={'marginLeft':'4px'}),
                    html.Label('Choix du Modèle', className='label-inline', style={'marginLeft':'4px'}),
                ]),
                html.Div(className='form-dropdown col', children=[
                    dcc.Dropdown(
                        id='model-choice',
                        options=[
                            {'label': 'Logit', 'value': 'Logit'},
                            {'label': 'XGBoost', 'value': 'XGBoost'}
                        ],
                        value='Logit',
                        className='dropdown-inline',
                        style={'background-color': '#4e5567'}
                    ),
                ]),
            ]),

            html.Br(),

            html.Div(className='form-input row', children=[
                html.Div(className='logo-and-label col', children=[
                    html.Img(src='./assets/images/target2.png', className='logo-inline'),
                    html.Label('Choix de la cible', className='label-inline'),
                ]),
                html.Div(className='form-dropdown col', children=[
                    dcc.Dropdown(id='target-dropdown',
                                 options=dataprep.get_features(),
                                 multi=False,
                                 placeholder="Choisir la cible",
                                 className='dropdown-inline',
                                 style={'background-color':'#4e5567'}),
                ])
            ]),

            html.Br(),

            html.Div(className='form-input row', children=[
                html.Div(className='logo-and-label col', children=[
                    html.Img(src='./assets/images/calendar.png', className='logo-inline'),
                    html.Label('Choix de la variable date', className='label-inline'),
                ]),
                html.Div(className='form-dropdown col', children=[
                    dcc.Dropdown(id='date-dropdown',
                                 options=dataprep.get_features(),
                                 multi=False,
                                 placeholder="Choisir la date",
                                 className='dropdown-inline',
                                 style={'background-color':'#4e5567'}),
                ])
            ]),

            html.Div(id='hidden-div'),
            html.Div(id='hidden-div1'),
            html.Div(id='hidden-div3'),

            html.Br(),

            html.Div(className='form-input row', children=[
                html.Div(className='logo-and-label col', children=[
                    html.Img(src='./assets/images/check.png', className='logo-inline'),
                    html.Label('Choix des variables explicatives', className='label-inline'),
                ]),
                html.Div(className='form-dropdown col', children=[
                    dcc.Dropdown(id='variables-dropdown',
                                 options=dataprep.get_features(),
                                 multi=True,
                                 placeholder="Choisir des variables",
                                 className='dropdown-inline',
                                 style={'background-color':'#4e5567'}),
                ])
            ]),

            html.Div(id='variables-info', className='variables-info',
                     children=[dcc.Markdown(id='variables-info-markdown', children='')]),

            html.Div(className='form-input row', children=[
                html.Div(id = 'predefined_vars_button', className='predefined-vars', children=[
                    html.Button('Interprétabilité', id='interpretabilite-button', n_clicks=0,
                                className='button-inline'),
                    html.Button('Performance', id='performance-button', n_clicks=0, className='button-inline'),
                ])]),

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
            html.Br(),
            html.Br(),
            html.Button('Lancer la Modélisation', id='launch-button', n_clicks=0, className='launch-button'),
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
def title_layout():
    return html.Div(children=[
        html.Div(className='graphpart',
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
                 ], style={'width': '65%'}),
        html.Div(className='graphpart',
                 children=[dcc.Graph(id='histo-graph')],
                 style={'width': '35%', 'margin-top': "2px"})],
        style={'display': 'flex', 'flexDirection': 'row'})


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
    grid_score = model.get_grid_score(dataprep.train, dataprep.target)

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
                                    'padding': '10px',
                                    'fontWeight': 'normal'
                                },
                                )


@render_this(graph_right)
def title_layout():
    return (html.Div(className='results-title',
                     children=[html.Br(), html.H3("4.Segmentation"), html.Br()]))

@render_this(graph_right)
def test():
    model.get_segmentation(dataprep.target)
    return html.Div(children=[
        html.Div(id = "slider-container", children=[dcc.RangeSlider(0,
                                           1000,
                                           id = "breaks-slider",
                                           value=model.breaks,
                                           marks={i: {'label': str(i), 'style': {'color': '#ffffff', 'fontSize': '14px'}}
                                                  for i in range(0, 1001, 250)},
                                           allowCross=False,
                                            tooltip={
                                                "placement": "top",
                                                "always_visible": True,
                                                "style": {"color": "white", "fontSize": "12px"},
                                            },
                                           className="custom-slider")],
                 style={'width': '95%', 'margin':"auto"}),
        html.Br(),])

@render_this(graph_right)
def test():
    return html.Div(children=[
        html.Div(children=[html.Br(),
                           html.Div([
                               dcc.Dropdown(
                                   id='graph-type-selector',
                                   className='dropdown-results',
                                   options=[
                                       {'label': 'Gini', 'value': 'gini'},
                                       {'label': 'Taux', 'value': 'taux'}
                                   ],
                                   value='gini',
                                   style={'marginBottom': '20px'}
                               ),
                               dcc.Graph(id='class-display', className='graphpart')
                           ])], style={'width': '65%'}),
        html.Div([html.Br(), html.Label("Segmentation - Tableau Récapitulatif", className="data-summary"), html.Br(),
            dash_table.DataTable(round(model.resultats, 2).to_dict('records'),
                                 [{"name": i, "id": i} for i in model.resultats.columns],
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
                                 id="table-id")
        ], style={'width': '35%', 'margin-top':"20px"}),
    ], style={'display': 'flex', 'flexDirection': 'row'})


@render_this(graph_right)
def title_layout():
    return (html.Div(className='results-title',
                     children=[html.H3("5. MOC")]))

@render_this(graph_right)
def table():
    moc_C = model.get_moc_c(dataprep.target)
    return dash_table.DataTable(round(moc_C, 4).to_dict('records'), [{"name": i, "id": i} for i in moc_C.columns],
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


@render_this(graph_left)
def title_layout():
    return (html.Div(className='left-title',
                     children=[html.Label("Données"),
                               html.Br()]))

@render_this(graph_left)
def title_layout():
   return(html.Div([html.Div(
       className='data-summary',
       children=[
           html.Label([
               f"- {model.df_score.shape[0]} observations.",
               html.Br(),
               f"- {model.df_score[dataprep.target].sum()} défauts.",
               html.Br(),
               f"- {round(model.df_score[dataprep.target].mean(), 2) * 100} % de taux de défaut."
           ])
       ]
   ),
   ]))


@render_this(graph_left)
def title_layout():
    return (html.Div(className='left-title',
                     children=[html.Br(), html.Label("Métriques"), html.Br()]))

@render_this(graph_left)
def stability_plot():
    dic_metrics = model.get_segmentation_metrics(dataprep.target, dataprep.date)
    roc_auc = model.get_metrics()["roc_auc"]*100
    return html.Div(className='graphpart',
                    children=[
                        html.Label("ROC-AUC :", className='left-panel-metric'),
                        dcc.Graph(figure=plot_metrics_leftpanel(roc_auc)),
                        html.Br(),
                        html.Label("Segmentation :", className='left-panel-metric'),
                        dcc.Graph(figure=plot_metrics_leftpanel(dic_metrics["count_seg"])),
                        html.Br(),
                        html.Label("Monotonie :", className='left-panel-metric'),
                        dcc.Graph(figure=plot_metrics_leftpanel(dic_metrics["count_monotonie"]))
                    ]
                    )

@render_this(graph_left)
def title_layout():
    return (html.Div(className='left-title',
                     children=[html.Br(), html.Label("Artefacts")]))

@render_this(graph_left)
def download_df_score():
    return html.Div(className='results-title',
                    children=[
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
        html.Br(),
        html.Br(),
        html.Br(),
        html.Div(
            className='panels-container',
            children=[
                html.Div(className='left-panel', children=auc_metric_panel),
                html.Div(className='right-panel', children=other_panels),
            ]
        )])

    return layout