import sys

sys.path.append("./script/")

from dash import dcc, html, dash_table
from plot_utils import *
from plot_analyse import *

graph_left = []
graph_right = []

def build_tabs():
    return html.Div([html.Div(className='header',
                              children=[html.Img(src='./assets/images/logo.png',
                                                 className='logo-title')]),
                                        html.Div(id="tabs",
                                                 className="tabs",
                                                 children=[
                                        dcc.Tabs(
                                            id="app-tabs",
                                            value="tab1",
                                            className="custom-tabs",
                                            children=[
                                                dcc.Tab(id='analyse-tab',
                                                        label='Analyse',
                                                        className="custom-tab",
                                                        value='tab9',
                                                        selected_className="custom-tab--selected",
                                                        children=analyse_layout()),
                                                dcc.Tab(id="Specs-tab",
                                                        label="Modélisation",
                                                        value="tab1",
                                                        className="custom-tab",
                                                        selected_className="custom-tab--selected",
                                                        children=create_layout(),
                                                ),
                                                dcc.Tab(id="Control-chart-tab",
                                                        label="Resultats",
                                                        value="tab2",
                                                        className="custom-tab",
                                                        selected_className="custom-tab--selected"
                                                ),
                                                dcc.Tab(id='chat-tab',
                                                        label='Chatbot',
                                                        className="custom-tab",
                                                        value='tab3',
                                                        selected_className="custom-tab--selected"),
                ],
            )
        ],
    )
])

################################################ ONGLET 0 : Analyse #################################################

def analyse_layout():
    return html.Div(className='hub', children = [

        html.Div([dcc.Upload(id='upload-data-analyse', className="uploader", children=html.Div(
            ['Glisser et déposer ou ', html.A('Sélectionner le fichier')]
        )), html.Div(id='output-data-upload-analyse', style={"color": "#ffffff", "textAlign": "center"}),
                  ]),

        html.Br(),

        html.Div(className='form-input row', children=[
            html.Div(className='logo-and-label col', children=[
                html.Label('Choix de la cible', className='label-inline'),
            ]),
            html.Div(className='form-dropdown col', children=[
                dcc.Dropdown(id='target-dropdown-analyse',
                             options=dataprep.get_features(),
                             multi=False,
                             placeholder="Choisir la cible",
                             className='dropdown-inline',
                             style={'background-color': '#4e5567'}),
            ])
        ]),

        html.Br(),

        html.Div(className='form-input row', children=[
            html.Div(className='logo-and-label col', children=[
                html.Label('Choix de la variable date', className='label-inline'),
            ]),
            html.Div(className='form-dropdown col', children=[
                dcc.Dropdown(id='date-dropdown-analyse',
                             options=dataprep.get_features(),
                             multi=False,
                             placeholder="Choisir la date",
                             className='dropdown-inline',
                             style={'background-color': '#4e5567'}),
            ])
        ]),

        html.Br(),
        html.Button('Lancer l\'analyse', id='launch-button-analyse', n_clicks=0, className='launch-button-mod'),

        html.Div(id = "Graph-Container", children=[])

    ])

def build_analyse_panel():
    return ([
        dcc.Graph(figure=missing_values())
        # Tu rajoutes ici les différents graphiques que tu fais
        # Si tu veux faire des styles différent (taille des graphs, disposition, ...) tu peux rajouter des
        # html.Div dans cette liste
        # tu peux rajouter tout ce que tu veux dans la liste (des html.Label, ...)
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
                     children=[html.Br(), dcc.Markdown(id='variables-info-markdown', children=''), html.Br()]),

            html.Div(className='form-input row', children=[
                html.Div(id = 'predefined_vars_button', className='predefined-vars', children=[
                    html.Button('Interprétabilité', id='interpretabilite-button', n_clicks=0,
                                className='button-inline'),
                    html.Button('Performance', id='performance-button', n_clicks=0, className='button-inline'),
                ])]),

            html.Br(),

            html.Div(id="loading-div",
                     style={'display': 'none'},
                     children=[
                         dcc.Loading(
                             id="loading",
                             children=[html.Div(id="loading-output",
                                                className="loading-page"),
                                       html.Div(id='test_loading', children=[html.H3("", id = "loading-statement",  style={'color':'#FFFFFF'})])],
                             type="default",
                             fullscreen=True,
                         ),
                     ],
                     ),
            html.Br(),
            html.Button('Lancer la Modélisation', id='launch-button', n_clicks=0, className='launch-button-mod'),
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
                     children=[html.Label("1.Vérification des hypothèses"), html.Br()]))


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
                   children=[html.Br(), html.Label("2.Performances du modèle"), html.Br()]))



@render_this(graph_right)
def stability_plot():
    return html.Div(className='graphpart',
                    children=[
                        dcc.Graph(figure=courbe_roc())
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
                     children=[html.Br(), html.Label("3.Grille de Score"), html.Br()]))


@render_this(graph_right)
def table():
    return dash_table.DataTable(round(model.grid_score, 2).to_dict('records'),
                                [{"name": i, "id": i} for i in model.grid_score.columns],
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
                     children=[html.Br(), html.Label("4.Segmentation"), html.Br()]))

@render_this(graph_right)
def test():
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
                     children=[html.Label("5. MOC")]))

@render_this(graph_right)
def table():

    return dash_table.DataTable(round(model.default_proba, 4).to_dict('records'), [{"name": i, "id": i} for i in model.default_proba.columns],
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
           html.Div(className='logo-and-label-left col', children=[
               html.Img(src='./assets/images/homme.png', className='logo-left', style={"margin-left":"8px"}),
               html.Label(f"{model.df_score.shape[0]} observations.", className='label-left', style={"margin-left":"10px"}),
           ]),
               html.Br(),
           html.Div(className='logo-and-label-left col', children=[
               html.Img(src='./assets/images/croix.png', className='logo-left', style={"height":'40px'}),
               html.Label(f"{model.df_score[dataprep.target].sum()} défauts.", className='label-left',  style={"margin-left":"5px"}),
           ]),
               html.Br(),
           html.Div(className='logo-and-label-left col-left', children=[
               html.Img(src='./assets/images/pcentage.png', className='logo-left'),
               html.Label(f"{round(model.df_score[dataprep.target].mean(), 2) * 100} % de taux de défaut.", className='label-left'),
           ]),
           ])
       ]
   )
   )


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
    return html.Div(
                    children=[
                        html.Br(),
                        html.Button(["Données"], id="btn_df_score", className='download-button'),
                        dcc.Download(id="download-df-score"),
                    ], style={'textAlign': 'center'}
                    )


@render_this(graph_left)
def download_grille_score():
    return html.Div(
                    children=[
                        html.Br(),
                        html.Button("Grille de score", id="btn_grille_score", className='download-button'),
                        dcc.Download(id="download-grille-score"),
                    ], style={'textAlign': 'center'}
                    )


@render_this(graph_left)
def download_model():
    return html.Div(
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

################################################ ONGLET 3 : Chatbot #################################################
def chatbot():
    df = model.df_score
    dropdown_columns = df.columns.difference(['Score_ind', 'Classes']).tolist()
    return html.Div([
        html.Div([
            html.Div(id='md_title_chatbot', style={'margin-bottom': '50px'}, children=[
                html.Label('Quelle catégorie vous correspond le mieux ?', className='md_title'),
            ]),
            html.Br(),
        ], className='container'),

        html.Div(id='dynamic-radioitems-container', children=[
            html.Div([
                html.Div([
                    html.Label(f'Pour la variable {dropdown_columns[0]}:', className='label-inline message-label'),
                ], className='message-container'),
                html.Div([
                    dcc.RadioItems(
                        id={'type': 'dynamic-radioitems', 'index': 0},
                        options=[{'label': value, 'value': value} for value in df[dropdown_columns[0]].unique() if
                                 pd.notnull(value)],
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                        # Espacement et alignement horizontal
                        className='radio-inline selection-radio'
                    ),
                ], className='radioitems-container',
                    style={'background-color': '#8B0000', 'border-radius': '20px', 'color': 'white'}),
            ], className='form-input row', style={'margin-bottom': '50px'})
        ]),

        html.Button('Voir votre octroi de crédit', id='launch-chatbot-modeling', n_clicks=0,
                    className='launch-button', style={'margin-top': '20px', 'display': 'none'}),
        html.Div(id='score-ind-result'),
    ], className='hub')