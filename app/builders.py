import sys

sys.path.append("./script/")

from dash import dcc, html, dash_table
import dash_daq as daq
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
                                                        value='tab1',
                                                        selected_className="custom-tab--selected",
                                                        children=analyse_layout()),
                                                dcc.Tab(id="Specs-tab",
                                                        label="Modélisation",
                                                        value="tab0",
                                                        className="custom-tab",
                                                        selected_className="custom-tab--selected",
                                                        children=create_layout(),
                                                ),
                                                dcc.Tab(id="Control-chart-tab",
                                                        label="Modèle Classique",
                                                        value="tab2",
                                                        className="custom-tab",
                                                        selected_className="custom-tab--selected"
                                                ),
                                                dcc.Tab(id="Control-chart-tab-2",
                                                        label="Modèle Challenger",
                                                        value="tab3",
                                                        className="custom-tab",
                                                        selected_className="custom-tab--selected"
                                                        ),
                                                dcc.Tab(id='chat-tab',
                                                        label='Chatbot',
                                                        className="custom-tab",
                                                        value='tab4',
                                                        selected_className="custom-tab--selected"),
                                                dcc.Tab(id='denotching-tab',
                                                        label='Denotching',
                                                        className="custom-tab",
                                                        value='tab5',
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
        html.Br(),
        html.Div(id = "analyze_glob_data", children=[], style={"display":'None'}),
        html.Div(id="analyze_var_data", children=[], style={"display":'None'}),

    ])

def build_analyse_data():
    return [html.Div(
        children=[
            html.Br(),
            html.Div(id='md_title_analyse_0', children=[
                html.Label(className='md_title', children='1. Informations globale sur les données :')
            ]),
            html.Br(),
            html.Label("Extrait des données :", style={'color': '#FFFFFF', 'fontSize': "19px", "fontWeight":"bold"}),
            html.Br(),
            html.Div([
                html.Div([
                    dash_table.DataTable(
                        data=analyse.df.iloc[:10,:10].round(2).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in analyse.df.iloc[:10,:10].columns],
                        style_header={
                            'backgroundColor': 'rgb(76, 82, 94)',
                            'color': 'white',
                            'fontSize': '15px',
                            'height': '50px',
                            'whiteSpace': 'normal',
                            'padding': '10px',
                            'fontWeight': 'bold'
                        },
                        style_data={
                            'backgroundColor': 'rgb(78, 85, 103)',
                            'color': 'white',
                            'fontSize': '10px',
                            'height': '25px',
                            'whiteSpace': 'normal',
                            'padding': '10px',
                            'fontWeight': 'normal'
                        },
                        id="table-id-analyse",
                        column_selectable="multi",
                        page_action="native",  # Pagination activée
                        page_size=10  # Nombre de lignes par page
                    ),
                ], style={'width': '70%', 'display': 'inline-block', 'padding': '5px 50px 0 30px'}),

                html.Div([
                    html.Div(className='variables-info', children=[
                        dcc.Markdown(texte_analyse_globale())
                    ])
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'display': 'flex', 'width': '100%'}),

            html.Br(),
            html.Div(id='md_title_analyse_1', children=[
                html.Label(className='md_title', children='2. Valeurs Manquantes et Corrélation :')
            ]),
            html.Br(),

            html.Div([
                html.Div([
                    dcc.Graph(figure=missing_values())
                ], style={'width': '65%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Graph(figure=plot_correlation_matrix(True))
                ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'display': 'flex', 'width': '100%'}),

            html.Button('Vers l\'analyse par variable', id="analyse-var-button", className='button-analyse-change'),

        ]
    )]

def build_analyse_feature():
    return [html.Div(
        children=[
            dcc.Dropdown(id='target-dropdown-analyse-var',
                         options=analyse.df.columns.to_list(),
                         value=analyse.df.columns[2],
                         multi=False,
                         placeholder="Choisir la cible",
                         style={'background-color': '#4e5567'}),

            html.Div([
                html.Div([dcc.Graph(id = "plot_distrib"),
                ], style={'width': '70%', 'display': 'inline-block', 'padding': '5px 50px 0 30px'}),

                html.Div([
                    html.Br(),
                    html.Div(className='variables-info', id='variables-info-analyse', children=[])
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'display': 'flex', 'width': '100%'}),

            dcc.Graph(id = "plot_stability"),

            html.Button('Analyse globale', id="analyse-global-button", className='button-analyse-change'),

        ]
    )]

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
                            {'label': 'XGBoost', 'value': 'XGBoost'},
                            {'label': 'Les deux', 'value': 'both'},
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


def title_layout():
    return (html.Div(className='results-title',
                     children=[html.Label("1.Vérification des hypothèses"), html.Br()]))


def stability_plot(model_name):
    return html.Div(children=[
        html.Div(className='graphpart',
                 children=[
                     dcc.Dropdown(
                         id=f'stability-dropdown-{model_name}',
                         className='dropdown-results',
                         options=[{'label': label, 'value': col} for col, label in
                                  zip(*dataprep.get_explicative_features())],
                         value=dataprep.get_explicative_features()[0][0],
                         style={'marginBottom': '20px'}

                     ),
                     dcc.Graph(id=f'stability-graph-{model_name}')
                 ], style={'width': '65%'}),
        html.Div(className='graphpart',
                 children=[dcc.Graph(id=f'histo-graph-{model_name}')],
                 style={'width': '35%', 'margin-top': "2px"})],
        style={'display': 'flex', 'flexDirection': 'row'})


def title_layout_1():
    return (html.Div(className='results-title',
                   children=[html.Br(), html.Label("2.Performances du modèle"), html.Br()]))



def roc_plot(model):
    return html.Div(className='graphpart',
                    children=[
                        dcc.Graph(figure=courbe_roc(model))
                    ]
                    )

def shap_values(model):
    if model.model_name == 'xgb' :
        return html.Div(className='graphpart',
                        children=[
                            dcc.Graph(figure=plot_shap_values())
                        ]
                        )
    else :
        return html.Div()


def title_layout_2():
    return (html.Div(className='results-title',
                     children=[html.Br(), html.Label("3.Grille de Score"), html.Br()]))


def table(model):
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

def graph_dist(model):
    return html.Div(className='graphpart',
                    children = [dcc.Graph(figure=update_graph_dist_column("Score_ind", model))])


def title_layout_3():
    return (html.Div(className='results-title',
                     children=[html.Br(), html.Label("4.Segmentation"), html.Br()]))

def slider_breaks(model, model_name):
    return html.Div(children=[
        html.Div(id = "slider-container", children=[dcc.RangeSlider(0,
                                           1000,
                                           id = f"breaks-slider-{model_name}",
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

def segmentation(model, model_name):
    return html.Div(children=[
        html.Div(children=[html.Br(),
                           html.Div([
                               dcc.Dropdown(
                                   id=f'graph-type-selector-{model_name}',
                                   className='dropdown-results',
                                   options=[
                                       {'label': 'Gini', 'value': 'gini'},
                                       {'label': 'Taux', 'value': 'taux'}
                                   ],
                                   value='gini',
                                   style={'marginBottom': '20px'}
                               ),
                               dcc.Graph(id=f'class-display-{model_name}', className='graphpart')
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
                                 id=f"table-id-{model_name}")
        ], style={'width': '35%', 'margin-top':"20px"}),
    ], style={'display': 'flex', 'flexDirection': 'row'})


def title_layout_4():
    return (html.Div(className='results-title',
                     children=[html.Label("5. MOC")]))

def default_proba(model):
    return html.Div(className='graphpart',
                    children=[dcc.Graph(figure=proba_defaut(model.default_proba))])


def title_layout_5():
    return (html.Div(className='left-title',
                     children=[html.Label("Données"),
                               html.Br()]))

def info_data(model):
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


def title_layout_7():
    return (html.Div(className='left-title',
                     children=[html.Br(), html.Label("Métriques"), html.Br()]))

def gauges_combined(model):
    dic_metrics = model.get_segmentation_metrics(dataprep.target, dataprep.date)

    return html.Div(className='metricspart',
                    children=[
                        html.Label("Segmentation", className='left-panel-metric'),
                        daq.Gauge(
                            id="score-gauge",
                            max=100,
                            min=0,
                            size=130,
                            color={
                                "gradient": True,
                                "ranges": {
                                    "red": [0, 50],
                                    "yellow": [50, 70],
                                    "green": [70, 100],
                                },
                            },
                            value=dic_metrics["count_seg"],
                            showCurrentValue=True,
                        ),
                        html.Label("Monotonie", className='left-panel-metric'),
                        daq.Gauge(
                            id="score-gauge",
                            max=100,
                            min=0,
                            size=130,
                            color={
                                "gradient": True,
                                "ranges": {
                                    "red": [0, 50],
                                    "yellow": [50, 70],
                                    "green": [70, 100],
                                },
                            },
                            value=dic_metrics["count_monotonie"],
                            showCurrentValue=True,
                        )
                    ]
                    )


def metrics(model):
    dic_metrics = model.get_segmentation_metrics(dataprep.target, dataprep.date)
    roc_auc = model.get_metrics()["roc_auc"]*100
    return html.Div(className='dash-graph-container',
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

def title_layout_8():
    return (html.Div(className='left-title',
                     children=[html.Label("Artefacts")]))

def download_df_score():
    return html.Div(
                    children=[
                        html.Br(),
                        html.Button(["Données"], id="btn_df_score", className='download-button'),
                        dcc.Download(id="download-df-score"),
                    ], style={'textAlign': 'center'}
                    )


def download_grille_score():
    return html.Div(
                    children=[
                        html.Br(),
                        html.Button("Grille de score", id="btn_grille_score", className='download-button'),
                        dcc.Download(id="download-grille-score"),
                    ], style={'textAlign': 'center'}
                    )


def download_model():
    return html.Div(
                    children=[
                        html.Br(),
                        html.Button("Modèle", id="btn_model", className='download-button'),
                        dcc.Download(id="download-model"),
                    ], style={'textAlign': 'center'}
                    )


def setup_models_panels(model, left_list, right_list, model_name):

    @render_this(right_list)
    def render():
        return title_layout()

    @render_this(right_list)
    def render():
        return stability_plot(model_name)

    @render_this(right_list)
    def render():
        return title_layout_1()

    @render_this(right_list)
    def render():
        return roc_plot(model)

    @render_this(right_list)
    def render():
        return shap_values(model)

    @render_this(right_list)
    def render():
        return title_layout_2()

    @render_this(right_list)
    def render():
        return table(model)

    @render_this(right_list)
    def render():
        return graph_dist(model)

    @render_this(right_list)
    def render():
        return title_layout_3()

    @render_this(right_list)
    def render():
        return slider_breaks(model, model_name)

    @render_this(right_list)
    def render():
        return segmentation(model, model_name)
    @render_this(right_list)
    def render():
        return title_layout_4()

    @render_this(right_list)
    def render():
        return default_proba(model)

    @render_this(left_list)
    def render():
        return title_layout_5()

    @render_this(left_list)
    def render():
        return info_data(model)
    @render_this(left_list)
    def render():
        return title_layout_7()

    @render_this(left_list)
    def render():
        return gauges_combined(model)

    @render_this(left_list)
    def render():
        return title_layout_8()

    @render_this(left_list)
    def render():
        return download_df_score()

    @render_this(left_list)
    def render():
        return download_grille_score()
    @render_this(left_list)
    def render():
        return download_model()

def build_all_panels_mod(graph_left, graph_right):
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
    

def build_both_model():
    graph_left_classique = []
    graph_right_classique = []

    graph_left_challenger = []
    graph_right_challenger = []

    setup_models_panels(model_classique, graph_left_classique, graph_right_classique, "clas")
    setup_models_panels(model_challenger, graph_left_challenger, graph_right_challenger, "chal")

    layout_classique = build_all_panels_mod(graph_left_classique, graph_right_classique)
    layout_challenger = build_all_panels_mod(graph_left_challenger, graph_right_challenger)

    return layout_classique, layout_challenger

def build_logit_model():
    graph_left_classique = []
    graph_right_classique = []

    setup_models_panels(model_classique, graph_left_classique, graph_right_classique, "clas")

    layout_classique = build_all_panels_mod(graph_left_classique, graph_right_classique)

    return layout_classique

def build_xgboost_model():
    graph_left_challenger = []
    graph_right_challenger = []

    setup_models_panels(model_challenger, graph_left_challenger, graph_right_challenger, "chal")
    layout_challenger = build_all_panels_mod(graph_left_challenger, graph_right_challenger)

    return layout_challenger

################################################ ONGLET 3 : Chatbot #################################################
def format_option_label(value):
    try:
        value_clean = value.strip('[]')
        if ';' in value_clean:
            parts = value_clean.split(';')
            if parts[0].split(".")[0] == '0' :
                formatted = f"[{float(parts[0])};{float(parts[1])}]"
            else :
                formatted = f"[{int(float(parts[0]))};{int(float(parts[1]))}]"
        elif value_clean.replace('.', '', 1).isdigit():
            formatted = f"{float(value_clean):.0f}"
        else:
            formatted = ' '.join(word.capitalize() for word in value.replace('_', ' ').split())
        return formatted
    except Exception as e:
        print(f"Error formatting value {value}: {e}")
        return value

def format_option_column(column):
    if column == 'AMT_CREDIT_SUM_DEBT_disc_int' :
        return("Quel est votre montant de dette en cours ?")
    elif column == 'AMT_CREDIT_SUM_disc_int' :
        return ("Quel est votre montant de crédit en cours ?")
    if column == "DAYS_EMPLOYED_disc_int" :
        return("Depuis combien de jours êtes vous en emploi ?")
    if column == "EXT_SOURCE_1_disc_int" :
        return("Quel est votre score sur le score externe 1 ? ")
    if column == "EXT_SOURCE_2_disc_int" :
        return("Quel est votre score sur le score externe 2 ? ")
    if column == "EXT_SOURCE_3_disc_int" :
        return("Quel est votre score sur le score externe 3 ? ")
    if column == "NAME_INCOME_TYPE_discret" :
        return("Dans quelle catégorie de revenu vous situez vous ? ")
    return(f"Pour la variable {column} :")


def chatbot():
    if dataprep.model_name == "logit":
        model = model_classique
    elif dataprep.model_name == "xgb":
        model = model_challenger

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
                    html.Label(format_option_column(dropdown_columns[0]), className='label-inline message-label'),
                ], className='message-container'),
                html.Div([
                    dcc.RadioItems(
                        id={'type': 'dynamic-radioitems', 'index': 0},
                        options=[{'label': format_option_label(str(value)), 'value': value} for value in
                                 df[dropdown_columns[0]].dropna().unique()],
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                        className='radio-inline selection-radio'
                    ),
                ], className='radioitems-container'),
            ], className='form-input row', style={'margin-bottom': '50px'})
        ]),

        html.Div(id='score-ind-result'),
    ], className='hub')



################################################ ONGLET 4 : Denotching #################################################

def layout_denot():
    return html.Div( className='hub', children = [
        html.Div(id='md_title_0', children=[
            html.Label(className='md_title',
                       children='Impact d\'une dénotation sur la Probabilité de Défaut',
                       style={"textAlign" :"center"}),
            html.Br()
        ]),

        html.Div(children=[
            html.Label("Décalage des seuils :", className='label-inline'),
            html.Button("-25", id='button-25', className='denot-button', n_clicks=0),
            html.Button("-50", id='button-50', className='denot-button', n_clicks=0),
            html.Button("-75", id='button-75', className='denot-button', n_clicks=0),
            html.Button("-100", id='button-100', className='denot-button', n_clicks=0),
        ], style={"display":"flex"}),

        html.Br(),
        html.Br(),

        html.Div(id = "denotching-graph", children=[
            html.Label(children="Comparaison de la Probabilité de défaut : Avant / Après",
                       style={'font-size': "22px", "color": "#FFFFFF"}),
            dcc.Graph(id = "compare_PD"),
            html.Br(),
            html.Div([
                html.Div([
                    html.Label(children = "Comparaison de la Monotonie : Avant / Après",
                               style={'font-size':"22px", "color": "#FFFFFF"}),
                    dcc.Graph(id="compare_monotonie")
                ], style={'width': '50%', 'display': 'inline-block', 'padding': '5px 50px 0 30px'}),

                html.Div([
                    html.Label(children = "Comparaison de la Population : Avant / Après",
                               style={'font-size': "22px", "color": "#FFFFFF"}),
                    dcc.Graph(id="compare_pop")
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'display': 'flex', 'width': '100%'})
        ], style={'display' : "None"}),
    ])
