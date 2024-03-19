from dash import dcc, html
import sys
sys.path.append("../script/")

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
                        # Intégrer ici le contenu de create_layout
                        children=create_layout(),
                    ),
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="Resultats",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=result_layout()
                    ),
                ],
            )
        ],
    )

def result_layout():
    return(html.Div(
        children = html.Div(id="result-output")
    ))


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
                        value='Logit-P',
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
                html.Div(className='form-dropdown col', children=[dcc.Dropdown(
                    id='variables-dropdown',
                    options=[{"label": "A", "value": "A"}, {"label": "B", "value": "B"}, {"label": "C", "value": "C"}],
                    multi=True,
                    placeholder="Choisir des variables",
                    className='dropdown-inline'
                ),
            ])]),

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
