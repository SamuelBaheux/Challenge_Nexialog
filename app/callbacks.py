import sys
sys.path.append('./script')

from dash.dependencies import Input, Output, State
import dash
from dash import html, dcc

from builders import create_layout, build_all_panels
from data_preparation import *
from Logit_utils import *
from XGB_utils import *
from vars import *
from plot_utils import *


def register_callbacks(app):
    @app.callback(
        [Output('app-tabs', 'value'),
         Output('app-tabs', 'children'),],
        [Input('launch-button', 'n_clicks'),
         Input("variables-dropdown", 'value'),
         Input('model-choice', 'value')],
        prevent_initial_call=True)

    def update_result(n_clicks, features, model_choice):
        if n_clicks and n_clicks > 0:
            if model_choice is None or features is None :
                return dash.no_update

            print("Initialisation des données")
            dataprep.initialize_data(features)
            train_prepared = dataprep.get_prepared_data()

            print("Entraînement du modèle")
            if model_choice == 'Logit' :
                model.init_model('logit')
                model.init_data(train_prepared, dataprep.discretizer.intervalles_dic)
                model.run_model()


            elif model_choice == 'XGBoost' :
                model.init_model('xgb')
                model.init_data(train_prepared, dataprep.discretizer.intervalles_dic)
                model.run_model()

            return ('tab2', [
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
                    label="Résultats",
                    value="tab2",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                    children=build_all_panels()
                ),
            ])

        return dash.no_update, dash.no_update

    @app.callback(
        [Output('variables-dropdown', 'value'),
         Output('variables-info-markdown', 'children')],
        [Input('interpretabilite-button', 'n_clicks'),
         Input('performance-button', 'n_clicks')],
        [State('variables-dropdown', 'value')]
    )
    def update_dropdown_variables(interpretabilite_n_clicks, performance_n_clicks, existing_values):
        ctx = dash.callback_context
        info_text = ''

        if not ctx.triggered:
            button_id = 'No clicks yet'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'interpretabilite-button':
            new_values = ['REGION_RATING_CLIENT_W_CITY', 'DAYS_CREDIT_ENDDATE', 'RATE_DOWN_PAYMENT',
                          'AMT_PAYMENT', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE']
            info_text = INFO_VAR_INT

        elif button_id == 'performance-button':
            new_values = ['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM', 'EXT_SOURCE_2',
                          'EXT_SOURCE_1', 'EXT_SOURCE_3', 'NAME_INCOME_TYPE',
                          "DAYS_EMPLOYED"]
            info_text = INFO_VAR_PERF

        else:
            new_values = existing_values

        return new_values, info_text
    @app.callback(
        Output('stability-graph', 'figure'),
        [Input('stability-dropdown', 'value')]
    )
    def update_graph(selected_variable):
        fig = plot_stability_plotly(selected_variable)
        return fig
