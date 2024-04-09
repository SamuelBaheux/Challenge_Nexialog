import pickle
import sys

sys.path.append('./script')

from dash.dependencies import Input, Output, State
import dash
from dash import dcc

from builders import build_all_panels
from data_preparation import *
from plot_utils import *
from plot_analyse import *
from app_utils import *
from vars import statement_list

def register_callbacks(app):

    ####################################### ANALYSE ########################################

    @app.callback([Output('output-data-upload-analyse', 'children'),
                   Output('target-dropdown-analyse', 'options'),
                   Output('date-dropdown-analyse', 'options')],
                  [Input('upload-data-analyse', 'contents')],
                  [State('upload-data-analyse', 'filename')])
    def update_output(contents, filename):
        if contents is not None:
            df, filename = parse_contents(contents, filename)
            if isinstance(df, pd.DataFrame):
                analyse.init_data(df)
                target_options = analyse.get_features()
                date_options = analyse.get_features()

                return [html.Div(html.H5(f'Le fichier {filename} a été téléchargé avec succès'))], target_options, date_options
        else:
            return dash.no_update,[], []

    @app.callback([Output("Graph-Container", "children")],
                  [Input("launch-button-analyse", "n_clicks")])
    def display_graph(n_clicks):
        if n_clicks and n_clicks > 0:
            return [
                dcc.Graph(figure=missing_values())
                # Tu rajoutes ici les différents graphiques que tu fais
                # Si tu veux faire des styles différent (taille des graphs, disposition, ...) tu peux rajouter des
                # html.Div dans cette liste
                # tu peux rajouter tout ce que tu veux dans la liste (des html.Label, ...)
            ]
        else :
            return dash.no_update


    ####################################### MODÉLISATION ########################################
    @app.callback([Output('output-data-upload', 'children'),
                   Output('variables-dropdown', 'options'),
                   Output('target-dropdown', 'options'),
                   Output('date-dropdown', 'options'),
                   Output('predefined_vars_button', 'style')],
                  [Input('upload-data', 'contents')],
                  [State('upload-data', 'filename')])
    def update_output(list_of_contents, list_of_names):
        if list_of_contents is None or not list_of_contents:
            return [html.Div(''), [], [], [], {'display': 'None'}]

        if not isinstance(list_of_contents, list):
            list_of_contents = [list_of_contents]
        if not isinstance(list_of_names, list):
            list_of_names = [list_of_names]

        children = []
        options = []
        target_options = []
        date_options = []

        for c, n in zip(list_of_contents, list_of_names):
            results = parse_contents(c, n)
            if results is not None:
                dataprep.initialize_df(results[0])
                options = dataprep.get_features()
                target_options = dataprep.get_features()
                date_options = dataprep.get_features()
                children = [html.Div(html.H5(f'Le fichier {results[1]} a été téléchargé avec succès'))]

        if list_of_names[0] == "application_train_vf.csv" :
            return children, options, target_options, date_options, {'display': 'flex'}
        else :
            return children, options, target_options, date_options, {'display': 'None'}

    @app.callback(
        Output('loading-statement', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def check_list_changes(n):
        return(statement_list[-1])

    @app.callback(Output('hidden-div', 'children'),
                  [Input('target-dropdown', 'value')])
    def update_target(target_selected):
        if target_selected is not None :
            dataprep.init_target(target_selected)
            return ''

    @app.callback(Output('hidden-div1', 'children'),
                  [Input('date-dropdown', 'value')])
    def update_target(date_selected):
        if date_selected is not None :
            dataprep.init_date(date_selected)
            return ''


    @app.callback(
        [Output('app-tabs', 'value'),
         Output('Control-chart-tab', 'style'),
         Output("loading-output", "children"),
         Output('Control-chart-tab', 'children')],
        [Input('launch-button', 'n_clicks'),
         Input("variables-dropdown", 'value'),
         Input('model-choice', 'value')],
        prevent_initial_call=True)
    def update_result(n_clicks, features, model_choice):
        if n_clicks and n_clicks > 0:
            if model_choice is None or features is None:
                return dash.no_update

            print("Initialisation des données")
            dataprep.initialize_data(features)
            train_prepared = dataprep.get_prepared_data()

            print("Entraînement du modèle")
            if model_choice == 'Logit':
                model.init_model('logit')
                model.init_data(train_prepared, dataprep.discretizer.intervalles_dic, dataprep.target, dataprep.date)
                model.run_model()

            elif model_choice == 'XGBoost':
                model.init_model('xgb')
                model.init_data(train_prepared, dataprep.discretizer.intervalles_dic, dataprep.target, dataprep.date)
                model.run_model()

            return ('tab2', {"display": "flex"}, "loaded", build_all_panels())

        return dash.no_update, {"display": "none"}, dash.no_update, dash.no_update

    @app.callback(
        Output('loading-div', 'style'),
        [Input('launch-button', 'n_clicks'),
         Input("variables-dropdown", 'value'),
         Input('model-choice', 'value')],
        [State('loading-div', 'style')],  # Utilisez l'état actuel pour conditionner la mise à jour
        prevent_initial_call=True
    )
    def toggle_loading_div(n_clicks, features, model_choice, current_style):
        if n_clicks and n_clicks > 0:
            if model_choice is None or features is None:
                # Si les entrées ne sont pas valides, ne changez pas le style
                return dash.no_update
            # Si le bouton est cliqué et les entrées sont valides, affichez loading-div
            return {'display': 'block'}
        # Dans d'autres cas, gardez loading-div caché
        return {'display': 'none'}

    @app.callback(
        Output('variables-dropdown', 'value'),
        [Input('interpretabilite-button', 'n_clicks'),
         Input('performance-button', 'n_clicks')],
        [State('variables-dropdown', 'value')]
    )
    def update_dropdown_variables(interpretabilite_n_clicks, performance_n_clicks, existing_values):
        ctx = dash.callback_context

        if not ctx.triggered:
            button_id = 'No clicks yet'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'interpretabilite-button':
            new_values = ['REGION_RATING_CLIENT_W_CITY', 'DAYS_CREDIT_ENDDATE', 'RATE_DOWN_PAYMENT',
                          'AMT_PAYMENT', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE']

        elif button_id == 'performance-button':
            new_values = ['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM', 'EXT_SOURCE_2',
                          'EXT_SOURCE_1', 'EXT_SOURCE_3', 'NAME_INCOME_TYPE',
                          "DAYS_EMPLOYED"]

        else:
            new_values = existing_values

        return new_values

    @app.callback(
        Output('variables-info-markdown', 'children'),
        [Input('variables-dropdown', 'value')]
    )
    def update_dropdown_variables(selected_features):
        if selected_features is not None :
            info = "**Détail des variables choisies pour la modélisation :**\n"
            for features in selected_features :
                desc = dictionnaire[dictionnaire['Row'] == features]['Description']
                if len(desc) > 0 :
                    info += f"- **{features}** : {desc.values[0]}\n"
                else :
                    info += f"- {features}\n"

            return info

    @app.callback(
        [Output('class-display', 'figure', allow_duplicate=True),  # Mise à jour de la figure du graphique
         Output('table-id', 'data')],
        [Input('breaks-slider', 'value'),
         Input('graph-type-selector', 'value')],
        prevent_initial_call = True
    )
    def update_breaks_graph(breaks, graph_type):
        if breaks is not None :
            model.update_segmentation(breaks, dataprep.target)

            if graph_type == 'gini':
                fig = create_gini_figure()
            elif graph_type == 'taux':
                fig = create_stability_figure()

            table_data = round(model.resultats, 2).to_dict('records')

            return fig, table_data


    @app.callback(
        [Output('stability-graph', 'figure'),
         Output('histo-graph', 'figure')],
        [Input('stability-dropdown', 'value')]
    )
    def update_graph(selected_variable):
        fig_stab = plot_stability_plotly(selected_variable)
        fig_hist = plot_hist(selected_variable)
        return [fig_stab, fig_hist]

    @app.callback(
        Output('class-display', 'figure'),
        [Input('graph-type-selector', 'value')]
    )
    def update_graph(selected_graph):
        if selected_graph == 'gini':
            fig = create_gini_figure()
        elif selected_graph == 'taux':
            fig = create_stability_figure()
        return fig

    @app.callback(
        Output("download-df-score", "data"),
        Input("btn_df_score", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_df_score(n_clicks):
        return dcc.send_data_frame(model.df_score.to_csv, "data_score.csv", index=False)

    @app.callback(
        Output("download-grille-score", "data"),
        Input("btn_grille_score", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_grille_score(n_clicks):
        return dcc.send_data_frame(model.grid_score.to_csv, "grille_score.csv", index=False)

    @app.callback(
        Output("download-model", "data"),
        Input("btn_model", "n_clicks"),
        prevent_initial_call=True
    )
    def download_model(n_clicks):
        serialized_model = pickle.dumps(model.model)
        return dcc.send_bytes(serialized_model, "model.pickle")
