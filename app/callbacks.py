import pickle
import sys

sys.path.append('./script')

from dash.dependencies import Input, Output, State, ALL
import dash
from dash import dcc

from builders import build_logit_model, build_xgboost_model, build_both_model, build_analyse_panel, chatbot
from data_preparation import *
from plot_utils import *
from plot_analyse import *
from app_utils import *

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
    
    #@app.callback(
   # Output("plot_correlation_matrix", "figure"),
   # [Input("launch-button-analyse", "n_clicks")],  # Trigger sur le bouton de lancement
   # [State("target-dropdown-analyse", "value"),  # Récupère la variable cible
   #  State("upload-data-analyse", "contents")]  # Récupère les données téléchargées
   # )
   # def update_correlation_matrix(n_clicks, target_variable, contents):
   #     if n_clicks > 0 and contents and target_variable:
   #         df, filename = parse_contents(contents)
   #         if isinstance(df, pd.DataFrame):
   #             return plot_correlation_matrix(df, target_variable)
   #         else:
   #             print("Le contenu téléchargé n'est pas un DataFrame valide.")
   #             return go.Figure()
   #     return go.Figure()  # Retourner une figure vide par défaut si les conditions ne sont pas remplies

    @app.callback([Output("Graph-Container", "children")],
                  [Input("target-dropdown-analyse", "value"),
                   Input("date-dropdown-analyse", "value"),
                   Input("launch-button-analyse", "n_clicks")])
    def display_graph(target, date, n_clicks):
        if n_clicks and n_clicks > 0:
            analyse.init_target(target)
            analyse.init_date(date)
            return build_analyse_panel()
        else :
            return dash.no_update

    @app.callback(
        [Output("stability-animated-graph", "figure"),
        Output("density-plot", "figure"),
        Output("missing-values-plot", "figure"),
        Output("plot_correlation_matrix", "figure")],  # Assurez-vous qu'il y a 4 sorties
        [Input("plot-stability-dropdown", "value"),
        Input("target-dropdown-analyse", "value")]  # Deux inputs
    )
    def update_graph(selected_variable, target_variable):
        if not selected_variable or not target_variable:
            return [go.Figure() for _ in range(4)]  # Retourne 4 figures vides si les variables ne sont pas définies

        fig = plot_stability_plotly_analyse(selected_variable)
        fig_d = plot_marginal_density(selected_variable)
        fig_m = missing_values()
        fig_c = plot_correlation_matrix(target_variable)  # Assurez-vous que cette fonction attend les bons paramètres

        return [fig, fig_d, fig_m, fig_c]  # Retourner une liste de 4 figures




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
         Output('Control-chart-tab', 'children'),
         Output("chat-tab", "style"),
         Output("chat-tab", "children"),
         Output("Control-chart-tab-2", "style"),
         Output("Control-chart-tab-2", "children")],
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
                model_classique.init_model('logit')
                model_classique.init_data(train_prepared, dataprep.discretizer.intervalles_dic, dataprep.target, dataprep.date)
                model_classique.run_model()
                model_classique.get_grid_score(dataprep.train, dataprep.target)
                model_classique.get_segmentation(dataprep.target)
                model_classique.get_default_proba(dataprep.target, dataprep.date)
                dataprep.init_model_name('logit')

                return ('tab2', {"display": "flex"}, "loaded", build_logit_model(), {"display": "flex"}, chatbot(), {"display": "none"}, dash.no_update)

            elif model_choice == 'XGBoost':
                model_challenger.init_model('xgb')
                model_challenger.init_data(train_prepared, dataprep.discretizer.intervalles_dic, dataprep.target, dataprep.date)
                model_challenger.run_model()
                model_challenger.get_grid_score(dataprep.train, dataprep.target)
                model_challenger.get_segmentation(dataprep.target)
                model_challenger.get_default_proba(dataprep.target, dataprep.date)
                dataprep.init_model_name('xgb')
                return ('tab3', {"display": "none"}, "loaded", dash.no_update, {"display": "flex"}, chatbot(), {"display": "flex"},  build_xgboost_model())

            elif model_choice == "both":
                model_classique.init_model('logit')
                model_classique.init_data(train_prepared, dataprep.discretizer.intervalles_dic, dataprep.target, dataprep.date)
                model_classique.run_model()
                model_classique.get_grid_score(dataprep.train, dataprep.target)
                model_classique.get_segmentation(dataprep.target)
                model_classique.get_default_proba(dataprep.target, dataprep.date)

                model_challenger.init_model('xgb')
                model_challenger.init_data(train_prepared, dataprep.discretizer.intervalles_dic, dataprep.target, dataprep.date)
                model_challenger.run_model()
                model_challenger.get_grid_score(dataprep.train, dataprep.target)
                model_challenger.get_segmentation(dataprep.target)
                model_challenger.get_default_proba(dataprep.target, dataprep.date)

                dataprep.init_model_name('logit')

                logit_layout, xgb_layout = build_both_model()
                return ('tab2', {"display": "flex"}, "loaded", logit_layout, {"display": "flex"}, chatbot(), {"display": "flex"}, xgb_layout)



        return dash.no_update, {"display": "none"}, dash.no_update, dash.no_update,  {"display": "none"}, dash.no_update, {"display": "none"}, dash.no_update

    @app.callback(
        Output('loading-div', 'style'),
        [Input('launch-button', 'n_clicks'),
         Input("variables-dropdown", 'value'),
         Input('model-choice', 'value')],
        [State('loading-div', 'style')],
        prevent_initial_call=True
    )
    def toggle_loading_div(n_clicks, features, model_choice, current_style):
        if n_clicks and n_clicks > 0:
            if model_choice is None or features is None:
                return dash.no_update
            return {'display': 'block'}
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
        [Output('class-display-clas', 'figure', allow_duplicate=True),  # Mise à jour de la figure du graphique
         Output('table-id-clas', 'data')],
        [Input('breaks-slider-clas', 'value'),
         Input('graph-type-selector-clas', 'value')],
        prevent_initial_call = True
    )
    def update_breaks_graph(breaks, graph_type):
        if breaks is not None :
            model_classique.update_segmentation(breaks, dataprep.target)

            if graph_type == 'gini':
                fig = create_gini_figure(model_classique)
            elif graph_type == 'taux':
                fig = create_stability_figure(model_classique)

            table_data = round(model_classique.resultats, 2).to_dict('records')

            return fig, table_data

    @app.callback(
        [Output('class-display-chal', 'figure', allow_duplicate=True),  # Mise à jour de la figure du graphique
         Output('table-id-chal', 'data')],
        [Input('breaks-slider-chal', 'value'),
         Input('graph-type-selector-chal', 'value')],
        prevent_initial_call = True
    )
    def update_breaks_graph(breaks, graph_type):
        if breaks is not None :
            model_challenger.update_segmentation(breaks, dataprep.target)

            if graph_type == 'gini':
                fig = create_gini_figure(model_challenger)
            elif graph_type == 'taux':
                fig = create_stability_figure(model_challenger)

            table_data = round(model_challenger.resultats, 2).to_dict('records')

            return fig, table_data



    @app.callback(
        [Output('stability-graph-clas', 'figure'),
         Output('histo-graph-clas', 'figure')],
        [Input('stability-dropdown-clas', 'value')]
    )
    def update_graph(selected_variable):
        fig_stab = plot_stability_plotly(selected_variable)
        fig_hist = plot_hist(selected_variable)
        return [fig_stab, fig_hist]

    @app.callback(
        [Output('stability-graph-chal', 'figure'),
         Output('histo-graph-chal', 'figure')],
        [Input('stability-dropdown-chal', 'value')]
    )
    def update_graph(selected_variable):
        fig_stab = plot_stability_plotly(selected_variable)
        fig_hist = plot_hist(selected_variable)
        return [fig_stab, fig_hist]

    @app.callback(
        Output('class-display-clas', 'figure'),
        [Input('graph-type-selector-clas', 'value')]
    )
    def update_graph(selected_graph):
        if selected_graph == 'gini':
            fig = create_gini_figure(model_classique)
        elif selected_graph == 'taux':
            fig = create_stability_figure(model_classique)
        return fig

    @app.callback(
        Output('class-display-chal', 'figure'),
        [Input('graph-type-selector-chal', 'value')]
    )
    def update_graph(selected_graph):
        if selected_graph == 'gini':
            fig = create_gini_figure(model_challenger)
        elif selected_graph == 'taux':
            fig = create_stability_figure(model_challenger)
        return fig

    @app.callback(
        Output("download-df-score", "data"),
        Input("btn_df_score", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_df_score(n_clicks):
        if dataprep.model_name == "logit" :
            model = model_classique
        elif dataprep.model_name == "xgb":
            model = model_challenger

        return dcc.send_data_frame(model.df_score.to_csv, "data_score.csv", index=False)

    @app.callback(
        Output("download-grille-score", "data"),
        Input("btn_grille_score", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_grille_score(n_clicks):
        if dataprep.model_name == "logit" :
            model = model_classique
        elif dataprep.model_name == "xgb":
            model = model_challenger

        return dcc.send_data_frame(model.grid_score.to_csv, "grille_score.csv", index=False)

    @app.callback(
        Output("download-model", "data"),
        Input("btn_model", "n_clicks"),
        prevent_initial_call=True
    )
    def download_model(n_clicks):
        if dataprep.model_name == "logit" :
            model = model_classique
        elif dataprep.model_name == "xgb":
            model = model_challenger

        serialized_model = pickle.dumps(model.model)
        return dcc.send_bytes(serialized_model, "model.pickle")

    ####################################### CHATBOT ########################################
    @app.callback(
        Output('dynamic-radioitems-container', 'children'),
        [Input({'type': 'dynamic-radioitems', 'index': ALL}, 'value')],
        [State('dynamic-radioitems-container', 'children')]
    )
    def add_radioitems(values, children):
        if dataprep.model_name == "logit" :
            model = model_classique
        elif dataprep.model_name == "xgb":
            model = model_challenger

        if not values or None in values:
            return dash.no_update

        df = model.df_score
        dropdown_columns = df.columns.difference(['Score_ind', 'Classes', dataprep.target,dataprep.date,
                                                  "date_trimestrielle"]).tolist()

        next_index = len(values)
        if next_index < len(dropdown_columns):
            new_element = html.Div([
                html.Div([
                    html.Label(f'Pour la variable {dropdown_columns[next_index]}:', className='label-inline message-label'),
                ], className='message-container'),
                html.Div([
                    dcc.RadioItems(
                        id={'type': 'dynamic-radioitems', 'index': next_index},
                        options=[{'label': str(v), 'value': v} for v in df[dropdown_columns[next_index]].dropna().unique()],
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'},  # Espacement et alignement horizontal
                        className='radio-inline selection-radio'
                    ),
                ], className='radioitems-container', style={'background-color': '8B0000', 'border-radius': '20px', 'color': 'white'}),
            ], className='form-input row', style={'margin-bottom': '50px'})
            children.append(new_element)
        return children


    @app.callback(
        Output('score-ind-result', 'children'),
        [Input({'type': 'dynamic-radioitems', 'index': ALL}, 'value')],
        prevent_initial_call=True
    )
    def update_score_ind(dropdown_values):
        if None in dropdown_values:
            return dash.no_update  # Retourne dash.no_update si toutes les sélections ne sont pas complétées.

        df = model_classique.df_score if dataprep.model_name == "logit" else model_challenger.df_score
        dropdown_columns = df.columns.difference(['Score_ind', 'Classes', dataprep.target, dataprep.date, "date_trimestrielle"]).tolist()

        if len(dropdown_values) < len(dropdown_columns):
            return dash.no_update

        model = model_classique if dataprep.model_name == "logit" else model_challenger
        filtered_df = df.copy()
        for column, value in zip(dropdown_columns, dropdown_values):
            if value is not None:
                filtered_df = filtered_df[filtered_df[column] == value]

        mean_score_ind = filtered_df['Score_ind'].mean() if not filtered_df.empty else None
        mean_classes = int(filtered_df['Classes'].mean()) if not filtered_df.empty else None

        message_lines = [
            f"Au vu de vos choix, votre score est {mean_score_ind:.2f}, vous êtes donc dans la classe {mean_classes}."
        ]

        if mean_classes > 5:
            message_lines.append("Un crédit vous sera octroyé.")
        elif 3 >= mean_classes >= 5:
            message_lines.append("Un crédit vous sera octroyé, mais avec un taux d'intérêt élevé.")
        else:
            message_lines.append("Malheureusement, aucun crédit ne vous sera octroyé.")

        message_divs = [html.Div(line, className='message-line') for line in message_lines]

        return html.Div([
            html.Div(message_divs, className='message-container', style={
                'background-color': '#007BFF', 'border-radius': '20px', 'color': 'white',
                'margin-bottom': '50px', 'padding': '10px 20px', 'font-size': '20px'
            }),
        ], className='form-input row')