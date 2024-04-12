import pickle
import sys

sys.path.append('./script')

from dash.dependencies import Input, Output, State, MATCH, ALL
import dash

import json
from dash import dcc

from builders import build_all_panels
from data_preparation import *
from plot_utils import *
from app_utils import *
from vars import statement_list


import pandas as pd

def register_callbacks(app):
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
        Output('Chatbot-tab', 'style'),  # Ajout de cet Output pour contrôler la visibilité de l'onglet chatbot
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

            return ('tab2', {"display": "flex"}, {"display": "flex"}, "loaded", build_all_panels())  # Rendez l'onglet chatbot visible
        return dash.no_update, {"display": "none"}, {"display": "none"}, dash.no_update, dash.no_update


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

    ######## chatbot
    @app.callback(
        Output('dynamic-dropdown-container', 'children'),
        [Input({'type': 'dynamic-dropdown', 'index': ALL}, 'value')],
        [State('dynamic-dropdown-container', 'children')]
    )
    def add_dropdown(values, children):
        if not values or None in values:
            return dash.no_update
        df = pd.read_csv("/Users/jinzhou/Cours_M2/S2/Challenge_Nexialog/datas/df_segmentation.csv")
        df = df[['REGION_RATING_CLIENT_W_CITY', 'DAYS_CREDIT_ENDDATE_disc_int',
                "RATE_DOWN_PAYMENT_disc_int",
                "AMT_PAYMENT_disc_int", "NAME_INCOME_TYPE_discret",
                "OCCUPATION_TYPE_discret", 'Score_ind', "Classes"]]
        dropdown_columns = df.columns.difference(['Score_ind', 'Classes']).tolist()

        next_index = len(values)
        if next_index < len(dropdown_columns):
            new_element = html.Div([
                html.Div([
                    html.Label(f'Pour la variable {dropdown_columns[next_index]}:', className='label-inline message-label'),
                ], className='message-container'),
                html.Div([
                    dcc.Dropdown(
                        id={'type': 'dynamic-dropdown', 'index': next_index},
                        options=[{'label': str(v), 'value': v} for v in df[dropdown_columns[next_index]].dropna().unique()],
                        placeholder="Sélectionnez...",
                        className='dropdown-inline selection-dropdown'
                    ),
                ], className='dropdown-container'),
            ], className='form-input row', style={'margin-bottom': '50px'})
            children.append(new_element)

        button_exists = any(isinstance(child, html.Button) and child.id == 'launch-chatbot-modeling' for child in children)

        # Si tous les dropdowns sont affichés et que le bouton n'existe pas, ajoutez le bouton
        if next_index == len(dropdown_columns) and not button_exists:
            children.append(html.Button('Voir votre octroi de crédit', id='launch-chatbot-modeling', n_clicks=0, className='launch-button', style={'margin-top': '20px', 'display': 'block'}))

        return children

    @app.callback(
        Output('score-ind-result', 'children'),
        [Input('launch-chatbot-modeling', 'n_clicks')],
        [State({'type': 'dynamic-dropdown', 'index': ALL}, 'value')]
    )

    def update_score_ind(n_clicks, dropdown_values):
        df = pd.read_csv("/Users/jinzhou/Cours_M2/S2/Challenge_Nexialog/datas/df_segmentation.csv", index_col=[0])
        df = df[['REGION_RATING_CLIENT_W_CITY', 'DAYS_CREDIT_ENDDATE_disc_int',
                 "RATE_DOWN_PAYMENT_disc_int", "AMT_PAYMENT_disc_int",
                 "NAME_INCOME_TYPE_discret", "OCCUPATION_TYPE_discret",
                 'Score_ind', "Classes"]]

        dropdown_columns = df.columns.difference(
            ['Score_ind', 'Classes']).tolist()

        if n_clicks > 0 and None not in dropdown_values:
            if None in dropdown_values:
                return "Please complete all selections before submitting."

            # Filter DataFrame based on dropdown selections
            filtered_df = df.copy()
            for column, value in zip(dropdown_columns, dropdown_values):
                if value is not None:
                    filtered_df = filtered_df[filtered_df[column] == value]
                # # print(
                # #     f"Column: {column}, Value: {value}, Unique values in DF: {
                # #         df[column].unique()}")

                # if value is not None:
                #     filtered_df = filtered_df[filtered_df[column] == value]
                #     print("Callback déclenché")

            mean_score_ind = filtered_df['Score_ind'].mean() if not filtered_df.empty else None
            mean_classes = int(filtered_df['Classes'].mean()) if not filtered_df.empty else None

            print(mean_score_ind, mean_classes)
            if mean_classes < 3:
                message = f"""
            Votre score est de : {mean_score_ind:.2f} \n
            Vous êtes dans la classe {mean_classes} \n
            Un crédit vous sera octroyé
            """
            elif 5 >= mean_classes >= 3:
                message = f"""
                Votre score est de : {mean_score_ind:.2f} \n
                Vous êtes dans la classe {mean_classes} \n
                Un crédit vous sera octroyé, mais avec un taux d'intérêt élevé
                """
            else:
                message = f"""
                Votre score est de : {mean_score_ind:.2f} \n
                Vous êtes dans la classe {mean_classes} \n
                Aucun crédit ne vous sera octroyé
                """
            return dcc.Markdown(message, style={'margin-top': '20px',
                                                "color": "#ffffff",
                                                'font-weight': 'bold',
                                                "font-size": "20px"})
        return "Veuillez faire toutes les sélections avant de soumettre."
